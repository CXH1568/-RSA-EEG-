"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Training-window robustness check for fixed-model ridge.

    Tests whether the fixed-model temporal concentration is sensitive to the
    choice of training window by comparing:
      50-200ms, 50-250ms (main), 75-225ms, 75-250ms, 100-250ms

    For each training window × subject: train ridge once, freeze, sweep all
    22 evaluation windows. Report group-level statistics.
    """

    import os, sys, gc, time
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from PIL import Image
    import numpy as np
    import pandas as pd
    import mne
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr, binomtest
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import clip

    _SCRIPT_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    _PROJECT_DIR = str(release_root())
    _RESULTS_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    CLIP_DIM = 512
    WIN_MS = 50; STEP_MS = 25
    TMIN_BASE = -0.1; TMAX_BASE = 0.496
    N_PERM = 5000; RNG_SEED = 42
    STRICT_PRE = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]

    TRAIN_WINDOWS = [
        (0.050, 0.200, "50-200ms"),
        (0.050, 0.250, "50-250ms"),
        (0.075, 0.225, "75-225ms"),
        (0.075, 0.250, "75-250ms"),
        (0.100, 0.250, "100-250ms"),
    ]

    # Eval windows
    win_s = WIN_MS / 1000.0; step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])
    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)

    # Concepts
    df_ref = pd.read_csv(f"{BASE}/sub-01/eeg/sub-01_task-rsvp_events.tsv", sep='\t')
    df_ref = df_ref[df_ref['istarget'] == 0].reset_index(drop=True)
    unique_labels = np.unique(df_ref['objectnumber'].values)
    label_map = {v: i for i, v in enumerate(unique_labels)}
    n_concepts = len(unique_labels)
    concept_by_label = {}
    for _, row in df_ref.iterrows():
        lbl = label_map.get(row['objectnumber'])
        if lbl is not None and lbl not in concept_by_label:
            concept_by_label[lbl] = row['object']
    concept_names = [concept_by_label[i] for i in range(n_concepts)]

    def get_image_paths(concept, max_imgs=5):
        return image_paths(IMAGES_ROOT, concept, max_imgs)

    # CLIP
    print("Computing CLIP embeddings...")
    clip_model_obj, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model_obj.eval()
    clip_embs = torch.zeros(n_concepts, CLIP_DIM)
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths: raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([clip_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            emb = clip_model_obj.encode_image(imgs).float()
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            clip_embs[i] = F.normalize(emb.unsqueeze(0), dim=-1)
    clip_embs_np = F.normalize(clip_embs, dim=-1).numpy()
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)
    clip_rdm_vec = (clip_embs_np @ clip_embs_np.T)[tri]
    del clip_model_obj, clip_preprocess; torch.cuda.empty_cache(); gc.collect()

    # Per-subject, per-training-window
    print(f"\n{'='*70}")
    print(f"Window robustness: {len(TRAIN_WINDOWS)} training windows × {len(SUBJECTS)} subjects")
    print(f"{'='*70}")

    # {tw_label: list of (n_windows,) arrays}
    all_results = {tw[2]: [] for tw in TRAIN_WINDOWS}
    successful = []

    for s_idx, subj in enumerate(SUBJECTS):
        t0 = time.time()
        print(f"  {subj} ({s_idx+1}/{len(SUBJECTS)})", end="", flush=True)
        vhdr = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_eeg.vhdr"
        evfile = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_events.tsv"
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            df_ev = pd.read_csv(evfile, sep='\t')
            df_ev = df_ev[df_ev['istarget'] == 0].reset_index(drop=True)
            events_np = np.zeros((len(df_ev), 3), dtype=int)
            events_np[:, 0] = df_ev['onset'].values
            events_np[:, 2] = df_ev['objectnumber'].values
            epochs = mne.Epochs(raw, events_np, tmin=-0.1, tmax=0.5,
                                baseline=(-0.1, 0.0), preload=True, verbose=False)
            epochs.resample(250, verbose=False)
            y_raw = df_ev['objectnumber'].values[:len(epochs)]
            valid_mask = np.array([v in label_map for v in y_raw])
            y = np.array([label_map[v] for v in y_raw[valid_mask]])
            eeg_data = epochs.get_data().astype(np.float32)[valid_mask]
            times = epochs.times
            del raw, epochs; gc.collect()

            for tw_min, tw_max, tw_label in TRAIN_WINDOWS:
                train_t = (times >= tw_min - 1e-6) & (times <= tw_max + 1e-6)
                feat_train = eeg_data[:, :, train_t].mean(axis=2)
                scaler = StandardScaler()
                feat_sc = scaler.fit_transform(feat_train)
                ridge = Ridge(alpha=1.0)
                ridge.fit(feat_sc, clip_embs_np[y])

                subj_r = np.zeros(n_windows)
                for w_idx, (tmin_w, tmax_w, _) in enumerate(windows):
                    t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                    feat = eeg_data[:, :, t_mask].mean(axis=2)
                    feat_s = scaler.transform(feat)
                    preds = ridge.predict(feat_s)
                    cp = np.zeros((n_concepts, CLIP_DIM), dtype=np.float32)
                    ct = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        cp[lbl] += preds[i]; ct[lbl] += 1
                    m = ct > 0; cp[m] /= ct[m, None]
                    norms = np.linalg.norm(cp, axis=1, keepdims=True) + 1e-8
                    cn = cp / norms
                    sim = cn @ cn.T
                    r, _ = spearmanr(sim[tri], clip_rdm_vec)
                    subj_r[w_idx] = r

                all_results[tw_label].append(subj_r)

            del eeg_data; gc.collect()
            successful.append(subj)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    for tw_label in all_results:
        all_results[tw_label] = np.array(all_results[tw_label])

    # Stats
    def sign_flip_test(vals, n_perm=5000, seed=42):
        rng_t = np.random.RandomState(seed)
        obs = vals.mean()
        null = np.zeros(n_perm)
        for p in range(n_perm):
            signs = rng_t.choice([-1.0, 1.0], size=len(vals))
            null[p] = (vals * signs).mean()
        p_val = (np.sum(null >= obs) + 1) / (n_perm + 1)
        d = obs / (vals.std(ddof=1) + 1e-12)
        return float(obs), float(d), float(p_val)

    rows = []
    for tw_min, tw_max, tw_label in TRAIN_WINDOWS:
        data = all_results[tw_label]
        gm = data.mean(axis=0)
        peak_idx = int(np.argmax(gm))
        pre_vals = data[:, pre_mask].mean(axis=1)
        post_vals = data[:, post_mask].mean(axis=1)
        deltas = post_vals - pre_vals
        delta_mean, delta_d, delta_p = sign_flip_test(deltas, N_PERM, RNG_SEED)
        n_pos = int((deltas > 0).sum())
        peak_mss = np.array([centers_ms[np.argmax(data[i])] for i in range(n_subj)])
        n_125_225 = int(((peak_mss >= 125) & (peak_mss <= 225)).sum())

        rows.append({
            "train_window": tw_label,
            "peak_r": float(gm[peak_idx]),
            "peak_ms": int(centers_ms[peak_idx]),
            "pre_mean": float(pre_vals.mean()),
            "post_mean": float(post_vals.mean()),
            "delta": delta_mean,
            "cohens_d": delta_d,
            "p_value": delta_p,
            "n_positive": n_pos,
            "pct_positive": n_pos / n_subj * 100,
            "median_peak_ms": float(np.median(peak_mss)),
            "peak_iqr_lo": float(np.percentile(peak_mss, 25)),
            "peak_iqr_hi": float(np.percentile(peak_mss, 75)),
            "n_peak_125_225": n_125_225,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "fixed_model_window_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Summary
    summary_path = os.path.join(_RESULTS_DIR, "fixed_model_window_robustness_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING-WINDOW ROBUSTNESS: FIXED-MODEL RIDGE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Subjects: {n_subj}\n\n")
        f.write(f"{'Window':>12s} {'Peak r':>8s} {'Peak ms':>8s} {'Pre':>8s} {'Post':>8s} "
                f"{'Delta':>8s} {'d':>6s} {'p':>8s} {'%pos':>6s} {'Med pk':>7s} {'125-225':>8s}\n")
        f.write("─" * 95 + "\n")
        for r in rows:
            f.write(f"{r['train_window']:>12s} {r['peak_r']:8.4f} {r['peak_ms']:8d} "
                    f"{r['pre_mean']:8.4f} {r['post_mean']:8.4f} {r['delta']:8.4f} "
                    f"{r['cohens_d']:6.2f} {r['p_value']:8.4f} {r['pct_positive']:5.0f}% "
                    f"{r['median_peak_ms']:7.0f} {r['n_peak_125_225']:4d}/{n_subj}\n")
    print(f"Saved: {summary_path}")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "tomato", "forestgreen", "darkorange", "purple"]
    for i, (tw_min, tw_max, tw_label) in enumerate(TRAIN_WINDOWS):
        data = all_results[tw_label]
        gm = data.mean(axis=0)
        se = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
        ls = "-" if tw_label == "50-250ms" else "--"
        lw = 2.5 if tw_label == "50-250ms" else 1.5
        ax.plot(centers_ms, gm, color=colors[i], linewidth=lw, linestyle=ls,
                marker="o", markersize=3, label=f"Train {tw_label}")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Evaluation window center (ms)")
    ax.set_ylabel("RDM correlation (Spearman r)")
    ax.set_title(f"Training-Window Robustness: Fixed-Model Ridge ({n_subj} subjects)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS_DIR, "fixed_model_window_robustness_plot.png"), dpi=150)
    print("Saved: fixed_model_window_robustness_plot.png")
    print("\n=== DONE ===")


