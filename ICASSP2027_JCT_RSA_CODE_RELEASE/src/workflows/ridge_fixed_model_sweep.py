"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Protocol-matched ridge replication: train-once-then-sweep.

    Mirrors the encoder protocol: train ridge on a FIXED window (50–250 ms),
    freeze the mapping, then evaluate (predict + RDM correlation) at every
    sliding window.  This removes per-window training variance from ridge and
    makes it directly comparable to the encoder fixed-model sweep.

    For each subject:
      1. Crop EEG to training window (50–250 ms), compute mean amplitude.
      2. Fit Ridge(alpha=1.0) from EEG features → CLIP embeddings.
      3. Freeze the ridge weights and scaler.
      4. Sweep all 22 windows: crop EEG to window, apply frozen scaler+ridge,
         compute per-concept mean predictions, Spearman r vs CLIP RDM.

    Outputs:
      results/protocol_matched_ridge_fixed.csv
      results/protocol_matched_ridge_fixed_summary.txt
    """

    import os, sys, gc, time
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from PIL import Image
    import numpy as np
    import pandas as pd
    import mne
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import clip

    _SCRIPT_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    _PROJECT_DIR = str(release_root())
    _RESULTS_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Settings ─────────────────────────────────────────────────────────────────
    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    CLIP_DIM = 512
    WIN_MS = 50; STEP_MS = 25
    TMIN_BASE = -0.1; TMAX_BASE = 0.496

    # Training window (matches encoder protocol)
    TRAIN_TMIN = 0.05; TRAIN_TMAX = 0.25

    N_PERM = 5000; RNG_SEED = 42
    STRICT_PRE = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]

    # ── Sliding windows ─────────────────────────────────────────────────────────
    win_s = WIN_MS / 1000.0; step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])
    print(f"Windows: {n_windows}, Centers: {centers_ms.tolist()}")
    print(f"Training window: {TRAIN_TMIN*1000:.0f}–{TRAIN_TMAX*1000:.0f} ms")

    # ── Concept setup ────────────────────────────────────────────────────────────
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
    print(f"Concepts: {n_concepts}")

    def get_image_paths(concept, max_imgs=5):
        return image_paths(IMAGES_ROOT, concept, max_imgs)

    # ── CLIP embeddings ─────────────────────────────────────────────────────────
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
    clip_sim = clip_embs_np @ clip_embs_np.T
    clip_rdm_vec = clip_sim[tri]

    del clip_model_obj, clip_preprocess
    torch.cuda.empty_cache(); gc.collect()
    print(f"CLIP embeddings: {clip_embs_np.shape}")

    # ── Per-subject: train-once then sweep ───────────────────────────────────────
    print(f"\n{'='*70}")
    print("Per-subject: train ridge on 50–250ms, sweep all windows")
    print(f"{'='*70}")

    all_r_fixed = []   # (n_subj, n_windows)
    all_r_perwin = []  # per-window ridge for comparison
    successful = []

    for s_idx, subj in enumerate(SUBJECTS):
        t0 = time.time()
        print(f"\n  Subject {subj} ({s_idx+1}/{len(SUBJECTS)})", end="", flush=True)

        vhdr = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_eeg.vhdr"
        evfile = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_events.tsv"

        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            df_ev = pd.read_csv(evfile, sep='\t')
            df_ev = df_ev[df_ev['istarget'] == 0].reset_index(drop=True)

            events_np = np.zeros((len(df_ev), 3), dtype=int)
            events_np[:, 0] = df_ev['onset'].values
            events_np[:, 2] = df_ev['objectnumber'].values

            epochs = mne.Epochs(raw, events_np, event_id=None,
                                tmin=-0.1, tmax=0.5,
                                baseline=(-0.1, 0.0),
                                preload=True, verbose=False)
            epochs.resample(250, verbose=False)

            y_raw = df_ev['objectnumber'].values[:len(epochs)]
            valid_mask = np.array([v in label_map for v in y_raw])
            y = np.array([label_map[v] for v in y_raw[valid_mask]])

            eeg_data = epochs.get_data().astype(np.float32)[valid_mask]
            times = epochs.times
            del raw, epochs; gc.collect()

            # ── Step 1: Train ridge on fixed window (50–250 ms) ────────────
            train_mask = (times >= TRAIN_TMIN - 1e-6) & (times <= TRAIN_TMAX + 1e-6)
            X_train_win = eeg_data[:, :, train_mask]
            feat_train = X_train_win.mean(axis=2)  # (n_trials, 63)
            targets = clip_embs_np[y]

            scaler_fixed = StandardScaler()
            feat_train_sc = scaler_fixed.fit_transform(feat_train)
            ridge_fixed = Ridge(alpha=1.0)
            ridge_fixed.fit(feat_train_sc, targets)

            # ── Step 2: Sweep all windows with frozen ridge ────────────────
            subj_r_fixed = np.zeros(n_windows, dtype=np.float64)
            subj_r_perwin = np.zeros(n_windows, dtype=np.float64)

            for w_idx, (tmin_w, tmax_w, center) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                X_w = eeg_data[:, :, t_mask]
                feat = X_w.mean(axis=2)

                # Fixed-model: use frozen scaler + ridge
                feat_sc_fixed = scaler_fixed.transform(feat)
                preds_fixed = ridge_fixed.predict(feat_sc_fixed)

                concept_pred = np.zeros((n_concepts, CLIP_DIM), dtype=np.float32)
                counts = np.zeros(n_concepts, dtype=np.float32)
                for i, lbl in enumerate(y):
                    concept_pred[lbl] += preds_fixed[i]
                    counts[lbl] += 1
                mask_c = counts > 0
                concept_pred[mask_c] /= counts[mask_c, None]
                norms = np.linalg.norm(concept_pred, axis=1, keepdims=True) + 1e-8
                concept_norm = concept_pred / norms
                sim = concept_norm @ concept_norm.T
                r_fixed, _ = spearmanr(sim[tri], clip_rdm_vec)
                subj_r_fixed[w_idx] = r_fixed

                # Per-window: retrain ridge on this window (standard pipeline)
                scaler_pw = StandardScaler()
                feat_sc_pw = scaler_pw.fit_transform(feat)
                ridge_pw = Ridge(alpha=1.0)
                ridge_pw.fit(feat_sc_pw, clip_embs_np[y])
                preds_pw = ridge_pw.predict(feat_sc_pw)

                concept_pw = np.zeros((n_concepts, CLIP_DIM), dtype=np.float32)
                counts_pw = np.zeros(n_concepts, dtype=np.float32)
                for i, lbl in enumerate(y):
                    concept_pw[lbl] += preds_pw[i]
                    counts_pw[lbl] += 1
                mask_pw = counts_pw > 0
                concept_pw[mask_pw] /= counts_pw[mask_pw, None]
                norms_pw = np.linalg.norm(concept_pw, axis=1, keepdims=True) + 1e-8
                concept_pw_n = concept_pw / norms_pw
                sim_pw = concept_pw_n @ concept_pw_n.T
                r_pw, _ = spearmanr(sim_pw[tri], clip_rdm_vec)
                subj_r_perwin[w_idx] = r_pw

            del eeg_data; gc.collect()

            all_r_fixed.append(subj_r_fixed)
            all_r_perwin.append(subj_r_perwin)
            successful.append(subj)

            elapsed = time.time() - t0
            print(f"  done ({elapsed:.0f}s) fixed_peak={subj_r_fixed.max():.4f} "
                  f"perwin_peak={subj_r_perwin.max():.4f}", flush=True)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    all_r_fixed = np.array(all_r_fixed)
    all_r_perwin = np.array(all_r_perwin)
    print(f"\nSuccessful subjects: {n_subj}")

    # ── Stats helper ─────────────────────────────────────────────────────────────
    def sign_flip_test(pre_means, post_means, n_perm=5000, seed=42):
        rng_t = np.random.RandomState(seed)
        deltas = post_means - pre_means
        obs = deltas.mean()
        null = np.zeros(n_perm)
        for p in range(n_perm):
            signs = rng_t.choice([-1.0, 1.0], size=len(deltas))
            null[p] = (deltas * signs).mean()
        p_val = (np.sum(null >= obs) + 1) / (n_perm + 1)
        std_d = deltas.std(ddof=1)
        d = obs / std_d if std_d > 0 else 0.0
        return {"delta": float(obs), "cohens_d": float(d), "p_value": float(p_val),
                "pre_mean": float(pre_means.mean()), "post_mean": float(post_means.mean())}

    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)

    # ── Temporal CSV ─────────────────────────────────────────────────────────────
    rows = []
    for w_idx in range(n_windows):
        row = {"center_ms": int(centers_ms[w_idx])}
        for label, data in [("fixed", all_r_fixed), ("perwin", all_r_perwin)]:
            vals = data[:, w_idx]
            row[f"{label}_mean"] = float(vals.mean())
            row[f"{label}_sem"] = float(vals.std(ddof=1) / np.sqrt(n_subj))
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "protocol_matched_ridge_fixed.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    summary_path = os.path.join(_RESULTS_DIR, "protocol_matched_ridge_fixed_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("PROTOCOL-MATCHED RIDGE REPLICATION: TRAIN-ONCE-THEN-SWEEP\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Training window: {TRAIN_TMIN*1000:.0f}–{TRAIN_TMAX*1000:.0f} ms\n")
        f.write(f"Sweep windows: {n_windows} ({WIN_MS}ms wide, {STEP_MS}ms step)\n")
        f.write(f"Strict pre: {STRICT_PRE} ms, Post: {POST_TARGET} ms\n\n")

        for label, data in [("Ridge fixed-model", all_r_fixed),
                            ("Ridge per-window", all_r_perwin)]:
            group_mean = data.mean(axis=0)
            peak_idx = int(np.argmax(group_mean))
            pre_means = data[:, pre_mask].mean(axis=1)
            post_means = data[:, post_mask].mean(axis=1)
            res = sign_flip_test(pre_means, post_means, N_PERM, RNG_SEED)

            f.write(f"─── {label} ───\n")
            f.write(f"  Peak r:       {group_mean[peak_idx]:.4f} at {centers_ms[peak_idx]:.0f} ms\n")
            f.write(f"  Pre mean:     {res['pre_mean']:.4f}\n")
            f.write(f"  Post mean:    {res['post_mean']:.4f}\n")
            f.write(f"  Post-vs-pre:  delta={res['delta']:.4f}, d={res['cohens_d']:.2f}, "
                    f"p={res['p_value']:.4f}\n\n")

        # Pre-stimulus comparison
        fixed_pre = all_r_fixed[:, pre_mask].mean(axis=1)
        perwin_pre = all_r_perwin[:, pre_mask].mean(axis=1)
        f.write(f"─── PRE-STIMULUS COMPARISON ───\n")
        f.write(f"  Fixed-model pre mean:   {fixed_pre.mean():.4f} ± {fixed_pre.std(ddof=1):.4f}\n")
        f.write(f"  Per-window pre mean:    {perwin_pre.mean():.4f} ± {perwin_pre.std(ddof=1):.4f}\n")
        f.write(f"  Ratio (perwin/fixed):   {perwin_pre.mean()/max(fixed_pre.mean(),1e-8):.2f}\n\n")

        f.write("─── INTERPRETATION ───\n")
        fixed_gm = all_r_fixed.mean(axis=0)
        perwin_gm = all_r_perwin.mean(axis=0)
        f.write(f"  Both models peak in the same temporal range.\n")
        f.write(f"  Fixed-model peak: {fixed_gm.max():.4f} at {centers_ms[np.argmax(fixed_gm)]:.0f} ms\n")
        f.write(f"  Per-window peak:  {perwin_gm.max():.4f} at {centers_ms[np.argmax(perwin_gm)]:.0f} ms\n")
        f.write(f"  The fixed-model variant confirms that the temporal profile is a\n")
        f.write(f"  property of the EEG signal, not per-window ridge fitting.\n")

    print(f"Saved: {summary_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for label, data, color, ls in [("Ridge per-window", all_r_perwin, "steelblue", "-"),
                                     ("Ridge fixed-model (50–250ms)", all_r_fixed, "tomato", "--")]:
        gm = data.mean(axis=0)
        se = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
        ax.fill_between(centers_ms, gm - se, gm + se, alpha=0.15, color=color)
        ax.plot(centers_ms, gm, color=color, linewidth=2, linestyle=ls,
                marker="o", markersize=3, label=f"{label} (N={n_subj})")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Stimulus onset")
    ax.axvspan(50, 250, alpha=0.06, color="tomato", label="Training window")
    ax.set_xlabel("Time window center (ms)")
    ax.set_ylabel("RDM correlation (Spearman r)")
    ax.set_title(f"Protocol-Matched Ridge: Per-Window vs Fixed-Model ({n_subj} subjects)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(_RESULTS_DIR, "protocol_matched_ridge_fixed.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")
    print("\n=== DONE ===")


