"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Subject-level robustness analysis for fixed-model ridge temporal effect.

    Reads the protocol_matched_ridge_fixed.csv (which has group summaries)
    and re-runs the fixed-model ridge to extract per-subject temporal curves,
    then computes:
      1. Per-subject post-vs-pre delta distribution
      2. Per-subject peak latency distribution
      3. Per-subject peak magnitude distribution
      4. Bootstrap CI, sign test, histograms

    Outputs:
      results/fixed_model_subject_robustness.csv
      results/fixed_model_subject_robustness_summary.txt
      results/fixed_model_peak_latency_hist.png
      results/fixed_model_delta_distribution.png
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
    TRAIN_TMIN = 0.05; TRAIN_TMAX = 0.25
    N_BOOT = 10000; RNG_SEED = 42

    STRICT_PRE = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]

    # Windows
    win_s = WIN_MS / 1000.0; step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])

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

    # CLIP embeddings
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

    # Per-subject fixed-model ridge
    print(f"\n{'='*60}")
    print("Per-subject fixed-model ridge sweep")
    print(f"{'='*60}")

    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)

    rows = []
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

            # Train on fixed window
            train_mask_t = (times >= TRAIN_TMIN - 1e-6) & (times <= TRAIN_TMAX + 1e-6)
            feat_train = eeg_data[:, :, train_mask_t].mean(axis=2)
            scaler = StandardScaler()
            feat_train_sc = scaler.fit_transform(feat_train)
            ridge = Ridge(alpha=1.0)
            ridge.fit(feat_train_sc, clip_embs_np[y])

            # Sweep
            subj_r = np.zeros(n_windows)
            for w_idx, (tmin_w, tmax_w, center) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_data[:, :, t_mask].mean(axis=2)
                feat_sc = scaler.transform(feat)
                preds = ridge.predict(feat_sc)
                concept_pred = np.zeros((n_concepts, CLIP_DIM), dtype=np.float32)
                counts = np.zeros(n_concepts, dtype=np.float32)
                for i, lbl in enumerate(y):
                    concept_pred[lbl] += preds[i]; counts[lbl] += 1
                m = counts > 0; concept_pred[m] /= counts[m, None]
                norms = np.linalg.norm(concept_pred, axis=1, keepdims=True) + 1e-8
                cn = concept_pred / norms
                sim = cn @ cn.T
                r, _ = spearmanr(sim[tri], clip_rdm_vec)
                subj_r[w_idx] = r

            del eeg_data; gc.collect()

            pre_mean = subj_r[pre_mask].mean()
            post_mean = subj_r[post_mask].mean()
            delta = post_mean - pre_mean
            peak_idx = int(np.argmax(subj_r))
            peak_r = subj_r[peak_idx]
            peak_ms = int(centers_ms[peak_idx])

            rows.append({
                "subject": subj,
                "pre_mean": pre_mean, "post_mean": post_mean, "delta": delta,
                "peak_r": peak_r, "peak_ms": peak_ms,
            })
            # Also store full curve
            for w_idx in range(n_windows):
                rows[-1][f"r_{int(centers_ms[w_idx])}ms"] = subj_r[w_idx]

            successful.append(subj)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.0f}s) delta={delta:.4f} peak={peak_r:.4f}@{peak_ms}ms", flush=True)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "fixed_model_subject_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Extract arrays
    deltas = df["delta"].values
    peak_rs = df["peak_r"].values
    peak_mss = df["peak_ms"].values

    # ── Summary statistics ───────────────────────────────────────────────────────
    n_positive = int((deltas > 0).sum())
    prop_positive = n_positive / n_subj

    # Sign test (binomial)
    sign_p = binomtest(n_positive, n_subj, 0.5, alternative='greater').pvalue

    # Bootstrap CI for mean delta
    rng_boot = np.random.RandomState(RNG_SEED)
    boot_means = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = rng_boot.choice(n_subj, size=n_subj, replace=True)
        boot_means[b] = deltas[idx].mean()
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    # Peak latency stats
    n_125_225 = int(((peak_mss >= 125) & (peak_mss <= 225)).sum())
    n_150_200 = int(((peak_mss >= 150) & (peak_mss <= 200)).sum())

    summary_path = os.path.join(_RESULTS_DIR, "fixed_model_subject_robustness_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("SUBJECT-LEVEL ROBUSTNESS: FIXED-MODEL RIDGE\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Training window: {TRAIN_TMIN*1000:.0f}–{TRAIN_TMAX*1000:.0f} ms\n")
        f.write(f"Pre interval: {STRICT_PRE} ms, Post interval: {POST_TARGET} ms\n\n")

        f.write("─── POST-VS-PRE DELTA DISTRIBUTION ───\n")
        f.write(f"  N positive:       {n_positive}/{n_subj} ({prop_positive*100:.0f}%)\n")
        f.write(f"  Sign test p:      {sign_p:.6f}\n")
        f.write(f"  Mean delta:       {deltas.mean():.4f}\n")
        f.write(f"  Bootstrap 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"  Median delta:     {np.median(deltas):.4f}\n")
        f.write(f"  IQR:              [{np.percentile(deltas, 25):.4f}, {np.percentile(deltas, 75):.4f}]\n")
        f.write(f"  Min:              {deltas.min():.4f}\n")
        f.write(f"  Max:              {deltas.max():.4f}\n\n")

        f.write("─── PEAK LATENCY DISTRIBUTION ───\n")
        f.write(f"  Median peak:      {np.median(peak_mss):.0f} ms\n")
        f.write(f"  IQR:              [{np.percentile(peak_mss, 25):.0f}, {np.percentile(peak_mss, 75):.0f}] ms\n")
        f.write(f"  Min:              {peak_mss.min():.0f} ms\n")
        f.write(f"  Max:              {peak_mss.max():.0f} ms\n")
        f.write(f"  In 125–225 ms:    {n_125_225}/{n_subj} ({n_125_225/n_subj*100:.0f}%)\n")
        f.write(f"  In 150–200 ms:    {n_150_200}/{n_subj} ({n_150_200/n_subj*100:.0f}%)\n\n")

        f.write("─── PEAK MAGNITUDE DISTRIBUTION ───\n")
        f.write(f"  Mean peak r:      {peak_rs.mean():.4f}\n")
        f.write(f"  Median peak r:    {np.median(peak_rs):.4f}\n")
        f.write(f"  IQR:              [{np.percentile(peak_rs, 25):.4f}, {np.percentile(peak_rs, 75):.4f}]\n")
        f.write(f"  Min:              {peak_rs.min():.4f}\n")
        f.write(f"  Max:              {peak_rs.max():.4f}\n")

    print(f"Saved: {summary_path}")

    # ── Plots ────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Delta distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deltas, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(deltas.mean(), color="gold", linestyle="-", linewidth=2,
               label=f"Mean={deltas.mean():.4f}")
    ax.axvline(np.median(deltas), color="orange", linestyle=":", linewidth=2,
               label=f"Median={np.median(deltas):.4f}")
    ax.set_xlabel("Post-vs-pre Δr (fixed-model ridge)")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Subject-Level Post-vs-Pre Delta Distribution (N={n_subj})\n"
                 f"{n_positive}/{n_subj} positive ({prop_positive*100:.0f}%), sign test p={sign_p:.4f}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS_DIR, "fixed_model_delta_distribution.png"), dpi=150)
    print("Saved: fixed_model_delta_distribution.png")

    # 2. Peak latency histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(centers_ms.min() - STEP_MS/2, centers_ms.max() + STEP_MS, STEP_MS)
    ax.hist(peak_mss, bins=bins, color="tomato", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(peak_mss), color="gold", linestyle="-", linewidth=2,
               label=f"Median={np.median(peak_mss):.0f} ms")
    ax.axvspan(125, 225, alpha=0.1, color="blue", label=f"125–225 ms ({n_125_225}/{n_subj})")
    ax.set_xlabel("Peak latency (ms)")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"Subject-Level Peak Latency Distribution (Fixed-Model Ridge, N={n_subj})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(_RESULTS_DIR, "fixed_model_peak_latency_hist.png"), dpi=150)
    print("Saved: fixed_model_peak_latency_hist.png")

    print("\n=== DONE ===")


