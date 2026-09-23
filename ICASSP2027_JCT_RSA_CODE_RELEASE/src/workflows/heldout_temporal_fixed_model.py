"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Held-out temporal sweep using the protocol-matched fixed-model ridge.

    This script answers: does the post-stimulus temporal concentration observed
    under the fixed-model ridge replicate on 654 concept-disjoint held-out concepts?

    Design (mirrors scripts/fixed_model_subject_robustness.py, but with a
    concept-disjoint split):
      - Same held-out split as scripts/heldout_uncertainty.py
        (seed=42, shuffle all 1854 concepts, first 1200 train / last 654 eval).
      - For each subject:
          1. Train Ridge ONCE on training-concept trials' mean amplitude in
             TRAIN_TMIN-TRAIN_TMAX ms (default 50-250).
          2. Freeze weights and scaler.
          3. Sweep 22 sliding windows (50 ms, 25 ms step, -100..+500 ms).
             At each window, evaluate on EVAL-concept trials only.
             Aggregate predictions concept-wise on the 654 eval concepts,
             compute Spearman r vs CLIP RDM on those 654 concepts.
      - Per subject: peak r / peak latency / pre mean / post mean / delta.
      - Group-level: sign-flip permutation test on delta, bootstrap CI,
        peak-latency distribution, fraction of subjects with post > pre.

    Outputs (under results/):
      heldout_temporal_fixed_model.csv           # per-subject per-window r
      heldout_temporal_subject_summary.csv       # per-subject summary
      heldout_temporal_fixed_model_summary.txt   # group summary
      heldout_temporal_fixed_model_plot.png      # group mean curve + per-subject
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

    # ── Settings ─────────────────────────────────────────────────────────────────
    SUBJECTS      = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    CLIP_DIM      = 512
    WIN_MS        = 50
    STEP_MS       = 25
    TMIN_BASE     = -0.1
    TMAX_BASE     = 0.496
    TRAIN_TMIN    = 0.05
    TRAIN_TMAX    = 0.25
    N_TRAIN       = 1200
    SPLIT_SEED    = 42
    N_BOOT        = 10000
    N_PERM        = 5000
    RNG_SEED      = 42

    STRICT_PRE  = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]

    # Windows
    win_s  = WIN_MS / 1000.0
    step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])

    # ── Concept table ────────────────────────────────────────────────────────────
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


    # ── CLIP embeddings (all 1854 concepts) ─────────────────────────────────────
    print("Computing CLIP embeddings...")
    clip_model_obj, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model_obj.eval()
    clip_embs = torch.zeros(n_concepts, CLIP_DIM)
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths:
            raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([clip_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            emb = clip_model_obj.encode_image(imgs).float()
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            clip_embs[i] = F.normalize(emb.unsqueeze(0), dim=-1)
    clip_embs_np = F.normalize(clip_embs, dim=-1).numpy()
    del clip_model_obj, clip_preprocess
    torch.cuda.empty_cache(); gc.collect()

    # ── Held-out concept split (matches heldout_uncertainty.py) ─────────────────
    rng_split = np.random.RandomState(SPLIT_SEED)
    all_ids = np.arange(n_concepts)
    rng_split.shuffle(all_ids)
    train_ids = set(all_ids[:N_TRAIN].tolist())
    eval_ids_set = set(all_ids[N_TRAIN:].tolist())
    eval_ids_sorted = sorted(eval_ids_set)
    N_EVAL = len(eval_ids_sorted)
    id_to_eval_local = {c: i for i, c in enumerate(eval_ids_sorted)}

    # CLIP RDM on the 654 held-out concepts
    eval_clip = clip_embs_np[eval_ids_sorted]
    tri_eval = np.triu(np.ones((N_EVAL, N_EVAL), dtype=bool), k=1)
    clip_rdm_eval = (eval_clip @ eval_clip.T)[tri_eval]

    print(f"Split: {N_TRAIN} train concepts / {N_EVAL} eval concepts (seed={SPLIT_SEED})")
    print(f"Train window: {TRAIN_TMIN*1000:.0f}-{TRAIN_TMAX*1000:.0f} ms")
    print(f"Sliding windows: {n_windows} ({WIN_MS} ms, step {STEP_MS} ms), "
          f"centers {int(centers_ms.min())}..{int(centers_ms.max())} ms\n")

    # ── Per-subject fixed-model ridge on eval concepts only ─────────────────────
    print(f"{'='*60}")
    print("Held-out fixed-model ridge sweep")
    print(f"{'='*60}")

    pre_mask  = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)

    rows = []
    successful = []

    for s_idx, subj in enumerate(SUBJECTS):
        t0 = time.time()
        print(f"  {subj} ({s_idx+1}/{len(SUBJECTS)})", end="", flush=True)
        vhdr  = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_eeg.vhdr"
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

            # Split trials by concept
            train_trial_mask = np.array([yi in train_ids for yi in y])
            eval_trial_mask  = np.array([yi in eval_ids_set for yi in y])

            # ── Train ridge ONCE on training concepts, training window ──────────
            train_time_mask = (times >= TRAIN_TMIN - 1e-6) & (times <= TRAIN_TMAX + 1e-6)
            feat_train = eeg_data[train_trial_mask][:, :, train_time_mask].mean(axis=2)
            y_train    = y[train_trial_mask]
            scaler = StandardScaler()
            feat_train_sc = scaler.fit_transform(feat_train)
            ridge = Ridge(alpha=1.0)
            ridge.fit(feat_train_sc, clip_embs_np[y_train])

            # ── Sweep over all windows, evaluate on held-out trials only ────────
            y_eval = y[eval_trial_mask]
            eeg_eval = eeg_data[eval_trial_mask]
            del eeg_data; gc.collect()

            subj_r = np.zeros(n_windows)
            for w_idx, (tmin_w, tmax_w, _center) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_eval[:, :, t_mask].mean(axis=2)
                feat_sc = scaler.transform(feat)
                preds = ridge.predict(feat_sc)
                concept_pred = np.zeros((N_EVAL, CLIP_DIM), dtype=np.float32)
                counts = np.zeros(N_EVAL, dtype=np.float32)
                for i, gid in enumerate(y_eval):
                    lid = id_to_eval_local[gid]
                    concept_pred[lid] += preds[i]
                    counts[lid] += 1
                m = counts > 0
                concept_pred[m] /= counts[m, None]
                norms = np.linalg.norm(concept_pred, axis=1, keepdims=True) + 1e-8
                cn = concept_pred / norms
                sim = cn @ cn.T
                r, _ = spearmanr(sim[tri_eval], clip_rdm_eval)
                subj_r[w_idx] = r

            del eeg_eval; gc.collect()

            pre_mean  = subj_r[pre_mask].mean()
            post_mean = subj_r[post_mask].mean()
            delta     = post_mean - pre_mean
            peak_idx  = int(np.argmax(subj_r))
            peak_r    = subj_r[peak_idx]
            peak_ms   = int(centers_ms[peak_idx])

            row = {
                "subject":   subj,
                "pre_mean":  pre_mean,
                "post_mean": post_mean,
                "delta":     delta,
                "peak_r":    peak_r,
                "peak_ms":   peak_ms,
                "n_train_trials": int(train_trial_mask.sum()),
                "n_eval_trials":  int(eval_trial_mask.sum()),
            }
            for w_idx in range(n_windows):
                row[f"r_{int(centers_ms[w_idx])}ms"] = subj_r[w_idx]
            rows.append(row)
            successful.append(subj)
            print(f"  done ({time.time()-t0:.0f}s) delta={delta:+.4f} peak={peak_r:+.4f}@{peak_ms}ms",
                  flush=True)
        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "heldout_temporal_fixed_model.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    summary_df = df[["subject", "pre_mean", "post_mean", "delta", "peak_r", "peak_ms",
                     "n_train_trials", "n_eval_trials"]]
    subj_csv = os.path.join(_RESULTS_DIR, "heldout_temporal_subject_summary.csv")
    summary_df.to_csv(subj_csv, index=False)
    print(f"Saved: {subj_csv}")

    # ── Group-level statistics ──────────────────────────────────────────────────
    deltas   = df["delta"].values
    peak_rs  = df["peak_r"].values
    peak_mss = df["peak_ms"].values
    pre_ms   = df["pre_mean"].values
    post_ms  = df["post_mean"].values

    # Per-window group mean / SEM
    window_cols = [f"r_{int(c)}ms" for c in centers_ms]
    window_mat = df[window_cols].values   # (n_subj, n_windows)
    group_mean = window_mat.mean(axis=0)
    group_sem  = window_mat.std(axis=0, ddof=1) / np.sqrt(n_subj)

    group_peak_idx = int(np.argmax(group_mean))
    group_peak_r   = float(group_mean[group_peak_idx])
    group_peak_ms  = int(centers_ms[group_peak_idx])

    # Sign-flip permutation on delta (subject-level, one-tailed post > pre)
    rng_perm = np.random.RandomState(RNG_SEED)
    obs_mean_delta = float(deltas.mean())
    perm_means = np.zeros(N_PERM)
    for b in range(N_PERM):
        signs = rng_perm.choice([-1.0, 1.0], size=n_subj)
        perm_means[b] = (signs * deltas).mean()
    perm_p = float(((perm_means >= obs_mean_delta).sum() + 1) / (N_PERM + 1))

    # Cohen's d_z for paired difference (delta vs 0)
    delta_sd = float(deltas.std(ddof=1))
    cohens_dz = float(obs_mean_delta / delta_sd) if delta_sd > 0 else float("nan")

    # Bootstrap CI for mean delta
    rng_boot = np.random.RandomState(RNG_SEED)
    boot_means = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = rng_boot.choice(n_subj, size=n_subj, replace=True)
        boot_means[b] = deltas[idx].mean()
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    # Sign test
    n_positive = int((deltas > 0).sum())
    prop_positive = n_positive / n_subj
    sign_p = binomtest(n_positive, n_subj, 0.5, alternative='greater').pvalue

    # Peak latency distribution
    n_125_225 = int(((peak_mss >= 125) & (peak_mss <= 225)).sum())
    n_150_200 = int(((peak_mss >= 150) & (peak_mss <= 200)).sum())
    n_post_peak = int((peak_mss > 0).sum())

    summary_txt = os.path.join(_RESULTS_DIR, "heldout_temporal_fixed_model_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("HELD-OUT TEMPORAL SWEEP (Fixed-Model Ridge, concept-disjoint eval)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Subjects: {n_subj}/{len(SUBJECTS)}\n")
        f.write(f"Held-out split: {N_TRAIN} train / {N_EVAL} eval (seed={SPLIT_SEED})\n")
        f.write(f"Training window: {TRAIN_TMIN*1000:.0f}-{TRAIN_TMAX*1000:.0f} ms\n")
        f.write(f"Pre interval (centers):  {STRICT_PRE} ms\n")
        f.write(f"Post interval (centers): {POST_TARGET} ms\n")
        f.write(f"Sliding windows: {n_windows} ({WIN_MS} ms, step {STEP_MS} ms)\n\n")

        f.write("--- GROUP-LEVEL TEMPORAL CURVE ---\n")
        f.write(f"  Group peak r:    {group_peak_r:+.4f} @ {group_peak_ms} ms\n")
        f.write(f"  Group pre  mean: {group_mean[pre_mask].mean():+.4f}\n")
        f.write(f"  Group post mean: {group_mean[post_mask].mean():+.4f}\n")
        f.write(f"  Group delta:     {group_mean[post_mask].mean() - group_mean[pre_mask].mean():+.4f}\n\n")

        f.write("--- SUBJECT-LEVEL POST-VS-PRE DELTA ---\n")
        f.write(f"  N positive:        {n_positive}/{n_subj} ({prop_positive*100:.0f}%)\n")
        f.write(f"  Sign test p:       {sign_p:.6f}\n")
        f.write(f"  Mean delta:        {obs_mean_delta:+.4f}\n")
        f.write(f"  SD delta:          {delta_sd:.4f}\n")
        f.write(f"  Cohen's d_z:       {cohens_dz:+.3f}\n")
        f.write(f"  Sign-flip p:       {perm_p:.6f} ({N_PERM} permutations, one-tailed)\n")
        f.write(f"  Bootstrap 95% CI:  [{ci_lo:+.4f}, {ci_hi:+.4f}]\n")
        f.write(f"  Median delta:      {np.median(deltas):+.4f}\n")
        f.write(f"  IQR:               [{np.percentile(deltas,25):+.4f}, {np.percentile(deltas,75):+.4f}]\n")
        f.write(f"  Min / Max:         [{deltas.min():+.4f}, {deltas.max():+.4f}]\n\n")

        f.write("--- PEAK LATENCY DISTRIBUTION ---\n")
        f.write(f"  Median peak:       {np.median(peak_mss):.0f} ms\n")
        f.write(f"  IQR:               [{np.percentile(peak_mss,25):.0f}, {np.percentile(peak_mss,75):.0f}] ms\n")
        f.write(f"  Min / Max:         [{peak_mss.min():.0f}, {peak_mss.max():.0f}] ms\n")
        f.write(f"  Post-stimulus peak: {n_post_peak}/{n_subj} ({n_post_peak/n_subj*100:.0f}%)\n")
        f.write(f"  In 125-225 ms:     {n_125_225}/{n_subj} ({n_125_225/n_subj*100:.0f}%)\n")
        f.write(f"  In 150-200 ms:     {n_150_200}/{n_subj} ({n_150_200/n_subj*100:.0f}%)\n\n")

        f.write("--- PEAK MAGNITUDE ---\n")
        f.write(f"  Mean peak r:       {peak_rs.mean():+.4f}\n")
        f.write(f"  Median peak r:     {np.median(peak_rs):+.4f}\n")
        f.write(f"  IQR:               [{np.percentile(peak_rs,25):+.4f}, {np.percentile(peak_rs,75):+.4f}]\n\n")

        f.write("--- PER-WINDOW GROUP MEAN (r +/- SEM) ---\n")
        for w_idx in range(n_windows):
            f.write(f"  {int(centers_ms[w_idx]):+4d} ms: {group_mean[w_idx]:+.4f}  "
                    f"+/- {group_sem[w_idx]:.4f}\n")

    print(f"Saved: {summary_txt}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    # per-subject curves (light)
    for i in range(n_subj):
        ax.plot(centers_ms, window_mat[i], color="steelblue", alpha=0.15, lw=0.8)
    # group mean + SEM band
    ax.plot(centers_ms, group_mean, color="crimson", lw=2.2,
            label=f"Group mean (N={n_subj})")
    ax.fill_between(centers_ms, group_mean - group_sem, group_mean + group_sem,
                    color="crimson", alpha=0.20, label="+/- SEM")
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(0, color="black", lw=0.6, alpha=0.5)
    ax.axvspan(STRICT_PRE[0] - STEP_MS/2, STRICT_PRE[-1] + STEP_MS/2,
               color="gray", alpha=0.10, label="Pre")
    ax.axvspan(POST_TARGET[0] - STEP_MS/2, POST_TARGET[-1] + STEP_MS/2,
               color="gold", alpha=0.15, label="Post")
    ax.plot(group_peak_ms, group_peak_r, marker="v", color="crimson", ms=12,
            label=f"Peak {group_peak_r:+.4f}@{group_peak_ms} ms")
    ax.set_xlabel("Time (ms, stimulus onset = 0)")
    ax.set_ylabel("Held-out Spearman r (654 eval concepts)")
    ax.set_title(f"Held-Out Temporal Sweep (Fixed-Model Ridge, CLIP, N={n_subj})\n"
                 f"delta={obs_mean_delta:+.4f}, d_z={cohens_dz:+.3f}, "
                 f"sign-flip p={perm_p:.4f}, {n_positive}/{n_subj} positive")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(_RESULTS_DIR, "heldout_temporal_fixed_model_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"  N subjects              : {n_subj}")
    print(f"  Group peak (held-out r) : {group_peak_r:+.4f} @ {group_peak_ms} ms")
    print(f"  Mean delta (post-pre)   : {obs_mean_delta:+.4f}  CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Cohen's d_z             : {cohens_dz:+.3f}")
    print(f"  Sign-flip p             : {perm_p:.4f}")
    print(f"  Subjects positive       : {n_positive}/{n_subj}  ({prop_positive*100:.0f}%)")
    print(f"  Peak in 125-225 ms      : {n_125_225}/{n_subj}  ({n_125_225/n_subj*100:.0f}%)")
    print(f"  Median peak latency     : {np.median(peak_mss):.0f} ms")
    print("=" * 60)
    print("DONE")


