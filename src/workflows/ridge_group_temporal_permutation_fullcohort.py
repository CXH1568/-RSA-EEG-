"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Group-level ridge temporal permutation test — FULL COHORT (50 subjects).

    Identical pipeline to the 10-subject version
    (ridge_group_temporal_permutation.py), extended to all 50 available subjects.

    For each subject and each 50 ms sliding window (25 ms step):
      1. Crop EEG to window, compute per-trial mean amplitude (63-dim).
      2. Fit Ridge(alpha=1.0) on all trials: features -> CLIP embeddings.
      3. Predict, average per concept, L2-normalise, compute Spearman r vs CLIP RDM.

    Then at each window:
      - Run 5,000 sign-flip permutations on the group-mean r across all subjects.
      - Apply BH-FDR correction across all tested windows (alpha=0.05).

    Outputs:
      scripts/ridge_group_temporal_permutation_fullcohort.csv
      scripts/ridge_group_temporal_permutation_fullcohort_summary.txt
      scripts/ridge_group_temporal_permutation_fullcohort.png
      scripts/ridge_group_temporal_permutation_fullcohort_log.txt
      scripts/ridge_subject_level_temporal_values_fullcohort.csv
    """

    import os, sys, csv
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import numpy as np
    import pandas as pd
    import mne
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import clip
    import gc

    _SCRIPT_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    os.makedirs(_SCRIPT_DIR, exist_ok=True)

    # ── Logging ──────────────────────────────────────────────────────────────────
    _log = open(os.path.join(_SCRIPT_DIR, "ridge_group_temporal_permutation_fullcohort_log.txt"),
                "w", encoding="utf-8")
    import builtins as _bi
    _orig = _bi.print
    def print(*a, **kw):
        kw.setdefault("file", _log); _orig(*a, **kw); _log.flush()
        kw2 = {k: v for k, v in kw.items() if k != "file"}
        try: _orig(*a, **kw2)
        except Exception: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Settings ─────────────────────────────────────────────────────────────────
    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]  # sub-01 to sub-50, all 50
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    CLIP_DIM   = 512
    WIN_MS     = 50
    STEP_MS    = 25
    TMIN_BASE  = -0.1
    TMAX_BASE  = 0.496
    N_PERM     = 5000
    FDR_ALPHA  = 0.05

    print(f"Subjects: {SUBJECTS}")
    print(f"N_PERM={N_PERM}, FDR_ALPHA={FDR_ALPHA}")
    print(f"Window: {WIN_MS}ms, Step: {STEP_MS}ms")

    # ── Sliding window list ──────────────────────────────────────────────────────
    win_s  = WIN_MS / 1000.0
    step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    print(f"Windows: {n_windows}")

    # ── CLIP embeddings (computed once, shared across subjects) ──────────────────
    print("\nLoading concept names from sub-01 events...")
    df_ref = pd.read_csv(f"{BASE}/sub-01/eeg/sub-01_task-rsvp_events.tsv", sep='\t')
    df_ref = df_ref[df_ref['istarget'] == 0].reset_index(drop=True)
    unique_labels_ref = np.unique(df_ref['objectnumber'].values)
    label_map_ref = {v: i for i, v in enumerate(unique_labels_ref)}
    n_concepts = len(unique_labels_ref)

    concept_by_label = {}
    for _, row in df_ref.iterrows():
        lbl = label_map_ref.get(row['objectnumber'])
        if lbl is not None and lbl not in concept_by_label:
            concept_by_label[lbl] = row['object']
    concept_names = [concept_by_label[i] for i in range(n_concepts)]
    print(f"Concepts: {n_concepts}")

    print("Computing CLIP embeddings...")
    clip_model_obj, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model_obj.eval()

    def get_image_paths(concept, max_imgs=5):
        return image_paths(IMAGES_ROOT, concept, max_imgs)

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

    clip_embs_norm = F.normalize(clip_embs, dim=-1)
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)
    with torch.no_grad():
        clip_sim = clip_embs_norm @ clip_embs_norm.T
        clip_rdm_vec = clip_sim.numpy()[tri]

    clip_embs_np = clip_embs_norm.numpy()
    print(f"CLIP embeddings: {clip_embs.shape}, RDM vec: {clip_rdm_vec.shape}")

    # Free CLIP model
    del clip_model_obj, clip_preprocess
    torch.cuda.empty_cache(); gc.collect()

    # ── Per-subject, per-window ridge RDM computation ────────────────────────────
    # Track which subjects succeed (some may fail due to data issues)
    successful_subjects = []
    subject_r_list = []

    for s_idx, subj in enumerate(SUBJECTS):
        print(f"\n=== Subject {subj} ({s_idx+1}/{len(SUBJECTS)}) ===")

        vhdr   = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_eeg.vhdr"
        evfile = f"{BASE}/{subj}/eeg/{subj}_task-rsvp_events.tsv"

        try:
            # Load EEG
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            df_ev = pd.read_csv(evfile, sep='\t')
            df_ev = df_ev[df_ev['istarget'] == 0].reset_index(drop=True)

            events_np       = np.zeros((len(df_ev), 3), dtype=int)
            events_np[:, 0] = df_ev['onset'].values
            events_np[:, 2] = df_ev['objectnumber'].values

            epochs = mne.Epochs(raw, events_np, event_id=None,
                                tmin=-0.1, tmax=0.5,
                                baseline=(-0.1, 0.0),
                                preload=True, verbose=False)
            epochs.resample(250, verbose=False)

            y_raw         = df_ev['objectnumber'].values[:len(epochs)]
            valid_mask    = np.array([v in label_map_ref for v in y_raw])
            y             = np.array([label_map_ref[v] for v in y_raw[valid_mask]])

            eeg_data_all = epochs.get_data().astype(np.float32)
            eeg_data = eeg_data_all[valid_mask]
            times    = epochs.times
            del raw, epochs, eeg_data_all; gc.collect()

            print(f"  EEG shape: {eeg_data.shape}, trials: {len(y)}")

            subj_r = np.zeros(n_windows, dtype=np.float64)
            for w_idx, (tmin_w, tmax_w, center_ms) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                X_w = eeg_data[:, :, t_mask]

                feat = X_w.mean(axis=2)
                y_w  = y[:len(feat)]

                targets = clip_embs_np[y_w]
                scaler  = StandardScaler()
                feat_sc = scaler.fit_transform(feat)

                ridge = Ridge(alpha=1.0)
                ridge.fit(feat_sc, targets)

                preds = ridge.predict(feat_sc)

                concept_emb = np.zeros((n_concepts, CLIP_DIM), dtype=np.float32)
                counts      = np.zeros(n_concepts, dtype=np.float32)
                for i, lbl in enumerate(y_w):
                    concept_emb[lbl] += preds[i]
                    counts[lbl]      += 1
                mask = counts > 0
                concept_emb[mask] /= counts[mask, None]

                norms = np.linalg.norm(concept_emb, axis=1, keepdims=True) + 1e-8
                concept_norm = concept_emb / norms
                sim = concept_norm @ concept_norm.T
                r, _ = spearmanr(sim[tri], clip_rdm_vec)

                subj_r[w_idx] = r

            del eeg_data; gc.collect()
            print(f"  Peak r={subj_r.max():.4f} at {windows[np.argmax(subj_r)][2]}ms")

            successful_subjects.append(subj)
            subject_r_list.append(subj_r)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    # Assemble results
    SUBJECTS_FINAL = successful_subjects
    n_subj = len(SUBJECTS_FINAL)
    subject_r = np.array(subject_r_list)  # (n_subj, n_windows)
    print(f"\n=== Successfully processed {n_subj}/{len(SUBJECTS)} subjects ===")
    if n_subj < len(SUBJECTS):
        failed = set(SUBJECTS) - set(SUBJECTS_FINAL)
        print(f"  Failed subjects: {sorted(failed)}")

    # ── Group-level sign-flip permutation test ───────────────────────────────────
    print(f"\n=== Group-level sign-flip permutation test ({N_PERM} permutations, N={n_subj}) ===")

    rng = np.random.RandomState(0)

    group_means  = subject_r.mean(axis=0)
    group_stds   = subject_r.std(axis=0)

    null_means = np.zeros((N_PERM, n_windows), dtype=np.float64)
    for p in range(N_PERM):
        signs = rng.choice([-1, 1], size=n_subj)
        null_means[p] = (subject_r * signs[:, None]).mean(axis=0)

    p_unc = np.zeros(n_windows)
    for w in range(n_windows):
        p_unc[w] = (np.sum(null_means[:, w] >= group_means[w]) + 1) / (N_PERM + 1)

    def bh_fdr(pvals, alpha=0.05):
        pvals = np.asarray(pvals, dtype=float)
        m     = len(pvals)
        order = np.argsort(pvals)
        p_sorted   = pvals[order]
        adj_sorted = np.minimum.accumulate(
            (m / np.arange(1, m + 1)) * p_sorted[::-1])[::-1]
        adj_sorted = np.minimum(adj_sorted, 1.0)
        p_adj = np.empty(m)
        p_adj[order] = adj_sorted
        return p_adj, p_adj <= alpha

    p_fdr, sig = bh_fdr(p_unc, alpha=FDR_ALPHA)

    n_sig = int(sig.sum())
    print(f"\nFDR correction (BH, alpha={FDR_ALPHA}): {n_sig}/{n_windows} windows significant")
    for w in range(n_windows):
        if sig[w]:
            print(f"  * center={windows[w][2]:+4d}ms  group_r={group_means[w]:.4f}  "
                  f"p_unc={p_unc[w]:.5f}  p_fdr={p_fdr[w]:.5f}")

    # ── Save subject-level values ────────────────────────────────────────────────
    subj_csv = os.path.join(_SCRIPT_DIR, "ridge_subject_level_temporal_values_fullcohort.csv")
    with open(subj_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["center_ms"] + [f"{s}_ridge_r" for s in SUBJECTS_FINAL] + ["group_mean", "group_std"]
        writer.writerow(header)
        for w in range(n_windows):
            row = [windows[w][2]]
            row += [f"{subject_r[s, w]:.6f}" for s in range(n_subj)]
            row += [f"{group_means[w]:.6f}", f"{group_stds[w]:.6f}"]
            writer.writerow(row)
    print(f"Saved: {subj_csv}")

    # ── Save main results CSV ────────────────────────────────────────────────────
    main_csv = os.path.join(_SCRIPT_DIR, "ridge_group_temporal_permutation_fullcohort.csv")
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["center_ms", "group_mean_r", "group_std_r",
                         "ci95_lo", "ci95_hi",
                         "p_uncorrected", "p_fdr", "significant"])
        for w in range(n_windows):
            sem = group_stds[w] / np.sqrt(n_subj)
            ci_lo = group_means[w] - 1.96 * sem
            ci_hi = group_means[w] + 1.96 * sem
            writer.writerow([
                windows[w][2],
                f"{group_means[w]:.6f}", f"{group_stds[w]:.6f}",
                f"{ci_lo:.6f}", f"{ci_hi:.6f}",
                f"{p_unc[w]:.6f}", f"{p_fdr[w]:.6f}",
                int(sig[w])
            ])
    print(f"Saved: {main_csv}")

    # ── Save summary text ────────────────────────────────────────────────────────
    summary_path = os.path.join(_SCRIPT_DIR, "ridge_group_temporal_permutation_fullcohort_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("=== Ridge Group-Level Temporal Permutation Test — FULL COHORT ===\n\n")
        sf.write(f"Subjects: {', '.join(SUBJECTS_FINAL)}\n")
        sf.write(f"N subjects: {n_subj}\n")
        if n_subj < len(SUBJECTS):
            failed = sorted(set(SUBJECTS) - set(SUBJECTS_FINAL))
            sf.write(f"Excluded subjects (processing failure): {', '.join(failed)}\n")
        sf.write(f"Windows: {n_windows} (50ms window, 25ms step, -100 to +500ms)\n")
        sf.write(f"Ridge: alpha=1.0, not tuned, sklearn Ridge\n")
        sf.write(f"Features: mean EEG amplitude per channel per window (63-dim)\n")
        sf.write(f"Target: CLIP ViT-B/32 embeddings (512-dim)\n")
        sf.write(f"Metric: Spearman RDM correlation (all 1854 concepts, in-sample)\n")
        sf.write(f"Permutation: sign-flip, {N_PERM} permutations, seed=0\n")
        sf.write(f"Correction: BH-FDR, alpha={FDR_ALPHA}\n\n")
        sf.write(f"FDR-significant windows: {n_sig}/{n_windows}\n")
        if n_sig > 0:
            sig_centers = [windows[w][2] for w in range(n_windows) if sig[w]]
            sf.write(f"Significant range: {min(sig_centers)}ms to {max(sig_centers)}ms\n")
            gaps = []
            for w in range(n_windows):
                c = windows[w][2]
                if c >= min(sig_centers) and c <= max(sig_centers) and not sig[w]:
                    gaps.append(c)
            if gaps:
                sf.write(f"Gaps in cluster: {gaps}\n")
            else:
                sf.write("Cluster is fully contiguous (no gaps)\n")

        sf.write(f"\nPeak window:\n")
        peak_w = np.argmax(group_means)
        sf.write(f"  center={windows[peak_w][2]}ms, group_r={group_means[peak_w]:.4f}, "
                 f"p_unc={p_unc[peak_w]:.5f}, p_fdr={p_fdr[peak_w]:.5f}\n")

        sf.write(f"\nAll windows:\n")
        sf.write(f"  {'center_ms':>10}  {'group_r':>8}  {'p_unc':>8}  {'p_fdr':>8}  sig\n")
        sf.write(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  ---\n")
        for w in range(n_windows):
            flag = "*" if sig[w] else " "
            sf.write(f"  {windows[w][2]:>+10}  {group_means[w]:>8.4f}  "
                     f"{p_unc[w]:>8.5f}  {p_fdr[w]:>8.5f}  {flag}\n")
    print(f"Saved: {summary_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────────
    centers = np.array([w[2] for w in windows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Null band
    null_mean_per_w = null_means.mean(axis=0)
    null_std_per_w  = null_means.std(axis=0)
    ax1.fill_between(centers,
                     null_mean_per_w - 2 * null_std_per_w,
                     null_mean_per_w + 2 * null_std_per_w,
                     alpha=0.15, color="gray", label="Null +/-2 SD (5000 sign-flip)")
    ax1.plot(centers, null_mean_per_w, color="gray", linewidth=0.7, linestyle=":")

    # Group mean +/- SEM
    sem = group_stds / np.sqrt(n_subj)
    ax1.fill_between(centers, group_means - sem, group_means + sem,
                     alpha=0.2, color="steelblue")
    ax1.plot(centers, group_means,
             color="steelblue", linewidth=2.0, marker="o", markersize=4,
             label=f"Ridge group mean (N={n_subj})")

    # Significant windows
    sig_mask = np.array(sig)
    if sig_mask.any():
        ax1.scatter(centers[sig_mask], group_means[sig_mask],
                    color="gold", edgecolors="steelblue", s=70, zorder=5,
                    label=f"FDR-significant (alpha={FDR_ALPHA}, n={n_sig})")

    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax1.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Stimulus onset")
    ax1.set_ylabel("RDM correlation (Spearman r)", fontsize=11)
    ax1.set_title(
        f"Ridge Group-Level Temporal Significance — Full Cohort ({n_subj} subjects, {N_PERM} sign-flip permutations)\n"
        f"BH FDR alpha={FDR_ALPHA}: {n_sig}/{n_windows} windows significant",
        fontsize=10)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # p-value bars
    log_p  = -np.log10(np.clip(p_fdr, 1e-10, 1.0))
    thresh = -np.log10(FDR_ALPHA)
    colors = ["gold" if s else "lightsteelblue" for s in sig]
    ax2.bar(centers, log_p, width=STEP_MS * 0.8, color=colors, edgecolor="none")
    ax2.axhline(thresh, color="tomato", linestyle="--", linewidth=1.0,
                label=f"-log10({FDR_ALPHA}) = {thresh:.2f}")
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Time window center (ms)", fontsize=11)
    ax2.set_ylabel("-log10(p_fdr)", fontsize=10)
    ax2.legend(handles=[
        Patch(facecolor="gold", label="Significant (FDR)"),
        Patch(facecolor="lightsteelblue", label="Not significant"),
    ], fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    png_path = os.path.join(_SCRIPT_DIR, "ridge_group_temporal_permutation_fullcohort.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")

    # ── Final log summary ────────────────────────────────────────────────────────
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  Subjects       : {n_subj} ({', '.join(SUBJECTS_FINAL)})")
    print(f"  Windows        : {n_windows}")
    print(f"  Ridge          : alpha=1.0, sklearn Ridge, mean-amplitude features")
    print(f"  Permutations   : {N_PERM} sign-flip")
    print(f"  FDR alpha      : {FDR_ALPHA}")
    print(f"  Significant    : {n_sig}/{n_windows}")
    if n_sig > 0:
        sig_centers = [windows[w][2] for w in range(n_windows) if sig[w]]
        print(f"  Sig. range     : {min(sig_centers)}ms to {max(sig_centers)}ms")
        sig_idxs = np.where(sig)[0]
        peak_sig_w = sig_idxs[np.argmax(group_means[sig_idxs])]
        print(f"  Peak sig window: {windows[peak_sig_w][2]}ms, r={group_means[peak_sig_w]:.4f}, "
              f"p_fdr={p_fdr[peak_sig_w]:.5f}")
    print()
    print(f"  {'center_ms':>10}  {'group_r':>8}  {'p_unc':>8}  {'p_fdr':>8}  sig")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  ---")
    for w in range(n_windows):
        flag = "*" if sig[w] else " "
        print(f"  {windows[w][2]:>+10}  {group_means[w]:>8.4f}  "
              f"{p_unc[w]:>8.5f}  {p_fdr[w]:>8.5f}  {flag}")

    _log.close()


