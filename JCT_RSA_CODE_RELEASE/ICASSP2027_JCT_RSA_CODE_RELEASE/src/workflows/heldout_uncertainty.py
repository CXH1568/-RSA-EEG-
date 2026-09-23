"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Held-out uncertainty enhancement: bootstrap 95% CI and subject-wise
    distribution for all target embedding spaces.

    Reads existing v2 specificity results (per-subject held-out r) and computes:
      1. Bootstrap 95% CI (10,000 resamples) for each target
      2. Subject-wise percentiles (min, 25%, median, 75%, max)
      3. Between-target bootstrap CIs for CLIP-vs-each

    Outputs:
      results/heldout_uncertainty.csv
      results/heldout_uncertainty_summary.txt
    """

    import os, sys
    import numpy as np
    import pandas as pd

    _SCRIPT_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    _PROJECT_DIR = str(release_root())
    _RESULTS_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    N_BOOT = 10000
    RNG_SEED = 42

    # ── Load per-subject held-out r from v2 specificity checkpoint ──────────────
    # The v2 script saved per-subject temporal data; we need per-subject held-out r.
    # These are stored in the v2 heldout CSV as group summaries only.
    # Let's recompute from the checkpoint or re-extract.

    # Actually, the v2 script stored heldout_r as arrays but only saved summary.
    # We need to re-run the held-out extraction. Let's load the temporal checkpoint
    # which has per-subject per-window data and use the per-window held-out approach.

    # Simpler: load ridge_subject_level CSV (original CLIP results) + v2 temporal CSV
    # But the held-out values per subject weren't saved individually in any CSV.

    # Best approach: run a lightweight extraction using the same split and embeddings.
    # Since the full pipeline already ran, let's use a faster approach:
    # load the v2 temporal CSV which has per-target per-window group stats,
    # but that doesn't have per-subject held-out values.

    # Let me instead directly re-extract held-out r per subject from saved data.
    # The v2 script computed these but only saved mean/std. Let me compute fresh.

    print("Computing held-out uncertainty from scratch (fast, no EEG loading)...")
    print("Loading embeddings and running held-out ridge for all subjects...\n")

    # We need per-subject held-out r. The fastest way is to load the checkpoint CSV
    # from the v2 run and extract per-subject held-out values.

    # Check if v2 checkpoint exists with per-subject data
    checkpoint_path = os.path.join(_RESULTS_DIR, "clip_specificity_ridge_checkpoint.csv")
    v2_checkpoint_path = os.path.join(_RESULTS_DIR, "clip_specificity_v2_checkpoint.csv")

    # Since per-subject held-out values weren't saved to CSV, we'll compute
    # bootstrap CIs from the group summary statistics using a different approach:
    # load the existing per-subject temporal values and compute held-out
    # from the temporal data at the held-out window (175ms).

    # Actually, let me just load what we have and compute bootstrap from
    # the per-subject per-window temporal data at center=175ms.

    # The v2 temporal CSV has group summaries. Let me check if per-subject data
    # was saved anywhere...

    # Since per-subject held-out r wasn't saved, let me run a lightweight version
    # that only does held-out evaluation (no temporal sweep) using cached embeddings.

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import mne
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import clip
    import gc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Settings ─────────────────────────────────────────────────────────────────
    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    EMBED_DIM = 512
    N_TRAIN = 1200

    # Fixed evaluation window
    EVAL_TMIN = 0.05; EVAL_TMAX = 0.25

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

    def get_image_paths(concept, max_imgs=5):
        return image_paths(IMAGES_ROOT, concept, max_imgs)

    # ── Embeddings ───────────────────────────────────────────────────────────────
    print("Computing CLIP embeddings...")
    clip_model_obj, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model_obj.eval()
    clip_embs = torch.zeros(n_concepts, EMBED_DIM)
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths: raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([clip_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            emb = clip_model_obj.encode_image(imgs).float()
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            clip_embs[i] = F.normalize(emb.unsqueeze(0), dim=-1)
    clip_embs_np = F.normalize(clip_embs, dim=-1).numpy()
    del clip_model_obj, clip_preprocess; torch.cuda.empty_cache(); gc.collect()

    print("Computing ResNet-50 embeddings...")
    from torchvision.models import resnet50, ResNet50_Weights
    resnet_weights = ResNet50_Weights.IMAGENET1K_V2
    resnet_model = resnet50(weights=resnet_weights).to(device)
    resnet_model.eval()
    resnet_preprocess = resnet_weights.transforms()
    resnet_raw = np.zeros((n_concepts, 2048), dtype=np.float32)
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths: raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([resnet_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            x = resnet_model.conv1(imgs); x = resnet_model.bn1(x); x = resnet_model.relu(x)
            x = resnet_model.maxpool(x); x = resnet_model.layer1(x); x = resnet_model.layer2(x)
            x = resnet_model.layer3(x); x = resnet_model.layer4(x); x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)
            resnet_raw[i] = x.float().mean(dim=0).cpu().numpy()
    del resnet_model; torch.cuda.empty_cache(); gc.collect()
    pca_resnet = PCA(n_components=EMBED_DIM, random_state=RNG_SEED)
    resnet_embs_np = pca_resnet.fit_transform(resnet_raw).astype(np.float32)
    resnet_embs_np /= (np.linalg.norm(resnet_embs_np, axis=1, keepdims=True) + 1e-8)

    print("Computing Pixel-PCA + Random...")
    rng = np.random.RandomState(RNG_SEED)
    rand_embs_np = rng.randn(n_concepts, EMBED_DIM).astype(np.float32)
    rand_embs_np /= (np.linalg.norm(rand_embs_np, axis=1, keepdims=True) + 1e-8)

    pixel_features = []
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name, max_imgs=3)
        if not paths:
            raise FileNotFoundError(DATASET_ERROR)
        pix_list = []
        for p in paths:
            img = Image.open(p).convert("RGB").resize((64, 64))
            pix_list.append(np.array(img, dtype=np.float32).flatten() / 255.0)
        pixel_features.append(np.mean(pix_list, axis=0))
    pixel_mat = np.array(pixel_features)
    pca_pix = PCA(n_components=EMBED_DIM, random_state=RNG_SEED)
    pixel_pca_np = pca_pix.fit_transform(pixel_mat).astype(np.float32)
    pixel_pca_np /= (np.linalg.norm(pixel_pca_np, axis=1, keepdims=True) + 1e-8)

    TARGET_EMBS = {"CLIP": clip_embs_np, "ResNet50": resnet_embs_np,
                   "PixelPCA": pixel_pca_np, "Random": rand_embs_np}

    # ── Held-out split ───────────────────────────────────────────────────────────
    rng_split = np.random.RandomState(42)
    all_ids = np.arange(n_concepts)
    rng_split.shuffle(all_ids)
    train_concepts = set(all_ids[:N_TRAIN].tolist())
    eval_concepts = set(all_ids[N_TRAIN:].tolist())
    N_EVAL = n_concepts - N_TRAIN
    eval_ids_sorted = sorted(eval_concepts)
    id_to_eval_local = {c: i for i, c in enumerate(eval_ids_sorted)}
    tri_eval = np.triu(np.ones((N_EVAL, N_EVAL), dtype=bool), k=1)

    EVAL_RDMS = {}
    for name, embs in TARGET_EMBS.items():
        eval_embs = embs[eval_ids_sorted]
        sim = eval_embs @ eval_embs.T
        EVAL_RDMS[name] = sim[tri_eval]

    # ── Per-subject held-out evaluation ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Per-subject held-out evaluation (window: {EVAL_TMIN*1000:.0f}–{EVAL_TMAX*1000:.0f} ms)")
    print(f"{'='*60}")

    heldout_r = {name: [] for name in TARGET_EMBS}
    successful = []

    for s_idx, subj in enumerate(SUBJECTS):
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

            t_mask = (times >= EVAL_TMIN - 1e-6) & (times <= EVAL_TMAX + 1e-6)
            feat = eeg_data[:, :, t_mask].mean(axis=2)

            train_mask = np.array([y[i] in train_concepts for i in range(len(y))])
            eval_mask = np.array([y[i] in eval_concepts for i in range(len(y))])
            y_train = y[train_mask]; y_eval = y[eval_mask]
            feat_train = feat[train_mask]; feat_eval = feat[eval_mask]

            for name in TARGET_EMBS:
                target_embs = TARGET_EMBS[name]
                scaler = StandardScaler()
                ft = scaler.fit_transform(feat_train)
                fe = scaler.transform(feat_eval)
                ridge = Ridge(alpha=1.0)
                ridge.fit(ft, target_embs[y_train])
                preds = ridge.predict(fe)

                concept_pred = np.zeros((N_EVAL, EMBED_DIM), dtype=np.float32)
                counts = np.zeros(N_EVAL, dtype=np.float32)
                for i, gid in enumerate(y_eval):
                    lid = id_to_eval_local[gid]
                    concept_pred[lid] += preds[i]
                    counts[lid] += 1
                mask_e = counts > 0
                concept_pred[mask_e] /= counts[mask_e, None]
                norms = np.linalg.norm(concept_pred, axis=1, keepdims=True) + 1e-8
                cn = concept_pred / norms
                sim = cn @ cn.T
                rdm = sim[tri_eval]
                r, _ = spearmanr(rdm, EVAL_RDMS[name])
                heldout_r[name].append(r)

            del eeg_data; gc.collect()
            successful.append(subj)
            print(f"  CLIP={heldout_r['CLIP'][-1]:.4f} ResNet={heldout_r['ResNet50'][-1]:.4f}", flush=True)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    for name in TARGET_EMBS:
        heldout_r[name] = np.array(heldout_r[name])

    # ── Bootstrap 95% CI ────────────────────────────────────────────────────────
    print(f"\nComputing bootstrap 95% CI ({N_BOOT} resamples)...")
    rng_boot = np.random.RandomState(RNG_SEED)

    rows = []
    for name in TARGET_EMBS:
        vals = heldout_r[name]
        boot_means = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_boot.choice(len(vals), size=len(vals), replace=True)
            boot_means[b] = vals[idx].mean()
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        rows.append({
            "target": name,
            "mean_r": float(vals.mean()),
            "std_r": float(vals.std(ddof=1)),
            "sem_r": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "boot_ci_lo": float(ci_lo),
            "boot_ci_hi": float(ci_hi),
            "min": float(vals.min()),
            "p25": float(np.percentile(vals, 25)),
            "median": float(np.median(vals)),
            "p75": float(np.percentile(vals, 75)),
            "max": float(vals.max()),
            "n_subjects": len(vals),
        })
        print(f"  {name:12s}: mean={vals.mean():.4f}, 95% CI=[{ci_lo:.4f}, {ci_hi:.4f}], "
              f"median={np.median(vals):.4f}")

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "heldout_uncertainty.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Between-target bootstrap ────────────────────────────────────────────────
    print("\nBetween-target bootstrap CIs:")
    summary_lines = []
    for bname in ["ResNet50", "PixelPCA", "Random"]:
        diff = heldout_r["CLIP"] - heldout_r[bname]
        boot_diffs = np.zeros(N_BOOT)
        rng_b2 = np.random.RandomState(RNG_SEED)
        for b in range(N_BOOT):
            idx = rng_b2.choice(len(diff), size=len(diff), replace=True)
            boot_diffs[b] = diff[idx].mean()
        ci_lo = np.percentile(boot_diffs, 2.5)
        ci_hi = np.percentile(boot_diffs, 97.5)
        line = (f"  CLIP - {bname:12s}: diff={diff.mean():.4f}, "
                f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        print(line)
        summary_lines.append(line)

    # ── Per-subject CSV ──────────────────────────────────────────────────────────
    rows_subj = []
    for i, subj in enumerate(successful):
        row = {"subject": subj}
        for name in TARGET_EMBS:
            row[f"{name}_heldout_r"] = float(heldout_r[name][i])
        rows_subj.append(row)
    df_subj = pd.DataFrame(rows_subj)
    subj_csv = os.path.join(_RESULTS_DIR, "heldout_uncertainty_per_subject.csv")
    df_subj.to_csv(subj_csv, index=False)
    print(f"Saved: {subj_csv}")

    # ── Summary text ─────────────────────────────────────────────────────────────
    summary_path = os.path.join(_RESULTS_DIR, "heldout_uncertainty_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("HELD-OUT UNCERTAINTY ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Evaluation window: {EVAL_TMIN*1000:.0f}–{EVAL_TMAX*1000:.0f} ms\n")
        f.write(f"Bootstrap: {N_BOOT} resamples, seed={RNG_SEED}\n\n")

        f.write(f"{'Target':12s} {'Mean':>8s} {'95% CI':>20s} {'Median':>8s} "
                f"{'IQR':>16s} {'Range':>20s}\n")
        f.write("─" * 90 + "\n")
        for _, row in df_out.iterrows():
            f.write(f"{row['target']:12s} {row['mean_r']:8.4f} "
                    f"[{row['boot_ci_lo']:.4f}, {row['boot_ci_hi']:.4f}] "
                    f"{row['median']:8.4f} "
                    f"[{row['p25']:.4f}, {row['p75']:.4f}] "
                    f"[{row['min']:.4f}, {row['max']:.4f}]\n")

        f.write(f"\nBetween-target (CLIP - baseline):\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"Saved: {summary_path}")
    print("\n=== DONE ===")


