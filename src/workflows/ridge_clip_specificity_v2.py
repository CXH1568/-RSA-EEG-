"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    CLIP Specificity Control v2 — Ridge Pipeline with Stronger Baselines

    Extends v1 by adding ImageNet-supervised ResNet-50 penultimate layer (2048-dim,
    PCA-reduced to 512-dim) as a stronger non-CLIP visual backbone control.

    Targets:
      1. CLIP ViT-B/32 (512-dim) — paper's main target
      2. Random embeddings (512-dim, seed=42) — null control
      3. Pixel-PCA (512-dim) — low-level visual control
      4. ResNet-50 ImageNet-supervised penultimate layer (2048→512 via PCA)

    For each subject × window, fits Ridge regression from EEG mean channel
    amplitude (63-dim) to each target space, computes Spearman RDM correlation.

    Outputs:
      results/clip_specificity_v2_temporal.csv
      results/clip_specificity_v2_heldout.csv
      results/clip_specificity_v2_post_vs_pre.csv
      results/clip_specificity_v2_between.csv
      results/clip_specificity_v2_summary.txt
      results/clip_specificity_v2_plot.png
      results/clip_specificity_v2_log.txt
    """

    import os, sys, gc, time, csv
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Import PIL first to avoid torchvision DLL issue
    from PIL import Image

    import numpy as np
    import pandas as pd
    import mne
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import clip

    _SCRIPT_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    _PROJECT_DIR = str(release_root())
    _RESULTS_DIR = str(resolve_path(settings, "RESULTS_ROOT"))
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # ── Logging ─────────────────────────────────────────────────────────────────
    _log_path = os.path.join(_RESULTS_DIR, "clip_specificity_v2_log.txt")
    _log = open(_log_path, "w", encoding="utf-8")
    import builtins as _bi
    _orig_print = _bi.print
    def print(*a, **kw):
        kw.setdefault("file", _log); _orig_print(*a, **kw); _log.flush()
        kw2 = {k: v for k, v in kw.items() if k != "file"}
        try: _orig_print(*a, **kw2)
        except Exception: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Settings ─────────────────────────────────────────────────────────────────
    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    EMBED_DIM = 512
    WIN_MS = 50; STEP_MS = 25
    TMIN_BASE = -0.1; TMAX_BASE = 0.496
    N_PERM = 5000
    RNG_SEED = 42

    STRICT_PRE = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]
    N_TRAIN = 1200

    # ── Windows ──────────────────────────────────────────────────────────────────
    win_s = WIN_MS / 1000.0; step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])
    print(f"Windows: {n_windows}, Centers: {centers_ms.tolist()}")

    # ── Concept setup ────────────────────────────────────────────────────────────
    print("\nLoading concept names from sub-01...")
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

    # ══════════════════════════════════════════════════════════════════════════════
    #  COMPUTE TARGET EMBEDDINGS
    # ══════════════════════════════════════════════════════════════════════════════

    # 1. CLIP ViT-B/32
    print("\n--- Computing CLIP ViT-B/32 embeddings ---")
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
    del clip_model_obj, clip_preprocess
    torch.cuda.empty_cache(); gc.collect()
    print(f"CLIP embeddings: {clip_embs_np.shape}")

    # 2. Random (null control)
    print("--- Computing Random embeddings (seed=42) ---")
    rng = np.random.RandomState(RNG_SEED)
    rand_embs_np = rng.randn(n_concepts, EMBED_DIM).astype(np.float32)
    rand_embs_np /= (np.linalg.norm(rand_embs_np, axis=1, keepdims=True) + 1e-8)
    print(f"Random embeddings: {rand_embs_np.shape}")

    # 3. Pixel-PCA (low-level control)
    print("--- Computing Pixel-PCA embeddings (512-dim) ---")
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
    print(f"Pixel-PCA embeddings: {pixel_pca_np.shape}, var: {pca_pix.explained_variance_ratio_.sum():.3f}")

    # 4. ResNet-50 ImageNet-supervised penultimate layer
    print("--- Computing ResNet-50 penultimate embeddings ---")
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision import transforms as T

    resnet_weights = ResNet50_Weights.IMAGENET1K_V2
    resnet_model = resnet50(weights=resnet_weights).to(device)
    resnet_model.eval()

    # Remove classification head — hook penultimate (avgpool output = 2048-dim)
    resnet_preprocess = resnet_weights.transforms()

    resnet_raw = np.zeros((n_concepts, 2048), dtype=np.float32)
    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths: raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([resnet_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            # Forward through all layers except fc
            x = resnet_model.conv1(imgs)
            x = resnet_model.bn1(x)
            x = resnet_model.relu(x)
            x = resnet_model.maxpool(x)
            x = resnet_model.layer1(x)
            x = resnet_model.layer2(x)
            x = resnet_model.layer3(x)
            x = resnet_model.layer4(x)
            x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)  # (batch, 2048)
            # Average across exemplars
            emb = x.float().mean(dim=0)
            resnet_raw[i] = emb.cpu().numpy()

    del resnet_model
    torch.cuda.empty_cache(); gc.collect()

    # PCA to 512-dim for fair comparison
    pca_resnet = PCA(n_components=EMBED_DIM, random_state=RNG_SEED)
    resnet_embs_np = pca_resnet.fit_transform(resnet_raw).astype(np.float32)
    resnet_embs_np /= (np.linalg.norm(resnet_embs_np, axis=1, keepdims=True) + 1e-8)
    resnet_var = pca_resnet.explained_variance_ratio_.sum()
    print(f"ResNet-50 embeddings: {resnet_embs_np.shape}, PCA var: {resnet_var:.3f}")

    # ── Target dictionary ───────────────────────────────────────────────────────
    TARGET_EMBS = {
        "CLIP": clip_embs_np,
        "ResNet50": resnet_embs_np,
        "PixelPCA": pixel_pca_np,
        "Random": rand_embs_np,
    }
    TARGET_NAMES = list(TARGET_EMBS.keys())

    # ── RDM vectors ─────────────────────────────────────────────────────────────
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)

    def compute_rdm_vec(embs):
        sim = embs @ embs.T
        return sim[tri]

    TARGET_RDMS = {name: compute_rdm_vec(embs) for name, embs in TARGET_EMBS.items()}
    print(f"\nTarget RDM vectors: { {k: v.shape for k, v in TARGET_RDMS.items()} }")

    # ── Held-out split ───────────────────────────────────────────────────────────
    rng_split = np.random.RandomState(42)
    all_concept_ids = np.arange(n_concepts)
    rng_split.shuffle(all_concept_ids)
    train_concepts = set(all_concept_ids[:N_TRAIN].tolist())
    eval_concepts = set(all_concept_ids[N_TRAIN:].tolist())
    N_EVAL = n_concepts - N_TRAIN
    eval_ids_sorted = sorted(eval_concepts)
    id_to_eval_local = {c: i for i, c in enumerate(eval_ids_sorted)}

    tri_eval = np.triu(np.ones((N_EVAL, N_EVAL), dtype=bool), k=1)
    EVAL_RDMS = {}
    for name, embs in TARGET_EMBS.items():
        eval_embs = embs[eval_ids_sorted]
        sim = eval_embs @ eval_embs.T
        EVAL_RDMS[name] = sim[tri_eval]

    print(f"Held-out split: {N_TRAIN} train / {N_EVAL} eval concepts")

    # ══════════════════════════════════════════════════════════════════════════════
    #  PER-SUBJECT RIDGE SWEEP + HELD-OUT
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Per-subject ridge temporal sweep (all targets)")
    print(f"{'='*70}")

    all_r = {name: [] for name in TARGET_NAMES}
    heldout_r = {name: [] for name in TARGET_NAMES}
    successful = []

    HELDOUT_CENTER = 175
    heldout_widx = int(np.argmin(np.abs(centers_ms - HELDOUT_CENTER)))
    print(f"Held-out window: center={centers_ms[heldout_widx]}ms (index {heldout_widx})")

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

            n_trials = len(y)
            train_trial_mask = np.array([y[i] in train_concepts for i in range(n_trials)])
            eval_trial_mask = np.array([y[i] in eval_concepts for i in range(n_trials)])
            y_train = y[train_trial_mask]
            y_eval = y[eval_trial_mask]

            subj_r = {name: np.zeros(n_windows, dtype=np.float64) for name in TARGET_NAMES}

            for w_idx, (tmin_w, tmax_w, center) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                X_w = eeg_data[:, :, t_mask]
                feat = X_w.mean(axis=2)

                for name in TARGET_NAMES:
                    target_embs = TARGET_EMBS[name]
                    targets = target_embs[y]

                    scaler = StandardScaler()
                    feat_sc = scaler.fit_transform(feat)

                    ridge = Ridge(alpha=1.0)
                    ridge.fit(feat_sc, targets)
                    preds = ridge.predict(feat_sc)

                    concept_pred = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                    counts = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        concept_pred[lbl] += preds[i]
                        counts[lbl] += 1
                    mask_c = counts > 0
                    concept_pred[mask_c] /= counts[mask_c, None]

                    norms = np.linalg.norm(concept_pred, axis=1, keepdims=True) + 1e-8
                    concept_norm = concept_pred / norms
                    pred_sim = concept_norm @ concept_norm.T
                    pred_rdm = pred_sim[tri]

                    r, _ = spearmanr(pred_rdm, TARGET_RDMS[name])
                    subj_r[name][w_idx] = r

                # Held-out at target window
                if w_idx == heldout_widx:
                    feat_train = feat[train_trial_mask]
                    feat_eval = feat[eval_trial_mask]

                    for name in TARGET_NAMES:
                        target_embs = TARGET_EMBS[name]

                        scaler_ho = StandardScaler()
                        feat_train_sc = scaler_ho.fit_transform(feat_train)
                        feat_eval_sc = scaler_ho.transform(feat_eval)

                        ridge_ho = Ridge(alpha=1.0)
                        ridge_ho.fit(feat_train_sc, target_embs[y_train])
                        preds_eval = ridge_ho.predict(feat_eval_sc)

                        concept_eval_pred = np.zeros((N_EVAL, EMBED_DIM), dtype=np.float32)
                        counts_eval = np.zeros(N_EVAL, dtype=np.float32)
                        for i, gid in enumerate(y_eval):
                            lid = id_to_eval_local[gid]
                            concept_eval_pred[lid] += preds_eval[i]
                            counts_eval[lid] += 1
                        mask_e = counts_eval > 0
                        concept_eval_pred[mask_e] /= counts_eval[mask_e, None]

                        norms_e = np.linalg.norm(concept_eval_pred, axis=1, keepdims=True) + 1e-8
                        concept_eval_norm = concept_eval_pred / norms_e
                        eval_sim = concept_eval_norm @ concept_eval_norm.T
                        eval_rdm = eval_sim[tri_eval]

                        r_ho, _ = spearmanr(eval_rdm, EVAL_RDMS[name])
                        heldout_r[name].append(r_ho)

            del eeg_data; gc.collect()

            for name in TARGET_NAMES:
                all_r[name].append(subj_r[name])
            successful.append(subj)

            elapsed = time.time() - t0
            peaks = " ".join([f"{n}={subj_r[n].max():.4f}" for n in TARGET_NAMES])
            ho_clip = heldout_r['CLIP'][-1] if heldout_r['CLIP'] else 0
            print(f"  done ({elapsed:.0f}s) peak: {peaks} HO_CLIP={ho_clip:.4f}", flush=True)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    print(f"\n\nSuccessful subjects: {n_subj}")

    for name in TARGET_NAMES:
        all_r[name] = np.array(all_r[name])
        heldout_r[name] = np.array(heldout_r[name])

    # ══════════════════════════════════════════════════════════════════════════════
    #  TEMPORAL PROFILE TABLE
    # ══════════════════════════════════════════════════════════════════════════════
    rows_temporal = []
    for w_idx in range(n_windows):
        row = {"center_ms": int(centers_ms[w_idx])}
        for name in TARGET_NAMES:
            vals = all_r[name][:, w_idx]
            row[f"{name}_mean"] = float(vals.mean())
            row[f"{name}_sem"] = float(vals.std(ddof=1) / np.sqrt(n_subj))
            row[f"{name}_std"] = float(vals.std(ddof=1))
        rows_temporal.append(row)

    df_temporal = pd.DataFrame(rows_temporal)
    temporal_csv = os.path.join(_RESULTS_DIR, "clip_specificity_v2_temporal.csv")
    df_temporal.to_csv(temporal_csv, index=False)
    print(f"\nSaved: {temporal_csv}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  HELD-OUT TABLE
    # ══════════════════════════════════════════════════════════════════════════════
    rows_heldout = []
    for name in TARGET_NAMES:
        vals = heldout_r[name]
        rows_heldout.append({
            "target": name,
            "mean_r": float(vals.mean()),
            "std_r": float(vals.std(ddof=1)),
            "sem_r": float(vals.std(ddof=1) / np.sqrt(len(vals))),
            "n_subjects": len(vals),
        })

    df_heldout = pd.DataFrame(rows_heldout)
    heldout_csv = os.path.join(_RESULTS_DIR, "clip_specificity_v2_heldout.csv")
    df_heldout.to_csv(heldout_csv, index=False)
    print(f"Saved: {heldout_csv}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  POST-VS-PRE TEST
    # ══════════════════════════════════════════════════════════════════════════════
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
        return {
            "delta": float(obs), "cohens_d": float(d), "p_value": float(p_val),
            "pre_mean": float(pre_means.mean()), "post_mean": float(post_means.mean()),
            "pre_std": float(pre_means.std(ddof=1)), "post_std": float(post_means.std(ddof=1)),
            "n": len(deltas),
        }

    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)

    rows_pv = []
    for name in TARGET_NAMES:
        data = all_r[name]
        pre_means = data[:, pre_mask].mean(axis=1)
        post_means = data[:, post_mask].mean(axis=1)
        res = sign_flip_test(pre_means, post_means, N_PERM, RNG_SEED)
        res["target"] = name

        group_mean = data.mean(axis=0)
        peak_idx = int(np.argmax(group_mean))
        res["peak_r"] = float(group_mean[peak_idx])
        res["peak_ms"] = int(centers_ms[peak_idx])

        rows_pv.append(res)
        print(f"\n{name:12s}: pre={res['pre_mean']:.4f}  post={res['post_mean']:.4f}  "
              f"delta={res['delta']:.4f}  d={res['cohens_d']:.2f}  p={res['p_value']:.4f}  "
              f"peak_r={res['peak_r']:.4f}@{res['peak_ms']}ms")

    df_pv = pd.DataFrame(rows_pv)
    pv_csv = os.path.join(_RESULTS_DIR, "clip_specificity_v2_post_vs_pre.csv")
    df_pv.to_csv(pv_csv, index=False)
    print(f"\nSaved: {pv_csv}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  BETWEEN-TARGET COMPARISONS
    # ══════════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("Between-target comparisons (CLIP vs each baseline)")
    print(f"{'='*70}")

    def paired_sign_flip(vals_a, vals_b, n_perm=5000, seed=42):
        rng_t = np.random.RandomState(seed)
        deltas = vals_a - vals_b
        obs = deltas.mean()
        null = np.zeros(n_perm)
        for p in range(n_perm):
            signs = rng_t.choice([-1.0, 1.0], size=len(deltas))
            null[p] = (deltas * signs).mean()
        p_val = (np.sum(null >= obs) + 1) / (n_perm + 1)
        std_d = deltas.std(ddof=1)
        d = obs / std_d if std_d > 0 else 0.0
        return {"delta": float(obs), "cohens_d": float(d), "p_value": float(p_val)}

    clip_pre = all_r["CLIP"][:, pre_mask].mean(axis=1)
    clip_post = all_r["CLIP"][:, post_mask].mean(axis=1)
    clip_delta = clip_post - clip_pre

    rows_between = []
    for bname in ["ResNet50", "PixelPCA", "Random"]:
        base_pre = all_r[bname][:, pre_mask].mean(axis=1)
        base_post = all_r[bname][:, post_mask].mean(axis=1)
        base_delta = base_post - base_pre

        # Post-vs-pre delta comparison
        res = paired_sign_flip(clip_delta, base_delta, N_PERM, RNG_SEED)
        res["comparison"] = f"CLIP_vs_{bname}"
        res["test"] = "post_vs_pre_delta"
        rows_between.append(res)
        print(f"  CLIP delta vs {bname} delta: diff={res['delta']:.4f}, d={res['cohens_d']:.2f}, p={res['p_value']:.4f}")

        # Peak r comparison
        clip_peak_vals = all_r["CLIP"][:, post_mask].max(axis=1)
        base_peak_vals = all_r[bname][:, post_mask].max(axis=1)
        res2 = paired_sign_flip(clip_peak_vals, base_peak_vals, N_PERM, RNG_SEED)
        res2["comparison"] = f"CLIP_vs_{bname}"
        res2["test"] = "peak_r"
        rows_between.append(res2)
        print(f"  CLIP peak vs {bname} peak:   diff={res2['delta']:.4f}, d={res2['cohens_d']:.2f}, p={res2['p_value']:.4f}")

        # Held-out comparison
        res3 = paired_sign_flip(heldout_r["CLIP"], heldout_r[bname], N_PERM, RNG_SEED)
        res3["comparison"] = f"CLIP_vs_{bname}"
        res3["test"] = "heldout_r"
        rows_between.append(res3)
        print(f"  CLIP held-out vs {bname} held-out: diff={res3['delta']:.4f}, d={res3['cohens_d']:.2f}, p={res3['p_value']:.4f}")

    df_between = pd.DataFrame(rows_between)
    between_csv = os.path.join(_RESULTS_DIR, "clip_specificity_v2_between.csv")
    df_between.to_csv(between_csv, index=False)
    print(f"\nSaved: {between_csv}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════════
    summary_path = os.path.join(_RESULTS_DIR, "clip_specificity_v2_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CLIP SPECIFICITY v2 — RIDGE PIPELINE WITH STRONGER BASELINES\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Ridge: alpha=1.0, EEG mean amplitude (63-dim) → target embeddings (512-dim)\n")
        f.write(f"Windows: {n_windows} ({WIN_MS}ms wide, {STEP_MS}ms step)\n")
        f.write(f"ResNet-50 PCA explained variance: {resnet_var:.3f}\n")
        f.write(f"Strict pre-onset: {STRICT_PRE} ms\n")
        f.write(f"Post target: {POST_TARGET} ms\n")
        f.write(f"Permutations: {N_PERM}\n\n")

        f.write("─── TEMPORAL PROFILE ───\n")
        f.write(f"{'Target':12s} {'Peak r':>10s} {'Peak ms':>10s} {'Pre mean':>10s} "
                f"{'Post mean':>10s} {'Delta':>10s} {'d':>8s} {'p':>10s}\n")
        f.write("─" * 82 + "\n")
        for r in rows_pv:
            f.write(f"{r['target']:12s} {r['peak_r']:10.4f} {r['peak_ms']:10d} "
                    f"{r['pre_mean']:10.4f} {r['post_mean']:10.4f} {r['delta']:10.4f} "
                    f"{r['cohens_d']:8.2f} {r['p_value']:10.4f}\n")

        f.write(f"\n─── HELD-OUT CONCEPT EVALUATION (window center={centers_ms[heldout_widx]}ms) ───\n")
        f.write(f"{'Target':12s} {'Mean r':>10s} {'SEM':>10s} {'N':>5s}\n")
        f.write("─" * 40 + "\n")
        for _, row in df_heldout.iterrows():
            f.write(f"{row['target']:12s} {row['mean_r']:10.4f} {row['sem_r']:10.4f} "
                    f"{int(row['n_subjects']):5d}\n")

        f.write(f"\n─── BETWEEN-TARGET TESTS (CLIP vs baselines) ───\n")
        f.write(f"{'Comparison':25s} {'Test':20s} {'Delta':>10s} {'d':>8s} {'p':>10s}\n")
        f.write("─" * 75 + "\n")
        for _, row in df_between.iterrows():
            f.write(f"{row['comparison']:25s} {row['test']:20s} "
                    f"{row['delta']:10.4f} {row['cohens_d']:8.2f} {row['p_value']:10.4f}\n")

        f.write(f"\n─── VERDICT ───\n")
        clip_pv = [r for r in rows_pv if r['target'] == 'CLIP'][0]
        resnet_pv = [r for r in rows_pv if r['target'] == 'ResNet50'][0]
        pix_pv = [r for r in rows_pv if r['target'] == 'PixelPCA'][0]
        rand_pv = [r for r in rows_pv if r['target'] == 'Random'][0]

        for name, pv in [("CLIP", clip_pv), ("ResNet50", resnet_pv),
                         ("PixelPCA", pix_pv), ("Random", rand_pv)]:
            sig = "YES" if pv['p_value'] < 0.05 else "NO"
            f.write(f"{name:12s} post-vs-pre significant: {sig} "
                    f"(d={pv['cohens_d']:.2f}, p={pv['p_value']:.4f})\n")

        # Key question: CLIP vs ResNet50
        clip_vs_resnet_delta = [r for r in rows_between
                                if r['comparison'] == 'CLIP_vs_ResNet50' and r['test'] == 'post_vs_pre_delta'][0]
        clip_vs_resnet_ho = [r for r in rows_between
                             if r['comparison'] == 'CLIP_vs_ResNet50' and r['test'] == 'heldout_r'][0]

        f.write(f"\nCRITICAL: CLIP vs ResNet-50\n")
        f.write(f"  Post-vs-pre delta: CLIP-ResNet diff = {clip_vs_resnet_delta['delta']:.4f}, "
                f"d = {clip_vs_resnet_delta['cohens_d']:.2f}, p = {clip_vs_resnet_delta['p_value']:.4f}\n")
        f.write(f"  Held-out: CLIP-ResNet diff = {clip_vs_resnet_ho['delta']:.4f}, "
                f"d = {clip_vs_resnet_ho['cohens_d']:.2f}, p = {clip_vs_resnet_ho['p_value']:.4f}\n")

        if clip_vs_resnet_delta['p_value'] < 0.05 and clip_vs_resnet_ho['p_value'] < 0.05:
            f.write("\nCONCLUSION: CLIP specificity SUPPORTED even against stronger baseline.\n")
            f.write("CLIP significantly exceeds supervised ResNet-50 in both temporal\n")
            f.write("concentration and held-out generalization.\n")
        elif clip_vs_resnet_delta['p_value'] < 0.05 or clip_vs_resnet_ho['p_value'] < 0.05:
            f.write("\nCONCLUSION: CLIP specificity PARTIALLY SUPPORTED.\n")
            f.write("CLIP exceeds ResNet-50 on some but not all metrics.\n")
        else:
            f.write("\nCONCLUSION: CLIP specificity NOT SUPPORTED against stronger baseline.\n")
            f.write("ResNet-50 matches or exceeds CLIP, suggesting the effect may reflect\n")
            f.write("generic high-level visual features rather than CLIP-specific semantics.\n")

    print(f"\nSaved: {summary_path}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  PLOT
    # ══════════════════════════════════════════════════════════════════════════════
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

    COLORS = {"CLIP": "steelblue", "ResNet50": "forestgreen", "PixelPCA": "darkorange", "Random": "gray"}

    ax = axes[0]
    for name in TARGET_NAMES:
        data = all_r[name]
        mean_curve = data.mean(axis=0)
        sem_curve = data.std(axis=0, ddof=1) / np.sqrt(n_subj)
        ax.fill_between(centers_ms, mean_curve - sem_curve, mean_curve + sem_curve,
                        alpha=0.15, color=COLORS[name])
        ax.plot(centers_ms, mean_curve, color=COLORS[name], linewidth=2,
                marker="o", markersize=3, label=f"{name} (N={n_subj})")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Stimulus onset")
    ax.axvspan(STRICT_PRE[0], STRICT_PRE[-1], alpha=0.08, color="blue", label="Strict pre")
    ax.axvspan(POST_TARGET[0], POST_TARGET[-1], alpha=0.08, color="red", label="Post target")
    ax.set_ylabel("RDM correlation (Spearman r)", fontsize=11)
    ax.set_title(f"CLIP Specificity v2 — Ridge Temporal Sweep ({n_subj} subjects)\n"
                 f"CLIP vs ResNet-50 vs Pixel-PCA vs Random", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ho_means = [heldout_r[name].mean() for name in TARGET_NAMES]
    ho_sems = [heldout_r[name].std(ddof=1) / np.sqrt(len(heldout_r[name])) for name in TARGET_NAMES]
    bars = ax2.bar(TARGET_NAMES, ho_means, yerr=ho_sems, capsize=5,
                   color=[COLORS[n] for n in TARGET_NAMES], alpha=0.8, edgecolor="black")
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax2.set_ylabel("Held-out r", fontsize=11)
    ax2.set_title(f"Held-Out Concept Evaluation (window={centers_ms[heldout_widx]}ms)", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, m, s in zip(bars, ho_means, ho_sems):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.001,
                 f"{m:.4f}", ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    plot_path = os.path.join(_RESULTS_DIR, "clip_specificity_v2_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    print("\n=== DONE ===")
    _log.close()


