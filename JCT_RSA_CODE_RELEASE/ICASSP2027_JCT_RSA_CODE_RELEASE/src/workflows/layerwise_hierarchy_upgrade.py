"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Layer-wise hierarchy upgrade.

    Two analyses in one pass (EEG loaded once per subject):

    Analysis A: Layer-wise temporal comparison (FIXED-MODEL ridge).
      For each ResNet-50 layer target in {layer1 (early), layer2 (mid),
      layer4/avgpool (late)}: train ridge once on 50-250 ms mean amplitude,
      freeze, sweep across 22 windows, correlate predicted RDM with the
      corresponding target's RDM.  Report per-layer temporal summary:
      peak r, peak latency, pre mean, post mean, delta, d, p.

    Analysis B: Extended partial RSA (PER-WINDOW ridge, matching section 4.6).
      For target in {CLIP, ResNet50_late (layer4)}, at each window train ridge
      on layer's embeddings, compute predicted RDM, and compute partial
      Spearman against the target's ground-truth RDM controlling for:
        - PixelPCA
        - PixelPCA + layer1 (early)
        - PixelPCA + layer2 (mid)
        - PixelPCA + layer1 + layer2 (early+mid)
      Report post-vs-pre delta and r@+175ms for each analysis.

    Outputs (under results/):
      layerwise_temporal_comparison.csv        per-subject per-window curves (A)
      layerwise_temporal_summary.txt           group-level A summary
      layerwise_partial_rsa.csv                per-subject per-window curves (B)
      layerwise_partial_rsa_summary.txt        group-level B summary
      layerwise_hierarchy_plot.png             combined visualisation
    """

    import os, sys, gc, time
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from PIL import Image
    import numpy as np
    import pandas as pd
    import mne
    import torch
    import torch.nn.functional as F
    from scipy.stats import spearmanr, rankdata
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
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
    EMBED_DIM     = 512
    WIN_MS        = 50
    STEP_MS       = 25
    TMIN_BASE     = -0.1
    TMAX_BASE     = 0.496
    TRAIN_TMIN    = 0.05
    TRAIN_TMAX    = 0.25
    N_PERM        = 5000
    RNG_SEED      = 42
    STRICT_PRE    = [-75, -50, -25]
    POST_TARGET   = [125, 150, 175, 200, 225, 250, 275, 300]

    # Windows
    win_s = WIN_MS / 1000.0
    step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])
    pre_mask  = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)
    w175_idx  = int(np.argmin(np.abs(centers_ms - 175)))

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


    # ── Embeddings ──────────────────────────────────────────────────────────────
    print("Computing CLIP embeddings...")
    clip_model_obj, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model_obj.eval()
    clip_embs = torch.zeros(n_concepts, EMBED_DIM)
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

    print("Computing ResNet-50 layer1 / layer2 / layer4 embeddings...")
    from torchvision.models import resnet50, ResNet50_Weights
    resnet_weights = ResNet50_Weights.IMAGENET1K_V2
    resnet_model = resnet50(weights=resnet_weights).to(device)
    resnet_model.eval()
    resnet_preprocess = resnet_weights.transforms()

    # layer1 = 256 channels, layer2 = 512, layer4 = 2048 (ResNet50 after blocks)
    resnet_layer1_raw = np.zeros((n_concepts, 256),  dtype=np.float32)
    resnet_layer2_raw = np.zeros((n_concepts, 512),  dtype=np.float32)
    resnet_layer4_raw = np.zeros((n_concepts, 2048), dtype=np.float32)

    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths:
            raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([resnet_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            x = resnet_model.conv1(imgs); x = resnet_model.bn1(x); x = resnet_model.relu(x)
            x = resnet_model.maxpool(x)
            x = resnet_model.layer1(x)
            l1_feat = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            resnet_layer1_raw[i] = l1_feat.float().mean(dim=0).cpu().numpy()
            x = resnet_model.layer2(x)
            l2_feat = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            resnet_layer2_raw[i] = l2_feat.float().mean(dim=0).cpu().numpy()
            x = resnet_model.layer3(x)
            x = resnet_model.layer4(x)
            x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)
            resnet_layer4_raw[i] = x.float().mean(dim=0).cpu().numpy()

    del resnet_model
    torch.cuda.empty_cache(); gc.collect()

    # PCA to 512-dim, L2 normalise
    def pca_normalize(raw, seed=RNG_SEED):
        pca = PCA(n_components=min(EMBED_DIM, raw.shape[1]), random_state=seed)
        emb = pca.fit_transform(raw).astype(np.float32)
        # pad to 512 if layer1 is 256-dim
        if emb.shape[1] < EMBED_DIM:
            pad = np.zeros((emb.shape[0], EMBED_DIM - emb.shape[1]), dtype=np.float32)
            emb = np.concatenate([emb, pad], axis=1)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        return emb, float(pca.explained_variance_ratio_.sum())


    resnet_l1_np, var_l1 = pca_normalize(resnet_layer1_raw)
    resnet_l2_np, var_l2 = pca_normalize(resnet_layer2_raw)
    resnet_l4_np, var_l4 = pca_normalize(resnet_layer4_raw)
    print(f"ResNet layer1 PCA var: {var_l1:.3f}")
    print(f"ResNet layer2 PCA var: {var_l2:.3f}")
    print(f"ResNet layer4 PCA var: {var_l4:.3f}")

    print("Computing PixelPCA...")
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

    # ── RDMs ─────────────────────────────────────────────────────────────────────
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)

    def rdm_vec(embs):
        return (embs @ embs.T)[tri]


    TARGET_EMBS = {
        "CLIP":             clip_embs_np,
        "ResNet50_layer1":  resnet_l1_np,
        "ResNet50_layer2":  resnet_l2_np,
        "ResNet50_layer4":  resnet_l4_np,
    }
    TARGET_RDMS = {name: rdm_vec(e) for name, e in TARGET_EMBS.items()}
    CONTROL_RDMS = {
        "PixelPCA":          rdm_vec(pixel_pca_np),
        "ResNet50_layer1":   rdm_vec(resnet_l1_np),
        "ResNet50_layer2":   rdm_vec(resnet_l2_np),
    }

    # ── Partial correlation helpers (rank-based partial Spearman) ───────────────
    def partial_spearman_multi(x, y, Z_list):
        """Partial Spearman r(x, y | Z1, Z2, ...) via rank OLS residuals."""
        rx = rankdata(x).astype(np.float64); rx -= rx.mean()
        ry = rankdata(y).astype(np.float64); ry -= ry.mean()
        if not Z_list:
            denom = (np.sqrt(np.dot(rx, rx)) * np.sqrt(np.dot(ry, ry))) + 1e-12
            return float(np.dot(rx, ry) / denom)
        Z = np.column_stack([rankdata(z).astype(np.float64) for z in Z_list])
        Z -= Z.mean(axis=0)
        ZtZ = Z.T @ Z
        ZtZ_inv = np.linalg.inv(ZtZ + 1e-10 * np.eye(len(ZtZ)))
        rx_res = rx - Z @ (ZtZ_inv @ (Z.T @ rx))
        ry_res = ry - Z @ (ZtZ_inv @ (Z.T @ ry))
        denom = (np.sqrt(np.dot(rx_res, rx_res)) * np.sqrt(np.dot(ry_res, ry_res))) + 1e-12
        return float(np.dot(rx_res, ry_res) / denom)


    # ── Analysis definitions ─────────────────────────────────────────────────────
    #   Analysis A: fixed-model ridge, per-layer target
    A_LAYERS = ["ResNet50_layer1", "ResNet50_layer2", "ResNet50_layer4"]

    #   Analysis B: per-window ridge + extended partial RSA
    B_TARGETS = ["CLIP", "ResNet50_layer4"]
    B_CONTROLS = [
        ("full",                          []),
        ("partial_PixelPCA",              ["PixelPCA"]),
        ("partial_PixelPCA+layer1",       ["PixelPCA", "ResNet50_layer1"]),
        ("partial_PixelPCA+layer2",       ["PixelPCA", "ResNet50_layer2"]),
        ("partial_PixelPCA+layer1+layer2",["PixelPCA", "ResNet50_layer1", "ResNet50_layer2"]),
    ]

    # ── Per-subject sweep ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Per-subject sweep: A (fixed-model, 3 layers) + B (per-window partial RSA, 2 targets)")
    print(f"{'='*60}")

    A_curves = {layer: [] for layer in A_LAYERS}
    B_curves = {(tgt, ctl_name): [] for tgt in B_TARGETS for ctl_name, _ in B_CONTROLS}
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

            # ── Analysis A: fixed-model ridge per layer target ──────────────────
            train_mask_t = (times >= TRAIN_TMIN - 1e-6) & (times <= TRAIN_TMAX + 1e-6)
            feat_train = eeg_data[:, :, train_mask_t].mean(axis=2)
            A_scalers = {}
            A_ridges  = {}
            scaler_train = StandardScaler()
            feat_train_sc = scaler_train.fit_transform(feat_train)
            for layer in A_LAYERS:
                ridge = Ridge(alpha=1.0)
                ridge.fit(feat_train_sc, TARGET_EMBS[layer][y])
                A_ridges[layer] = ridge
            # Sweep
            A_subj_curves = {layer: np.zeros(n_windows) for layer in A_LAYERS}
            for w_idx, (tmin_w, tmax_w, _) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_data[:, :, t_mask].mean(axis=2)
                feat_sc = scaler_train.transform(feat)
                for layer in A_LAYERS:
                    preds = A_ridges[layer].predict(feat_sc)
                    cp = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                    ct = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        cp[lbl] += preds[i]; ct[lbl] += 1
                    m = ct > 0; cp[m] /= ct[m, None]
                    norms = np.linalg.norm(cp, axis=1, keepdims=True) + 1e-8
                    cn = cp / norms
                    sim = cn @ cn.T
                    r, _ = spearmanr(sim[tri], TARGET_RDMS[layer])
                    A_subj_curves[layer][w_idx] = r
            for layer in A_LAYERS:
                A_curves[layer].append(A_subj_curves[layer])

            # ── Analysis B: per-window ridge + extended partial RSA ─────────────
            B_subj = {(tgt, ctl_name): np.zeros(n_windows) for tgt in B_TARGETS for ctl_name, _ in B_CONTROLS}
            for w_idx, (tmin_w, tmax_w, _) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_data[:, :, t_mask].mean(axis=2)
                sc = StandardScaler(); feat_sc = sc.fit_transform(feat)
                pred_rdms = {}
                for tgt in B_TARGETS:
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(feat_sc, TARGET_EMBS[tgt][y])
                    preds = ridge.predict(feat_sc)
                    cp = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                    ct = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        cp[lbl] += preds[i]; ct[lbl] += 1
                    m = ct > 0; cp[m] /= ct[m, None]
                    norms = np.linalg.norm(cp, axis=1, keepdims=True) + 1e-8
                    cn = cp / norms
                    sim = cn @ cn.T
                    pred_rdms[tgt] = sim[tri]
                for tgt in B_TARGETS:
                    for ctl_name, ctl_list in B_CONTROLS:
                        r = partial_spearman_multi(
                            pred_rdms[tgt], TARGET_RDMS[tgt],
                            [CONTROL_RDMS[c] for c in ctl_list],
                        )
                        B_subj[(tgt, ctl_name)][w_idx] = r
            for key in B_subj:
                B_curves[key].append(B_subj[key])

            del eeg_data; gc.collect()
            successful.append(subj)
            elapsed = time.time() - t0
            # quick print
            l1_peak = A_subj_curves["ResNet50_layer1"][post_mask].max()
            l2_peak = A_subj_curves["ResNet50_layer2"][post_mask].max()
            l4_peak = A_subj_curves["ResNet50_layer4"][post_mask].max()
            print(f"  done ({elapsed:.0f}s) "
                  f"A peaks l1/l2/l4 = {l1_peak:.3f}/{l2_peak:.3f}/{l4_peak:.3f}", flush=True)
        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    print(f"\nSuccessful subjects: {n_subj}/{len(SUBJECTS)}")

    # ── Statistics helpers ──────────────────────────────────────────────────────
    def sign_flip(vals, n_perm=N_PERM, seed=RNG_SEED):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        obs = float(vals.mean())
        sd = float(vals.std(ddof=1))
        dz = obs / sd if sd > 0 else float("nan")
        rng_p = np.random.RandomState(seed)
        null = np.zeros(n_perm)
        for p in range(n_perm):
            signs = rng_p.choice([-1.0, 1.0], size=len(vals))
            null[p] = (vals * signs).mean()
        p_two = float(((np.abs(null) >= abs(obs)).sum() + 1) / (n_perm + 1))
        p_pos = float(((null >= obs).sum() + 1) / (n_perm + 1))
        return obs, float(sd), float(dz), p_two, p_pos


    # ── Analysis A summary ──────────────────────────────────────────────────────
    A_mat = {layer: np.stack(A_curves[layer]) for layer in A_LAYERS}
    rows_A = []
    for layer in A_LAYERS:
        mat = A_mat[layer]
        group_mean = mat.mean(axis=0)
        group_sem  = mat.std(axis=0, ddof=1) / np.sqrt(n_subj)
        peak_idx = int(np.argmax(group_mean))
        peak_r   = float(group_mean[peak_idx])
        peak_ms  = int(centers_ms[peak_idx])
        pre_mean_g  = float(group_mean[pre_mask].mean())
        post_mean_g = float(group_mean[post_mask].mean())
        subj_deltas = mat[:, post_mask].mean(axis=1) - mat[:, pre_mask].mean(axis=1)
        d_mean, d_sd, d_dz, d_p_two, d_p_pos = sign_flip(subj_deltas)
        rows_A.append({
            "layer": layer,
            "group_peak_r": peak_r, "group_peak_ms": peak_ms,
            "group_pre_mean": pre_mean_g, "group_post_mean": post_mean_g,
            "group_post_minus_pre": post_mean_g - pre_mean_g,
            "subj_mean_delta": d_mean, "subj_sd_delta": d_sd,
            "subj_d_z": d_dz, "subj_perm_p_two": d_p_two, "subj_perm_p_pos": d_p_pos,
        })

    # Save per-subject A data (compact)
    rows_A_full = []
    for i, subj in enumerate(successful):
        row = {"subject": subj}
        for layer in A_LAYERS:
            for w_idx, c in enumerate(centers_ms):
                row[f"{layer}_{int(c)}ms"] = A_mat[layer][i, w_idx]
        rows_A_full.append(row)
    pd.DataFrame(rows_A_full).to_csv(
        os.path.join(_RESULTS_DIR, "layerwise_temporal_comparison.csv"), index=False
    )

    # Group summary text
    with open(os.path.join(_RESULTS_DIR, "layerwise_temporal_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 76 + "\n")
        f.write("LAYER-WISE TEMPORAL COMPARISON (Fixed-model ridge, 50 subjects)\n")
        f.write("=" * 76 + "\n\n")
        f.write(f"Train window: {int(TRAIN_TMIN*1000)}-{int(TRAIN_TMAX*1000)} ms\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Pre interval: {STRICT_PRE} ms, Post interval: {POST_TARGET} ms\n\n")
        f.write(f"{'Layer':20s} {'peak_r':>8s} {'peak_ms':>8s} {'pre':>8s} {'post':>8s} "
                f"{'Δr':>8s} {'d_z':>6s} {'perm_p':>10s}\n")
        f.write("-" * 78 + "\n")
        for r in rows_A:
            f.write(f"{r['layer']:20s} {r['group_peak_r']:8.4f} {r['group_peak_ms']:8d} "
                    f"{r['group_pre_mean']:8.4f} {r['group_post_mean']:8.4f} "
                    f"{r['subj_mean_delta']:+8.4f} {r['subj_d_z']:+6.2f} "
                    f"{r['subj_perm_p_two']:10.4f}\n")
        f.write("\nEarly = layer1 (256ch after layer1 block, AAP, PCA->512 padded)\n")
        f.write("Mid   = layer2 (512ch after layer2 block, AAP, PCA->512)\n")
        f.write("Late  = layer4 (2048ch avgpool / penultimate, PCA->512)\n")

    # ── Analysis B summary ──────────────────────────────────────────────────────
    B_mat = {k: np.stack(v) for k, v in B_curves.items()}
    rows_B = []
    for tgt in B_TARGETS:
        for ctl_name, _ in B_CONTROLS:
            mat = B_mat[(tgt, ctl_name)]
            # post-vs-pre at the subject level
            subj_deltas = mat[:, post_mask].mean(axis=1) - mat[:, pre_mask].mean(axis=1)
            d_mean, d_sd, d_dz, d_p_two, d_p_pos = sign_flip(subj_deltas)
            # at +175ms
            vals_175 = mat[:, w175_idx]
            r_mean, r_sd, r_dz, r_p_two, r_p_pos = sign_flip(vals_175)
            rows_B.append({
                "target": tgt, "analysis": ctl_name,
                "post_vs_pre_delta": d_mean, "post_vs_pre_sd": d_sd,
                "post_vs_pre_dz":    d_dz,   "post_vs_pre_p_two": d_p_two,
                "r175_mean": r_mean, "r175_sd": r_sd,
                "r175_dz":   r_dz,   "r175_p_two": r_p_two,
            })

    df_B = pd.DataFrame(rows_B)
    df_B.to_csv(os.path.join(_RESULTS_DIR, "layerwise_partial_rsa.csv"), index=False)

    with open(os.path.join(_RESULTS_DIR, "layerwise_partial_rsa_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 92 + "\n")
        f.write("LAYER-WISE PARTIAL RSA (per-window ridge, 50 subjects)\n")
        f.write("=" * 92 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Post interval: {POST_TARGET} ms; Pre interval: {STRICT_PRE} ms; "
                f"r175 uses center at +175 ms\n\n")
        f.write(f"{'Target':18s} {'Analysis':34s} {'PvP Δ':>8s} {'d_z':>6s} {'p':>8s} "
                f"{'r@175':>8s} {'d_z':>6s} {'p':>8s}\n")
        f.write("-" * 96 + "\n")
        for r in rows_B:
            f.write(f"{r['target']:18s} {r['analysis']:34s} "
                    f"{r['post_vs_pre_delta']:8.4f} {r['post_vs_pre_dz']:+6.2f} "
                    f"{r['post_vs_pre_p_two']:8.4f} "
                    f"{r['r175_mean']:8.4f} {r['r175_dz']:+6.2f} {r['r175_p_two']:8.4f}\n")

        # Reductions from full to each partial
        f.write("\n--- Reductions relative to full (post-vs-pre Δ) ---\n")
        for tgt in B_TARGETS:
            f.write(f"\n{tgt}:\n")
            full_delta = [r for r in rows_B if r['target']==tgt and r['analysis']=='full'][0]['post_vs_pre_delta']
            f.write(f"  Full:                                   {full_delta:+.4f}\n")
            for ctl_name, _ in B_CONTROLS:
                if ctl_name == "full":
                    continue
                d = [r for r in rows_B if r['target']==tgt and r['analysis']==ctl_name][0]['post_vs_pre_delta']
                red = (1 - d / full_delta) * 100 if full_delta != 0 else float("nan")
                sig = [r for r in rows_B if r['target']==tgt and r['analysis']==ctl_name][0]['post_vs_pre_p_two']
                f.write(f"  {ctl_name:38s} {d:+.4f}  (reduction {red:+.1f}%, p={sig:.4f})\n")

    print(f"\nSaved: layerwise_temporal_comparison.csv")
    print(f"Saved: layerwise_temporal_summary.txt")
    print(f"Saved: layerwise_partial_rsa.csv")
    print(f"Saved: layerwise_partial_rsa_summary.txt")

    # ── Combined plot ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: layer-wise temporal curves (fixed-model)
    ax = axes[0]
    colors = {"ResNet50_layer1": "steelblue",
              "ResNet50_layer2": "darkorange",
              "ResNet50_layer4": "crimson"}
    labels = {"ResNet50_layer1": "layer1 (early)",
              "ResNet50_layer2": "layer2 (mid)",
              "ResNet50_layer4": "layer4 (late)"}
    for layer in A_LAYERS:
        mat = A_mat[layer]
        gm = mat.mean(axis=0)
        se = mat.std(axis=0, ddof=1) / np.sqrt(n_subj)
        ax.plot(centers_ms, gm, color=colors[layer], lw=2.0, label=labels[layer])
        ax.fill_between(centers_ms, gm - se, gm + se, color=colors[layer], alpha=0.15)
    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Group-mean r (fixed-model ridge)")
    ax.set_title(f"A. Layer-wise temporal curves (fixed-model, N={n_subj})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: partial RSA reduction bars
    ax = axes[1]
    ctl_order = [c for c, _ in B_CONTROLS]
    # Use post-vs-pre delta for bar heights
    colors_b = {"CLIP": "steelblue", "ResNet50_layer4": "crimson"}
    width = 0.35
    xs = np.arange(len(ctl_order))
    for i, tgt in enumerate(B_TARGETS):
        ys = []
        for ctl_name in ctl_order:
            r = [rr for rr in rows_B if rr['target']==tgt and rr['analysis']==ctl_name][0]
            ys.append(r['post_vs_pre_delta'])
        ax.bar(xs + (i - 0.5) * width, ys, width=width, color=colors_b[tgt], alpha=0.8,
               label={"CLIP": "CLIP", "ResNet50_layer4": "late (layer4)"}[tgt])
    ax.set_xticks(xs)
    ax.set_xticklabels([{"full": "full",
                         "partial_PixelPCA": "- Pix",
                         "partial_PixelPCA+layer1": "- Pix - L1",
                         "partial_PixelPCA+layer2": "- Pix - L2",
                         "partial_PixelPCA+layer1+layer2": "- Pix - L1 - L2"}[c]
                        for c in ctl_order], fontsize=8, rotation=15)
    ax.set_ylabel("Post-vs-pre Δr (group mean of subject deltas)")
    ax.set_title("B. Partial RSA: post-vs-pre after progressive controls")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    plot_path = os.path.join(_RESULTS_DIR, "layerwise_hierarchy_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # ── Quick console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print("\nANALYSIS A (layer-wise fixed-model):")
    for r in rows_A:
        print(f"  {r['layer']:20s} peak={r['group_peak_r']:+.4f}@{r['group_peak_ms']:+4d}ms  "
              f"delta={r['subj_mean_delta']:+.4f}  d_z={r['subj_d_z']:+.2f}  "
              f"p={r['subj_perm_p_two']:.4f}")

    print("\nANALYSIS B (partial RSA, post-vs-pre delta):")
    for tgt in B_TARGETS:
        print(f"\n  {tgt}:")
        for ctl_name, _ in B_CONTROLS:
            r = [rr for rr in rows_B if rr['target']==tgt and rr['analysis']==ctl_name][0]
            print(f"    {ctl_name:34s} delta={r['post_vs_pre_delta']:+.4f}  "
                  f"d_z={r['post_vs_pre_dz']:+.2f}  p={r['post_vs_pre_p_two']:.4f}")

    print("=" * 60)
    print("DONE")


