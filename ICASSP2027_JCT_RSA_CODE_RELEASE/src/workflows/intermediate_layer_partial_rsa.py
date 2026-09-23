"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Intermediate-layer variance partitioning / partial RSA.

    Tests whether deep-model (CLIP / ResNet50-penultimate) residual alignment
    persists after controlling for BOTH PixelPCA AND ResNet50 intermediate layers.

    ResNet50 layers used:
      - layer2 output (mid-level, ~512-dim after PCA)
      - layer4/avgpool output (penultimate, late-level, ~512-dim after PCA)

    Partial RSA controls:
      A. partial(CLIP | PixelPCA)                    — already done
      B. partial(CLIP | PixelPCA + ResNet50-layer2)  — NEW
      C. partial(ResNet50-late | PixelPCA + ResNet50-layer2) — NEW

    Reports at +175ms and post interval (+125 to +300ms).
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

    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    EMBED_DIM = 512
    WIN_MS = 50; STEP_MS = 25
    TMIN_BASE = -0.1; TMAX_BASE = 0.496
    N_PERM = 5000; RNG_SEED = 42
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
    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)
    w175_idx = int(np.argmin(np.abs(centers_ms - 175)))

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

    print("Computing ResNet-50 layer2 + penultimate embeddings...")
    from torchvision.models import resnet50, ResNet50_Weights
    resnet_weights = ResNet50_Weights.IMAGENET1K_V2
    resnet_model = resnet50(weights=resnet_weights).to(device)
    resnet_model.eval()
    resnet_preprocess = resnet_weights.transforms()

    resnet_layer2_raw = np.zeros((n_concepts, 512), dtype=np.float32)  # layer2 output channels
    resnet_late_raw = np.zeros((n_concepts, 2048), dtype=np.float32)

    for i, name in enumerate(concept_names):
        paths = get_image_paths(name)
        if not paths: raise FileNotFoundError(DATASET_ERROR)
        imgs = torch.stack([resnet_preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
        with torch.no_grad():
            x = resnet_model.conv1(imgs); x = resnet_model.bn1(x); x = resnet_model.relu(x)
            x = resnet_model.maxpool(x)
            x = resnet_model.layer1(x)
            x = resnet_model.layer2(x)
            # layer2 output: global average pool to get feature vector
            l2_feat = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
            resnet_layer2_raw[i] = l2_feat.float().mean(dim=0).cpu().numpy()
            x = resnet_model.layer3(x)
            x = resnet_model.layer4(x)
            x = resnet_model.avgpool(x)
            x = torch.flatten(x, 1)
            resnet_late_raw[i] = x.float().mean(dim=0).cpu().numpy()

    del resnet_model; torch.cuda.empty_cache(); gc.collect()

    # PCA to 512-dim
    pca_l2 = PCA(n_components=EMBED_DIM, random_state=RNG_SEED)
    resnet_l2_np = pca_l2.fit_transform(resnet_layer2_raw).astype(np.float32)
    resnet_l2_np /= (np.linalg.norm(resnet_l2_np, axis=1, keepdims=True) + 1e-8)
    print(f"ResNet layer2: {resnet_l2_np.shape}, PCA var: {pca_l2.explained_variance_ratio_.sum():.3f}")

    pca_late = PCA(n_components=EMBED_DIM, random_state=RNG_SEED)
    resnet_late_np = pca_late.fit_transform(resnet_late_raw).astype(np.float32)
    resnet_late_np /= (np.linalg.norm(resnet_late_np, axis=1, keepdims=True) + 1e-8)
    print(f"ResNet penult: {resnet_late_np.shape}, PCA var: {pca_late.explained_variance_ratio_.sum():.3f}")

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

    # RDMs
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)
    def rdm_vec(embs):
        return (embs @ embs.T)[tri]

    TARGET_EMBS = {"CLIP": clip_embs_np, "ResNet50_late": resnet_late_np}
    TARGET_RDMS = {n: rdm_vec(e) for n, e in TARGET_EMBS.items()}
    CONTROL_RDMS = {
        "PixelPCA": rdm_vec(pixel_pca_np),
        "ResNet50_layer2": rdm_vec(resnet_l2_np),
    }

    # Partial correlation helpers
    def partial_spearman_single(x, y, z):
        """Partial Spearman r(x,y | z) where z is a single control vector."""
        rx = rankdata(x).astype(np.float64); rx -= rx.mean()
        ry = rankdata(y).astype(np.float64); ry -= ry.mean()
        rz = rankdata(z).astype(np.float64); rz -= rz.mean()
        zz = np.dot(rz, rz)
        if zz < 1e-12: return np.dot(rx, ry) / (np.sqrt(np.dot(rx, rx)) * np.sqrt(np.dot(ry, ry)) + 1e-12)
        rx_res = rx - rz * (np.dot(rx, rz) / zz)
        ry_res = ry - rz * (np.dot(ry, rz) / zz)
        return np.dot(rx_res, ry_res) / (np.sqrt(np.dot(rx_res, rx_res)) * np.sqrt(np.dot(ry_res, ry_res)) + 1e-12)

    def partial_spearman_multi(x, y, Z_list):
        """Partial Spearman r(x,y | Z1, Z2, ...) via sequential regression."""
        rx = rankdata(x).astype(np.float64); rx -= rx.mean()
        ry = rankdata(y).astype(np.float64); ry -= ry.mean()
        Z = np.column_stack([rankdata(z).astype(np.float64) for z in Z_list])
        Z -= Z.mean(axis=0)
        # OLS residuals
        ZtZ = Z.T @ Z
        try:
            ZtZ_inv = np.linalg.inv(ZtZ + 1e-10 * np.eye(len(ZtZ)))
        except:
            return partial_spearman_single(x, y, Z_list[0])
        rx_res = rx - Z @ (ZtZ_inv @ (Z.T @ rx))
        ry_res = ry - Z @ (ZtZ_inv @ (Z.T @ ry))
        denom = np.sqrt(np.dot(rx_res, rx_res)) * np.sqrt(np.dot(ry_res, ry_res))
        return np.dot(rx_res, ry_res) / (denom + 1e-12)

    # ── Per-subject sweep ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Intermediate-layer partial RSA")
    print(f"{'='*70}")

    ANALYSES = [
        ("CLIP", "full", []),
        ("CLIP", "partial_PixelPCA", ["PixelPCA"]),
        ("CLIP", "partial_PixelPCA+layer2", ["PixelPCA", "ResNet50_layer2"]),
        ("ResNet50_late", "full", []),
        ("ResNet50_late", "partial_PixelPCA", ["PixelPCA"]),
        ("ResNet50_late", "partial_PixelPCA+layer2", ["PixelPCA", "ResNet50_layer2"]),
    ]

    # {(target, analysis_type): list of per-subject (n_windows,) arrays}
    all_r = {(t, a): [] for t, a, _ in ANALYSES}
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

            subj_results = {k: np.zeros(n_windows) for k in all_r}

            for w_idx, (tmin_w, tmax_w, _) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_data[:, :, t_mask].mean(axis=2)
                scaler = StandardScaler()
                feat_sc = scaler.fit_transform(feat)

                # Compute predicted RDMs for all target spaces
                pred_rdms = {}
                for tgt_name, tgt_embs in TARGET_EMBS.items():
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(feat_sc, tgt_embs[y])
                    preds = ridge.predict(feat_sc)
                    cp = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                    ct = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        cp[lbl] += preds[i]; ct[lbl] += 1
                    m = ct > 0; cp[m] /= ct[m, None]
                    norms = np.linalg.norm(cp, axis=1, keepdims=True) + 1e-8
                    pred_rdms[tgt_name] = (cp / norms) @ (cp / norms).T
                    pred_rdms[tgt_name] = pred_rdms[tgt_name][tri]

                for tgt_name, analysis_type, controls in ANALYSES:
                    pred_vec = pred_rdms[tgt_name]
                    target_vec = TARGET_RDMS[tgt_name]
                    if not controls:
                        r, _ = spearmanr(pred_vec, target_vec)
                    elif len(controls) == 1:
                        r = partial_spearman_single(pred_vec, target_vec, CONTROL_RDMS[controls[0]])
                    else:
                        r = partial_spearman_multi(pred_vec, target_vec,
                                                  [CONTROL_RDMS[c] for c in controls])
                    subj_results[(tgt_name, analysis_type)][w_idx] = r

            del eeg_data; gc.collect()
            for k in all_r:
                all_r[k].append(subj_results[k])
            successful.append(subj)
            elapsed = time.time() - t0
            print(f"  done ({elapsed:.0f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    for k in all_r:
        all_r[k] = np.array(all_r[k])

    # ── Results ──────────────────────────────────────────────────────────────────
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
    for tgt_name, analysis_type, controls in ANALYSES:
        data = all_r[(tgt_name, analysis_type)]
        # Post-vs-pre
        deltas = data[:, post_mask].mean(axis=1) - data[:, pre_mask].mean(axis=1)
        d_mean, d_d, d_p = sign_flip_test(deltas, N_PERM, RNG_SEED)
        # At 175ms
        vals_175 = data[:, w175_idx]
        r_mean, r_d, r_p = sign_flip_test(vals_175, N_PERM, RNG_SEED)
        rows.append({
            "target": tgt_name, "analysis": analysis_type,
            "post_vs_pre_delta": d_mean, "post_vs_pre_d": d_d, "post_vs_pre_p": d_p,
            "r_175ms": r_mean, "r_175ms_d": r_d, "r_175ms_p": r_p,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "intermediate_layer_partial_rsa.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    summary_path = os.path.join(_RESULTS_DIR, "intermediate_layer_partial_rsa_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("INTERMEDIATE-LAYER PARTIAL RSA / VARIANCE PARTITIONING\n")
        f.write("=" * 85 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Controls: PixelPCA, ResNet50-layer2 (intermediate)\n\n")
        f.write(f"{'Target':15s} {'Analysis':25s} {'PvP delta':>10s} {'d':>6s} {'p':>8s} "
                f"{'r@175':>8s} {'d':>6s} {'p':>8s}\n")
        f.write("─" * 88 + "\n")
        for _, r in df.iterrows():
            f.write(f"{r['target']:15s} {r['analysis']:25s} "
                    f"{r['post_vs_pre_delta']:10.4f} {r['post_vs_pre_d']:6.2f} {r['post_vs_pre_p']:8.4f} "
                    f"{r['r_175ms']:8.4f} {r['r_175ms_d']:6.2f} {r['r_175ms_p']:8.4f}\n")

        f.write(f"\n─── KEY FINDINGS ───\n")
        for tgt in ["CLIP", "ResNet50_late"]:
            full = [r for r in rows if r['target']==tgt and r['analysis']=='full'][0]
            p1 = [r for r in rows if r['target']==tgt and r['analysis']=='partial_PixelPCA'][0]
            p2 = [r for r in rows if r['target']==tgt and r['analysis']=='partial_PixelPCA+layer2'][0]
            f.write(f"\n{tgt}:\n")
            f.write(f"  Full:                    delta={full['post_vs_pre_delta']:.4f}, d={full['post_vs_pre_d']:.2f}\n")
            f.write(f"  − PixelPCA:              delta={p1['post_vs_pre_delta']:.4f}, d={p1['post_vs_pre_d']:.2f}\n")
            f.write(f"  − PixelPCA − layer2:     delta={p2['post_vs_pre_delta']:.4f}, d={p2['post_vs_pre_d']:.2f}\n")
            red1 = (1 - p1['post_vs_pre_delta'] / full['post_vs_pre_delta']) * 100
            red2 = (1 - p2['post_vs_pre_delta'] / full['post_vs_pre_delta']) * 100
            f.write(f"  Reduction (−PixPCA):     {red1:.1f}%\n")
            f.write(f"  Reduction (−PixPCA−l2):  {red2:.1f}%\n")
            if p2['post_vs_pre_p'] < 0.05:
                f.write(f"  → Residual SIGNIFICANT after controlling PixelPCA + layer2\n")
            else:
                f.write(f"  → Residual NOT SIGNIFICANT after controlling PixelPCA + layer2\n")

    print(f"Saved: {summary_path}")
    print("\n=== DONE ===")


