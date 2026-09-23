"""Historical submission workflow; call run explicitly. See docs/CODE_ORIGINS.md."""


def run(settings):
    """Execute the original scientific workflow with configured paths (not a smoke test)."""
    from src.runtime import (require_dataset, resolve_path, release_root,
                             image_paths, DATASET_ERROR)
    require_dataset(settings)
    """
    Partial RSA / Variance Partitioning: Deep model residual after controlling PixelPCA.

    For each subject, at each time window:
      1. Ridge from EEG → CLIP (or ResNet50) embeddings → predicted RDM
      2. Ridge from EEG → PixelPCA embeddings → predicted RDM (pixel-level control)
      3. Partial correlation: correlation between predicted deep-model RDM and
         target deep-model RDM, after regressing out PixelPCA RDM from both.

    This tests: does the deep-model alignment persist after controlling for
    pixel-level structure?

    Two focal analyses:
      A. +175 ms single window
      B. +125 to +300 ms post interval (averaged)

    Statistical unit: subject. Sign-flip permutation (5,000) on group-mean residual.

    Outputs:
      results/partial_rsa_results.csv
      results/partial_rsa_summary.txt
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
    SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
    BASE = str(resolve_path(settings, "DATA_ROOT"))
    IMAGES_ROOT = str(resolve_path(settings, "IMAGES_ROOT"))
    EMBED_DIM = 512
    WIN_MS = 50; STEP_MS = 25
    TMIN_BASE = -0.1; TMAX_BASE = 0.496
    N_PERM = 5000; RNG_SEED = 42

    STRICT_PRE = [-75, -50, -25]
    POST_TARGET = [125, 150, 175, 200, 225, 250, 275, 300]

    # ── Windows ──────────────────────────────────────────────────────────────────
    win_s = WIN_MS / 1000.0; step_s = STEP_MS / 1000.0
    windows = []
    t = TMIN_BASE
    while t + win_s <= TMAX_BASE + 1e-9:
        windows.append((round(t, 4), round(t + win_s, 4), round((t + win_s / 2) * 1000)))
        t += step_s
    n_windows = len(windows)
    centers_ms = np.array([w[2] for w in windows])

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

    print("Computing Pixel-PCA embeddings...")
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

    # ── Target RDMs ──────────────────────────────────────────────────────────────
    tri = np.triu(np.ones((n_concepts, n_concepts), dtype=bool), k=1)

    TARGET_EMBS = {"CLIP": clip_embs_np, "ResNet50": resnet_embs_np}
    CONTROL_EMBS = pixel_pca_np

    def compute_rdm_vec(embs):
        sim = embs @ embs.T
        return sim[tri]

    TARGET_RDMS = {name: compute_rdm_vec(embs) for name, embs in TARGET_EMBS.items()}
    CONTROL_RDM = compute_rdm_vec(CONTROL_EMBS)

    # ── Partial correlation helper ───────────────────────────────────────────────
    def partial_spearman(x, y, z):
        """Spearman partial correlation between x and y, controlling for z."""
        rx = rankdata(x).astype(np.float64)
        ry = rankdata(y).astype(np.float64)
        rz = rankdata(z).astype(np.float64)
        # Regress z out of x and y
        rx -= rx.mean(); ry -= ry.mean(); rz -= rz.mean()
        zz = np.dot(rz, rz)
        if zz < 1e-12:
            # z is constant, partial = full
            num = np.dot(rx, ry)
            return num / (np.sqrt(np.dot(rx, rx)) * np.sqrt(np.dot(ry, ry)) + 1e-12)
        rx_res = rx - rz * (np.dot(rx, rz) / zz)
        ry_res = ry - rz * (np.dot(ry, rz) / zz)
        num = np.dot(rx_res, ry_res)
        denom = np.sqrt(np.dot(rx_res, rx_res)) * np.sqrt(np.dot(ry_res, ry_res))
        return num / (denom + 1e-12)

    # ── Per-subject sweep ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Per-subject partial RSA: deep models controlling for PixelPCA")
    print(f"{'='*70}")

    # Store: {target: {metric: list of per-subject values}}
    all_full = {name: [] for name in TARGET_EMBS}      # full r per window
    all_partial = {name: [] for name in TARGET_EMBS}    # partial r per window
    all_pixel = []                                       # pixel r per window
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

            subj_full = {name: np.zeros(n_windows) for name in TARGET_EMBS}
            subj_partial = {name: np.zeros(n_windows) for name in TARGET_EMBS}
            subj_pixel = np.zeros(n_windows)

            for w_idx, (tmin_w, tmax_w, center) in enumerate(windows):
                t_mask = (times >= tmin_w - 1e-6) & (times <= min(tmax_w, TMAX_BASE) + 1e-6)
                feat = eeg_data[:, :, t_mask].mean(axis=2)
                scaler = StandardScaler()
                feat_sc = scaler.fit_transform(feat)

                # Pixel-PCA ridge
                ridge_pix = Ridge(alpha=1.0)
                ridge_pix.fit(feat_sc, CONTROL_EMBS[y])
                preds_pix = ridge_pix.predict(feat_sc)
                concept_pix = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                counts = np.zeros(n_concepts, dtype=np.float32)
                for i, lbl in enumerate(y):
                    concept_pix[lbl] += preds_pix[i]
                    counts[lbl] += 1
                mask_c = counts > 0
                concept_pix[mask_c] /= counts[mask_c, None]
                norms_p = np.linalg.norm(concept_pix, axis=1, keepdims=True) + 1e-8
                pred_pix_rdm = (concept_pix / norms_p) @ (concept_pix / norms_p).T
                pred_pix_vec = pred_pix_rdm[tri]

                r_pix, _ = spearmanr(pred_pix_vec, CONTROL_RDM)
                subj_pixel[w_idx] = r_pix

                for name in TARGET_EMBS:
                    target_embs = TARGET_EMBS[name]
                    ridge_deep = Ridge(alpha=1.0)
                    ridge_deep.fit(feat_sc, target_embs[y])
                    preds_deep = ridge_deep.predict(feat_sc)

                    concept_deep = np.zeros((n_concepts, EMBED_DIM), dtype=np.float32)
                    counts_d = np.zeros(n_concepts, dtype=np.float32)
                    for i, lbl in enumerate(y):
                        concept_deep[lbl] += preds_deep[i]
                        counts_d[lbl] += 1
                    mask_d = counts_d > 0
                    concept_deep[mask_d] /= counts_d[mask_d, None]
                    norms_d = np.linalg.norm(concept_deep, axis=1, keepdims=True) + 1e-8
                    pred_deep_rdm = (concept_deep / norms_d) @ (concept_deep / norms_d).T
                    pred_deep_vec = pred_deep_rdm[tri]

                    # Full Spearman
                    r_full, _ = spearmanr(pred_deep_vec, TARGET_RDMS[name])
                    subj_full[name][w_idx] = r_full

                    # Partial Spearman: controlling for PixelPCA RDM
                    r_partial = partial_spearman(pred_deep_vec, TARGET_RDMS[name], CONTROL_RDM)
                    subj_partial[name][w_idx] = r_partial

            del eeg_data; gc.collect()

            for name in TARGET_EMBS:
                all_full[name].append(subj_full[name])
                all_partial[name].append(subj_partial[name])
            all_pixel.append(subj_pixel)
            successful.append(subj)

            elapsed = time.time() - t0
            clip_f = subj_full['CLIP'].max()
            clip_p = subj_partial['CLIP'].max()
            print(f"  done ({elapsed:.0f}s) CLIP full={clip_f:.4f} partial={clip_p:.4f}", flush=True)

        except Exception as e:
            raise RuntimeError(f"Participant {subj} failed; cohort will not be reduced.") from e

    n_subj = len(successful)
    for name in TARGET_EMBS:
        all_full[name] = np.array(all_full[name])
        all_partial[name] = np.array(all_partial[name])
    all_pixel = np.array(all_pixel)

    # ── Statistics ───────────────────────────────────────────────────────────────
    pre_mask = np.isin(centers_ms, STRICT_PRE)
    post_mask = np.isin(centers_ms, POST_TARGET)
    w175_idx = int(np.argmin(np.abs(centers_ms - 175)))

    def sign_flip_test(vals, n_perm=5000, seed=42):
        rng_t = np.random.RandomState(seed)
        obs = vals.mean()
        null = np.zeros(n_perm)
        for p in range(n_perm):
            signs = rng_t.choice([-1.0, 1.0], size=len(vals))
            null[p] = (vals * signs).mean()
        p_val = (np.sum(null >= obs) + 1) / (n_perm + 1)
        std_d = vals.std(ddof=1)
        d = obs / std_d if std_d > 0 else 0.0
        return {"mean": float(obs), "d": float(d), "p": float(p_val), "n": len(vals)}

    def post_pre_test(data, n_perm=5000, seed=42):
        pre = data[:, pre_mask].mean(axis=1)
        post = data[:, post_mask].mean(axis=1)
        delta = post - pre
        return sign_flip_test(delta, n_perm, seed)

    # ── Results table ────────────────────────────────────────────────────────────
    rows = []
    for name in TARGET_EMBS:
        # Full: post-vs-pre
        res_full = post_pre_test(all_full[name])
        res_full["target"] = name
        res_full["type"] = "full"
        res_full["metric"] = "post_vs_pre_delta"
        rows.append(res_full)

        # Partial: post-vs-pre
        res_partial = post_pre_test(all_partial[name])
        res_partial["target"] = name
        res_partial["type"] = "partial_controlling_PixelPCA"
        res_partial["metric"] = "post_vs_pre_delta"
        rows.append(res_partial)

        # Full: at +175ms
        vals_175_full = all_full[name][:, w175_idx]
        res_175f = sign_flip_test(vals_175_full)
        res_175f["target"] = name
        res_175f["type"] = "full"
        res_175f["metric"] = "r_at_175ms"
        rows.append(res_175f)

        # Partial: at +175ms
        vals_175_partial = all_partial[name][:, w175_idx]
        res_175p = sign_flip_test(vals_175_partial)
        res_175p["target"] = name
        res_175p["type"] = "partial_controlling_PixelPCA"
        res_175p["metric"] = "r_at_175ms"
        rows.append(res_175p)

    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(_RESULTS_DIR, "partial_rsa_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Summary ──────────────────────────────────────────────────────────────────
    summary_path = os.path.join(_RESULTS_DIR, "partial_rsa_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("PARTIAL RSA: DEEP MODEL RESIDUAL AFTER CONTROLLING PIXELPCA\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Subjects: {n_subj}\n")
        f.write(f"Control: PixelPCA RDM (partialled out via rank regression)\n")
        f.write(f"Permutations: {N_PERM}\n\n")

        f.write(f"{'Target':10s} {'Type':30s} {'Metric':20s} {'Mean':>8s} {'d':>8s} {'p':>8s}\n")
        f.write("─" * 86 + "\n")
        for _, row in df_results.iterrows():
            f.write(f"{row['target']:10s} {row['type']:30s} {row['metric']:20s} "
                    f"{row['mean']:8.4f} {row['d']:8.2f} {row['p']:8.4f}\n")

        f.write(f"\n─── KEY QUESTION ───\n")
        for name in TARGET_EMBS:
            full_pv = [r for r in rows if r['target']==name and r['type']=='full' and r['metric']=='post_vs_pre_delta'][0]
            part_pv = [r for r in rows if r['target']==name and r['type']=='partial_controlling_PixelPCA' and r['metric']=='post_vs_pre_delta'][0]
            f.write(f"\n{name}:\n")
            f.write(f"  Full post-vs-pre:    delta={full_pv['mean']:.4f}, d={full_pv['d']:.2f}, p={full_pv['p']:.4f}\n")
            f.write(f"  Partial (−PixPCA):   delta={part_pv['mean']:.4f}, d={part_pv['d']:.2f}, p={part_pv['p']:.4f}\n")
            if part_pv['p'] < 0.05:
                f.write(f"  → Residual is SIGNIFICANT. Deep model retains post-stimulus effect after controlling PixelPCA.\n")
            else:
                f.write(f"  → Residual is NOT SIGNIFICANT. Deep model effect may be largely explained by pixel-level structure.\n")

    print(f"Saved: {summary_path}")
    print("\n=== DONE ===")


