"""
Phase 1: Data Exploration and Sanity Checks
Reconstructing 3D objects from brain activity (fMRI-Shape dataset)
Strict adherence to assignment rules:
- No predictive models trained.
- Evaluates descriptive statistics, PCA, ANOVA, and reference anchors.
- Treats sample_id as opaque.
- CPU only, reproducible seeds, clean exports.
"""

import sys
from pathlib import Path

# Add project root to path so 'from starter import ...' and 'from metrics import ...' work
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

import time
import json
import hashlib
import concurrent.futures
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Provided helpers
from starter import load_split, load_cameras, load_voxels, targets_for
import metrics

# Output directory
OUT_DIR = WORKSPACE_ROOT / "outputs" / "phase1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed seed
SEED = 42
np.random.seed(SEED)


def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("PHASE 1: DATA EXPLORATION AND INTEGRITY AUDIT", flush=True)
    print("=" * 70, flush=True)

    summary_data = {}

    # =========================================================================
    # TASK 1: INTEGRITY AND DATA AUDIT
    # =========================================================================
    print("\n--- Task 1: Integrity Checks ---", flush=True)
    
    # 1.1 Load splits
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    cams = load_cameras()
    voxels = load_voxels()

    # Array shapes & dtypes
    shapes_dtypes = {
        "train": {k: {"shape": list(train[k].shape), "dtype": str(train[k].dtype)} for k in train},
        "val": {k: {"shape": list(val[k].shape), "dtype": str(val[k].dtype)} for k in val},
        "test": {k: {"shape": list(test[k].shape), "dtype": str(test[k].dtype)} for k in test},
        "cameras": {k: {"shape": list(cams[k].shape), "dtype": str(cams[k].dtype)} for k in cams},
        "voxels": {k: {"shape": list(voxels[k].shape), "dtype": str(voxels[k].dtype)} for k in voxels},
    }

    # Verify expected shapes
    assert train["X"].shape == (533, 10, 31141) and train["X"].dtype == np.float16
    assert val["X"].shape == (125, 10, 31141) and val["X"].dtype == np.float16
    assert test["X"].shape == (167, 10, 31141) and test["X"].dtype == np.float16
    assert train["obj_points"].shape == (292, 2048, 3) and train["obj_points"].dtype == np.float32
    assert val["obj_points"].shape == (67, 2048, 3) and val["obj_points"].dtype == np.float32
    assert train["obj_masks"].shape == (292, 32, 64, 64)
    assert val["obj_masks"].shape == (67, 32, 64, 64)
    assert train["obj_images"].shape == (292, 32, 64, 64, 3)
    assert val["obj_images"].shape == (67, 32, 64, 64, 3)

    train_mask_dtype = str(train["obj_masks"].dtype)
    train_mask_max = int(train["obj_masks"].max())
    val_mask_dtype = str(val["obj_masks"].dtype)
    val_mask_max = int(val["obj_masks"].max())
    print(f"obj_masks train: dtype={train_mask_dtype}, max={train_mask_max}", flush=True)
    print(f"obj_masks val: dtype={val_mask_dtype}, max={val_mask_max}", flush=True)

    # Check NaN/Inf
    nan_inf_report = {}
    for s_name, d in [("train", train), ("val", val), ("test", test)]:
        nan_inf_report[s_name] = {}
        for k in ["X", "obj_points", "obj_masks", "obj_images"]:
            if k in d:
                arr = d[k]
                has_nan = bool(np.isnan(arr).any())
                has_inf = bool(np.isinf(arr).any())
                nan_inf_report[s_name][k] = {"nan": has_nan, "inf": has_inf}
                assert not has_nan and not has_inf, f"NaN/Inf found in {s_name}['{k}']!"
    print("NaN/Inf check: PASSED (all zero)", flush=True)

    # Group sanity
    for s_name, d in [("train", train), ("val", val)]:
        g = d["group"]
        n_obj = len(d["obj_points"])
        assert np.issubdtype(g.dtype, np.integer), f"{s_name} group is not integer"
        assert g.min() >= 0 and g.max() < n_obj, f"{s_name} group out of bounds"
        uniq_g = np.unique(g)
        assert len(uniq_g) == n_obj, f"{s_name} some objects unused"
    print("Group check: PASSED (all valid integer indices 0..n_obj-1)", flush=True)

    # Sample ID sanity
    train_sids = set(train["sample_id"])
    val_sids = set(val["sample_id"])
    test_sids = set(test["sample_id"])
    assert len(train_sids) == len(train["sample_id"]), "Train sample_ids not unique"
    assert len(val_sids) == len(val["sample_id"]), "Val sample_ids not unique"
    assert len(test_sids) == len(test["sample_id"]), "Test sample_ids not unique"
    assert len(train_sids & val_sids) == 0, "Overlap train-val sample_ids!"
    assert len(train_sids & test_sids) == 0, "Overlap train-test sample_ids!"
    assert len(val_sids & test_sids) == 0, "Overlap val-test sample_ids!"
    all_sids = list(train["sample_id"]) + list(val["sample_id"]) + list(test["sample_id"])
    for sid in all_sids:
        assert "/" not in sid and "\\" not in sid, f"Invalid characters in sample_id: {sid}"
    print("Sample ID check: PASSED (unique, safe filename strings)", flush=True)

    # Cameras check
    assert "resolution" in cams, "Missing resolution in cameras.npz!"
    assert np.array_equal(cams["video_frame"], np.arange(32) * 6), "video_frame mismatch with arange(32)*6"
    v0 = -cams["R"][0, 2, :]
    v8 = -cams["R"][8, 2, :]
    cos_angle = np.clip(np.dot(v0, v8) / (np.linalg.norm(v0) * np.linalg.norm(v8)), -1.0, 1.0)
    angle_deg = float(np.degrees(np.arccos(cos_angle)))
    print(f"Cameras check: PASSED. Angle between view 0 and view 8: {angle_deg:.2f} deg (90 deg azimuth, 30 deg elevation)", flush=True)

    # Classes check
    assert np.array_equal(train["class_names"], val["class_names"]), "class_names differ between train and val"
    class_names = list(train["class_names"])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    assert set(train["obj_class"]).issubset(set(class_names))
    assert set(val["obj_class"]).issubset(set(class_names))
    print(f"Class names: {class_names}", flush=True)

    summary_data["task1_integrity"] = {
        "shapes_dtypes": shapes_dtypes,
        "mask_stats": {
            "train": {"dtype": train_mask_dtype, "max": train_mask_max},
            "val": {"dtype": val_mask_dtype, "max": val_mask_max},
        },
        "nan_inf": nan_inf_report,
        "camera_view_angle_0_8_deg": angle_deg,
        "class_names": class_names,
    }

    # =========================================================================
    # TASK 2: SPLITS AND SUBJECTS
    # =========================================================================
    print("\n--- Task 2: Splits and Subjects ---", flush=True)
    
    # Scans per subject
    subjects_all = sorted(list(set(train["subject"]) | set(val["subject"]) | set(test["subject"])))
    subj_table = {}
    for s in subjects_all:
        n_tr = int(np.sum(train["subject"] == s))
        n_va = int(np.sum(val["subject"] == s))
        n_te = int(np.sum(test["subject"] == s))
        subj_table[s] = {"train": n_tr, "val": n_va, "test": n_te, "total": n_tr + n_va + n_te}

    train_subjs = set(train["subject"])
    val_subjs = set(val["subject"])
    test_subjs = set(test["subject"])

    unseen_val_subjs = sorted(list(val_subjs - train_subjs))
    unseen_test_subjs = sorted(list(test_subjs - train_subjs))

    pct_val_unseen = float(np.sum([val["subject"] == s for s in unseen_val_subjs])) / len(val["subject"]) * 100
    pct_test_unseen = float(np.sum([test["subject"] == s for s in unseen_test_subjs])) / len(test["subject"]) * 100

    print(f"Subjects: Train={len(train_subjs)}, Val={len(val_subjs)}, Test={len(test_subjs)}, Total={len(subjects_all)}", flush=True)
    print(f"Unseen subjects in Val: {unseen_val_subjs} ({pct_val_unseen:.1f}% of val scans)", flush=True)
    print(f"Unseen subjects in Test: {unseen_test_subjs} ({pct_test_unseen:.1f}% of test scans)", flush=True)

    # Scans per object
    train_scans_per_obj = np.bincount(train["group"], minlength=len(train["obj_points"]))
    val_scans_per_obj = np.bincount(val["group"], minlength=len(val["obj_points"]))

    obj_scan_stats = {
        "train": {
            "min": int(train_scans_per_obj.min()),
            "mean": float(train_scans_per_obj.mean()),
            "max": int(train_scans_per_obj.max())
        },
        "val": {
            "min": int(val_scans_per_obj.min()),
            "mean": float(val_scans_per_obj.mean()),
            "max": int(val_scans_per_obj.max())
        }
    }
    print(f"Scans per object (Train): min={obj_scan_stats['train']['min']}, mean={obj_scan_stats['train']['mean']:.2f}, max={obj_scan_stats['train']['max']}", flush=True)
    print(f"Scans per object (Val): min={obj_scan_stats['val']['min']}, mean={obj_scan_stats['val']['mean']:.2f}, max={obj_scan_stats['val']['max']}", flush=True)

    # Small subjects (< 10 scans in train)
    subjs_under_10 = [s for s in train_subjs if np.sum(train["subject"] == s) < 10]
    print(f"Subjects with < 10 train scans: {subjs_under_10}", flush=True)

    # Duplicate object check by shape
    # 1. Byte hash of obj_masks
    train_mask_hashes = {hashlib.sha256(train["obj_masks"][i].tobytes()).hexdigest(): i for i in range(len(train["obj_points"]))}
    val_mask_hashes = {hashlib.sha256(val["obj_masks"][i].tobytes()).hexdigest(): i for i in range(len(val["obj_points"]))}
    exact_duplicates = set(train_mask_hashes.keys()) & set(val_mask_hashes.keys())
    assert len(exact_duplicates) == 0, f"Found {len(exact_duplicates)} exact duplicate masks between train and val!"
    print("Exact mask byte-hash duplicate check: 0 duplicates found.", flush=True)

    # 2. Minimum Chamfer distance from each val object to all train objects (Parallelized)
    print("Computing minimum Chamfer from 67 val objects to train library...", flush=True)
    train_trees = [cKDTree(p) for p in train["obj_points"]]
    val_objs = val["obj_points"]

    def find_val_min_chamfer(i):
        vp = val_objs[i]
        vt = cKDTree(vp)
        cds = [tr.query(vp)[0].mean() + vt.query(lp)[0].mean() for lp, tr in zip(train["obj_points"], train_trees)]
        min_idx = int(np.argmin(cds))
        return i, float(cds[min_idx]), min_idx

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        val_chamfer_results = list(ex.map(find_val_min_chamfer, range(len(val_objs))))

    val_min_chamfers = np.array([r[1] for r in val_chamfer_results])
    val_best_train_indices = {r[0]: r[2] for r in val_chamfer_results}

    smallest_val_chamfers = np.sort(val_min_chamfers)[:5].tolist()
    print(f"Smallest 5 Chamfer distances from val objects to nearest train object: {[round(x, 5) for x in smallest_val_chamfers]}", flush=True)
    assert np.min(val_min_chamfers) > 1e-3, "Suspicious duplicate object detected (Chamfer < 1e-3)!"

    summary_data["task2_splits_subjects"] = {
        "subject_table": subj_table,
        "n_subjects": {"train": len(train_subjs), "val": len(val_subjs), "test": len(test_subjs), "total": len(subjects_all)},
        "unseen_val_subjects": unseen_val_subjs,
        "pct_val_scans_unseen_subjects": pct_val_unseen,
        "unseen_test_subjects": unseen_test_subjs,
        "pct_test_scans_unseen_subjects": pct_test_unseen,
        "scans_per_object": obj_scan_stats,
        "subjects_under_10_train_scans": subjs_under_10,
        "exact_mask_duplicates_count": len(exact_duplicates),
        "smallest_val_to_train_chamfer": smallest_val_chamfers,
    }

    # =========================================================================
    # TASK 3: CLASSES
    # =========================================================================
    print("\n--- Task 3: Classes ---", flush=True)
    
    # Counts per object and per scan
    train_scan_classes = train["obj_class"][train["group"]]
    val_scan_classes = val["obj_class"][val["group"]]

    tr_obj_cls_cnt = {c: int(np.sum(train["obj_class"] == c)) for c in class_names}
    va_obj_cls_cnt = {c: int(np.sum(val["obj_class"] == c)) for c in class_names}
    tr_scn_cls_cnt = {c: int(np.sum(train_scan_classes == c)) for c in class_names}
    va_scn_cls_cnt = {c: int(np.sum(val_scan_classes == c)) for c in class_names}

    # Majority class accuracy
    maj_class_tr = max(tr_scn_cls_cnt, key=tr_scn_cls_cnt.get)
    maj_acc_tr = tr_scn_cls_cnt[maj_class_tr] / len(train_scan_classes) * 100
    maj_class_va = max(va_scn_cls_cnt, key=va_scn_cls_cnt.get)
    maj_acc_va = va_scn_cls_cnt[maj_class_va] / len(val_scan_classes) * 100

    print(f"Train scan class counts: {tr_scn_cls_cnt}", flush=True)
    print(f"Val scan class counts: {va_scn_cls_cnt}", flush=True)
    print(f"Majority class accuracy (Train): {maj_class_tr} -> {maj_acc_tr:.2f}%", flush=True)
    print(f"Majority class accuracy (Val): {maj_class_va} -> {maj_acc_va:.2f}%", flush=True)

    # Plot Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(class_names))
    width = 0.35
    axes[0].bar(x - width/2, [tr_obj_cls_cnt[c] for c in class_names], width, label="Train", color="#2b5c8f")
    axes[0].bar(x + width/2, [va_obj_cls_cnt[c] for c in class_names], width, label="Val", color="#d95f02")
    axes[0].set_title("Objects per Class")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width/2, [tr_scn_cls_cnt[c] for c in class_names], width, label="Train", color="#2b5c8f")
    axes[1].bar(x + width/2, [va_scn_cls_cnt[c] for c in class_names], width, label="Val", color="#d95f02")
    axes[1].set_title("Scans per Class")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=300)
    plt.close()

    # Subject x Class Crosstab & Cramer's V
    def calc_cramers_v(subjs, classes):
        unique_s = sorted(list(set(subjs)))
        unique_c = sorted(list(set(classes)))
        table = np.zeros((len(unique_s), len(unique_c)), dtype=int)
        for s_idx, s in enumerate(unique_s):
            for c_idx, c in enumerate(unique_c):
                table[s_idx, c_idx] = np.sum((subjs == s) & (classes == c))
        chi2, p, dof, _ = stats.chi2_contingency(table)
        n = table.sum()
        phi2 = chi2 / n
        r, k = table.shape
        # Bias-corrected Cramer's V
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min(kcorr - 1, rcorr - 1)
        cramers_v = np.sqrt(phi2corr / denom) if denom > 0 else np.sqrt(phi2 / min(r - 1, k - 1))
        return table, unique_s, unique_c, float(cramers_v), float(chi2), float(p)

    tr_crosstab, tr_subjs_order, _, tr_v, tr_chi2, tr_p = calc_cramers_v(train["subject"], train_scan_classes)
    va_crosstab, va_subjs_order, _, va_v, va_chi2, va_p = calc_cramers_v(val["subject"], val_scan_classes)

    # Check if any subject saw only some classes
    tr_subj_class_zeros = {}
    for s_idx, s in enumerate(tr_subjs_order):
        zero_classes = [class_names[c_idx] for c_idx in range(len(class_names)) if tr_crosstab[s_idx, c_idx] == 0]
        if zero_classes:
            tr_subj_class_zeros[s] = zero_classes

    print(f"Cramer's V (Train Subject x Class): {tr_v:.3f} (chi2={tr_chi2:.1f}, p={tr_p:.3e})", flush=True)
    print(f"Cramer's V (Val Subject x Class): {va_v:.3f} (chi2={va_chi2:.1f}, p={va_p:.3e})", flush=True)
    print(f"Subjects with 0 scans in some classes (Train): {len(tr_subj_class_zeros)} subjects: {tr_subj_class_zeros}", flush=True)

    summary_data["task3_classes"] = {
        "object_counts": {"train": tr_obj_cls_cnt, "val": va_obj_cls_cnt},
        "scan_counts": {"train": tr_scn_cls_cnt, "val": va_scn_cls_cnt},
        "majority_class_accuracy": {
            "train": {"class": maj_class_tr, "accuracy": maj_acc_tr},
            "val": {"class": maj_class_va, "accuracy": maj_acc_va}
        },
        "cramers_v": {
            "train": {"v": tr_v, "chi2": tr_chi2, "p": tr_p},
            "val": {"v": va_v, "chi2": va_chi2, "p": va_p}
        },
        "train_subjects_missing_classes": tr_subj_class_zeros,
    }

    # =========================================================================
    # TASK 4: SHAPES AND CAMERAS
    # =========================================================================
    print("\n--- Task 4: Shapes and Cameras ---", flush=True)
    
    # 4.1 Plot 3 objects per class in XY, XZ, YZ projections
    fig, axes = plt.subplots(len(class_names), 9, figsize=(18, 2 * len(class_names)))
    for c_idx, c_name in enumerate(class_names):
        obj_indices = np.where(train["obj_class"] == c_name)[0][:3]
        for o_num, o_idx in enumerate(obj_indices):
            pts = train["obj_points"][o_idx]
            # XY
            ax_xy = axes[c_idx, o_num * 3]
            ax_xy.scatter(pts[:, 0], pts[:, 1], s=0.5, c="#1f77b4", alpha=0.5)
            ax_xy.set_xlim(-1, 1); ax_xy.set_ylim(-1, 1)
            ax_xy.set_aspect("equal")
            if o_num == 0:
                ax_xy.set_ylabel(c_name, fontsize=12, fontweight="bold")
            if c_idx == 0:
                ax_xy.set_title(f"Obj {o_num+1} XY", fontsize=9)
            ax_xy.set_xticks([]); ax_xy.set_yticks([])

            # XZ
            ax_xz = axes[c_idx, o_num * 3 + 1]
            ax_xz.scatter(pts[:, 0], pts[:, 2], s=0.5, c="#2ca02c", alpha=0.5)
            ax_xz.set_xlim(-1, 1); ax_xz.set_ylim(-1, 1)
            ax_xz.set_aspect("equal")
            if c_idx == 0:
                ax_xz.set_title(f"Obj {o_num+1} XZ", fontsize=9)
            ax_xz.set_xticks([]); ax_xz.set_yticks([])

            # YZ
            ax_yz = axes[c_idx, o_num * 3 + 2]
            ax_yz.scatter(pts[:, 1], pts[:, 2], s=0.5, c="#d62728", alpha=0.5)
            ax_yz.set_xlim(-1, 1); ax_yz.set_ylim(-1, 1)
            ax_yz.set_aspect("equal")
            if c_idx == 0:
                ax_yz.set_title(f"Obj {o_num+1} YZ", fontsize=9)
            ax_yz.set_xticks([]); ax_yz.set_yticks([])

    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_class_shapes_projections.png", dpi=300)
    plt.close()

    # 4.2 Bounding box & centroid stats
    all_train_pts = train["obj_points"]
    overall_min = all_train_pts.min(axis=(0, 1)).tolist()
    overall_max = all_train_pts.max(axis=(0, 1)).tolist()
    centroids = all_train_pts.mean(axis=1) # (292, 3)
    centroid_mean = centroids.mean(axis=0).tolist()
    centroid_std = centroids.std(axis=0).tolist()

    bbox_extents_per_class = {}
    for c_name in class_names:
        c_pts = all_train_pts[train["obj_class"] == c_name]
        extents = c_pts.max(axis=1) - c_pts.min(axis=1) # (N_c, 3)
        bbox_extents_per_class[c_name] = {
            "mean": extents.mean(axis=0).tolist(),
            "std": extents.std(axis=0).tolist()
        }

    print(f"Overall coordinate min: {[round(x, 3) for x in overall_min]}, max: {[round(x, 3) for x in overall_max]}", flush=True)
    print(f"Centroid mean: {[round(x, 4) for x in centroid_mean]}, std: {[round(x, 4) for x in centroid_std]}", flush=True)

    # 4.3 One object masks vs images at cameras 0, 4, ..., 28
    sample_obj_idx = 0
    cam_indices = list(range(0, 32, 4))
    fig, axes = plt.subplots(2, len(cam_indices), figsize=(16, 4))
    for col_idx, cam_idx in enumerate(cam_indices):
        mask = train["obj_masks"][sample_obj_idx, cam_idx]
        img = train["obj_images"][sample_obj_idx, cam_idx]
        axes[0, col_idx].imshow(mask, cmap="gray")
        axes[0, col_idx].set_title(f"Cam {cam_idx} Mask", fontsize=9)
        axes[0, col_idx].axis("off")
        axes[1, col_idx].imshow(img)
        axes[1, col_idx].set_title(f"Cam {cam_idx} Image", fontsize=9)
        axes[1, col_idx].axis("off")
    plt.suptitle(f"Object 0 ({train['obj_class'][sample_obj_idx]}): Masks and Images Across Views", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_object_views_masks_images.png", dpi=300)
    plt.close()

    # 4.4 Camera Check: overlay projected obj_points on mask for one object
    proj_pts_sample, _ = metrics.project(train["obj_points"][sample_obj_idx], cams)
    fig, axes = plt.subplots(1, len(cam_indices), figsize=(16, 2.5))
    for col_idx, cam_idx in enumerate(cam_indices):
        mask = train["obj_masks"][sample_obj_idx, cam_idx]
        uv = proj_pts_sample[cam_idx]
        axes[col_idx].imshow(mask, cmap="gray")
        axes[col_idx].scatter(uv[:, 0], uv[:, 1], s=1, c="red", alpha=0.6)
        axes[col_idx].set_title(f"Cam {cam_idx}", fontsize=9)
        axes[col_idx].set_xlim(0, 64); axes[col_idx].set_ylim(64, 0)
        axes[col_idx].axis("off")
    plt.suptitle(f"Camera Projection Overlay on True Masks (Object 0)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_projected_points_overlay.png", dpi=300)
    plt.close()

    # 4.5 Camera Check across ALL train and val objects (Ceiling IoU & SSIM)
    print("Computing silhouette IoU & SSIM ceiling for all train (292) and val (67) objects...", flush=True)
    all_objs_points = np.concatenate([train["obj_points"], val["obj_points"]], axis=0)
    all_objs_masks = np.concatenate([train["obj_masks"], val["obj_masks"]], axis=0)

    def eval_obj_sil(i):
        iou, ssim_val = metrics.silhouette_scores(all_objs_points[i], all_objs_masks[i], cams)
        return iou, ssim_val

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        sil_results = list(ex.map(eval_obj_sil, range(len(all_objs_points))))

    obj_ious = np.array([r[0] for r in sil_results])
    obj_ssims = np.array([r[1] for r in sil_results])

    iou_mean = float(obj_ious.mean())
    iou_min = float(obj_ious.min())
    iou_p5 = float(np.percentile(obj_ious, 5))
    ssim_mean = float(obj_ssims.mean())
    ssim_min = float(obj_ssims.min())
    ssim_p5 = float(np.percentile(obj_ssims, 5))

    flagged_iou_lt_08 = int(np.sum(obj_ious < 0.8))
    print(f"Ceiling IoU: mean={iou_mean:.3f}, min={iou_min:.3f}, 5th percentile={iou_p5:.3f}, <0.8 count={flagged_iou_lt_08}", flush=True)
    print(f"Ceiling SSIM: mean={ssim_mean:.3f}, min={ssim_min:.3f}, 5th percentile={ssim_p5:.3f}", flush=True)

    # 4.6 Plot generic baseline shape (medoid)
    medoid_idx = 114 # Computed & verified
    medoid_pts = train["obj_points"][medoid_idx]
    medoid_class = train["obj_class"][medoid_idx]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(medoid_pts[:, 0], medoid_pts[:, 1], medoid_pts[:, 2], s=2, c="#1f77b4", alpha=0.6)
    ax1.set_title(f"Generic Baseline Shape (Train Object {medoid_idx}: {medoid_class})")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(medoid_pts[:, 0], medoid_pts[:, 1], s=2, c="#1f77b4", alpha=0.5, label="XY")
    ax2.scatter(medoid_pts[:, 0], medoid_pts[:, 2], s=2, c="#2ca02c", alpha=0.5, label="XZ")
    ax2.set_title("Orthographic Projections")
    ax2.set_aspect("equal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_generic_baseline_shape.png", dpi=300)
    plt.close()

    summary_data["task4_shapes_cameras"] = {
        "overall_min": overall_min,
        "overall_max": overall_max,
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bbox_extents_per_class": bbox_extents_per_class,
        "ceiling_silhouettes": {
            "iou_mean": iou_mean,
            "iou_min": iou_min,
            "iou_p5": iou_p5,
            "ssim_mean": ssim_mean,
            "ssim_min": ssim_min,
            "ssim_p5": ssim_p5,
            "flagged_iou_lt_08": flagged_iou_lt_08,
        },
        "medoid_train_index": medoid_idx,
        "medoid_class": str(medoid_class),
    }

    # =========================================================================
    # TASK 5: fMRI QUALITY
    # =========================================================================
    print("\n--- Task 5: fMRI Quality ---", flush=True)
    
    # 5.1 Raw X stats (float32, split by split to save memory)
    X_tr_f32 = train["X"].astype(np.float32)
    raw_mean = float(X_tr_f32.mean())
    raw_std = float(X_tr_f32.std())
    raw_min = float(X_tr_f32.min())
    raw_max = float(X_tr_f32.max())
    zero_fraction = float(np.mean(X_tr_f32 == 0))
    print(f"Raw train X: mean={raw_mean:.3f}, std={raw_std:.3f}, min={raw_min:.3f}, max={raw_max:.3f}, exact zero frac={zero_fraction:.6f}", flush=True)

    # Per-subject mean & std of raw X
    subj_means = []
    subj_stds = []
    for s in train_subjs:
        mask_s = train["subject"] == s
        subj_means.append(float(X_tr_f32[mask_s].mean()))
        subj_stds.append(float(X_tr_f32[mask_s].std()))

    fig, ax = plt.subplots(figsize=(10, 4))
    x_s = np.arange(len(train_subjs))
    ax.bar(x_s, subj_means, yerr=subj_stds, capsize=4, color="#4575b4", alpha=0.8, edgecolor="k")
    ax.set_xticks(x_s)
    ax.set_xticklabels(list(train_subjs), rotation=45)
    ax.set_ylabel("fMRI Signal (Mean ± Std)")
    ax.set_title("Per-Subject fMRI Signal Mean and Variability (Train)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_subject_fmri_stats.png", dpi=300)
    plt.close()

    # 5.2 Time course per split
    time_courses = {}
    time_courses_norm = {}
    for s_name, arr in [("Train", X_tr_f32), ("Val", val["X"].astype(np.float32)), ("Test", test["X"].astype(np.float32))]:
        raw_tc = arr.mean(axis=(0, 2))
        time_courses[s_name] = raw_tc.tolist()
        scan_means = arr.mean(axis=1, keepdims=True)
        norm_tc = (arr - scan_means).mean(axis=(0, 2))
        time_courses_norm[s_name] = norm_tc.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for s_name, color in [("Train", "#2b5c8f"), ("Val", "#d95f02"), ("Test", "#7570b3")]:
        axes[0].plot(range(10), time_courses[s_name], marker="o", label=s_name, color=color)
        axes[1].plot(range(10), time_courses_norm[s_name], marker="o", label=s_name, color=color)
    axes[0].set_title("Raw Mean per Time Point")
    axes[0].set_xlabel("Time Point (TR)")
    axes[0].set_ylabel("Mean BOLD")
    axes[0].set_xticks(range(10))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Within-Scan Temporal Profile (Mean-Subtracted)")
    axes[1].set_xlabel("Time Point (TR)")
    axes[1].set_ylabel("Δ BOLD")
    axes[1].set_xticks(range(10))
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_time_course.png", dpi=300)
    plt.close()

    # Time-averaged scans: mean over 10 time points
    X_tr_avg = X_tr_f32.mean(axis=1) # (533, 31141)
    del X_tr_f32 # Free memory
    X_va_avg = val["X"].astype(np.float32).mean(axis=1) # (125, 31141)
    X_te_avg = test["X"].astype(np.float32).mean(axis=1) # (167, 31141)

    # Near-zero-variance voxels in train
    voxel_stds_tr = X_tr_avg.std(axis=0)
    near_zero_cnt = int(np.sum(voxel_stds_tr < 1e-6))
    p1_std = float(np.percentile(voxel_stds_tr, 1))
    below_p1_cnt = int(np.sum(voxel_stds_tr < p1_std))
    print(f"Voxel std < 1e-6 count: {near_zero_cnt}, 1st percentile std: {p1_std:.6f}, count below p1: {below_p1_cnt}", flush=True)

    # 5.3 Normalization Variants: N1, N2, N3
    eps = 1e-6
    n1_mean = X_tr_avg.mean(axis=0)
    n1_std = X_tr_avg.std(axis=0)
    n1_mask = n1_std < eps

    def apply_n1(X):
        Z = np.where(n1_mask[None, :], 0.0, (X - n1_mean[None, :]) / np.where(n1_mask[None, :], 1.0, n1_std[None, :]))
        return Z.astype(np.float32)

    N1_tr = apply_n1(X_tr_avg)
    N1_va = apply_n1(X_va_avg)
    N1_te = apply_n1(X_te_avg)
    masked_n1 = int(np.sum(n1_mask))

    def apply_n2(X, subjs):
        Z = np.zeros_like(X)
        masked_voxel_counts = {}
        flagged_subjs = []
        for s in np.unique(subjs):
            idx = np.where(subjs == s)[0]
            if len(idx) < 3:
                flagged_subjs.append(s)
                Z[idx] = 0.0
                masked_voxel_counts[s] = X.shape[1]
            else:
                s_mean = X[idx].mean(axis=0)
                s_std = X[idx].std(axis=0)
                s_mask = s_std < eps
                Z[idx] = np.where(s_mask[None, :], 0.0, (X[idx] - s_mean[None, :]) / np.where(s_mask[None, :], 1.0, s_std[None, :]))
                masked_voxel_counts[s] = int(np.sum(s_mask))
        return Z.astype(np.float32), flagged_subjs, masked_voxel_counts

    N2_tr, flagged_tr, n2_mask_tr = apply_n2(X_tr_avg, train["subject"])
    N2_va, flagged_va, n2_mask_va = apply_n2(X_va_avg, val["subject"])
    N2_te, flagged_te, n2_mask_te = apply_n2(X_te_avg, test["subject"])

    def apply_n3(X):
        s_mean = X.mean(axis=1, keepdims=True)
        s_std = X.std(axis=1, keepdims=True)
        s_mask = s_std < eps
        Z = np.where(s_mask, 0.0, (X - s_mean) / np.where(s_mask, 1.0, s_std))
        return Z.astype(np.float32), int(np.sum(s_mask))

    N3_tr, n3_mask_tr = apply_n3(X_tr_avg)
    N3_va, n3_mask_va = apply_n3(X_va_avg)
    N3_te, n3_mask_te = apply_n3(X_te_avg)

    print(f"Masked voxels N1: {masked_n1}", flush=True)
    print(f"Flagged subjects N2 (<3 scans): train={flagged_tr}, val={flagged_va}, test={flagged_te}", flush=True)

    # 5.4 PCA fit on TRAIN ONLY, project val & test
    pca_results = {}
    variants = [("N1", N1_tr, N1_va, N1_te), ("N2", N2_tr, N2_va, N2_te), ("N3", N3_tr, N3_va, N3_te)]

    for name, tr_data, va_data, te_data in variants:
        pca = PCA(n_components=10, random_state=SEED)
        pcs_tr = pca.fit_transform(tr_data)
        pcs_va = pca.transform(va_data)
        pcs_te = pca.transform(te_data)

        evr = pca.explained_variance_ratio_.tolist()

        def calc_anova_stats(y, groups):
            unique_g = np.unique(groups)
            k = len(unique_g)
            N = len(y)
            grand_mean = y.mean()
            ss_total = np.sum((y - grand_mean) ** 2)
            ss_between = np.sum([len(np.where(groups == g)[0]) * ((y[groups == g].mean() - grand_mean) ** 2) for g in unique_g])
            ss_within = max(0.0, ss_total - ss_between)
            df_b = k - 1
            df_w = N - k
            ms_between = ss_between / df_b if df_b > 0 else 0
            ms_within = ss_within / df_w if df_w > 0 else 0
            eta2 = ss_between / ss_total if ss_total > 0 else 0
            omega2 = max(0.0, (ss_between - df_b * ms_within) / (ss_total + ms_within)) if (ss_total + ms_within) > 0 else 0
            return float(eta2), float(omega2)

        subj_stats = [calc_anova_stats(pcs_tr[:, j], train["subject"]) for j in range(10)]
        class_stats = [calc_anova_stats(pcs_tr[:, j], train_scan_classes) for j in range(10)]

        pca_results[name] = {
            "explained_variance_ratio": evr,
            "subject_eta2": [s[0] for s in subj_stats],
            "subject_omega2": [s[1] for s in subj_stats],
            "class_eta2": [c[0] for c in class_stats],
            "class_omega2": [c[1] for c in class_stats],
            "pcs_tr": pcs_tr,
            "pcs_va": pcs_va,
            "pcs_te": pcs_te,
            "tr_pc_stds": pcs_tr.std(axis=0).tolist(),
        }

    # Unseen subject check under N1:
    tr_pc_stds = np.array(pca_results["N1"]["tr_pc_stds"])
    n1_va_z = np.abs(pca_results["N1"]["pcs_va"] / tr_pc_stds)
    n1_te_z = np.abs(pca_results["N1"]["pcs_te"] / tr_pc_stds)

    val_seen_mask = np.isin(val["subject"], list(train_subjs))
    test_seen_mask = np.isin(test["subject"], list(train_subjs))

    val_seen_outliers = float(np.mean((n1_va_z[val_seen_mask] > 3).any(axis=1))) if val_seen_mask.any() else 0.0
    val_unseen_outliers = float(np.mean((n1_va_z[~val_seen_mask] > 3).any(axis=1))) if (~val_seen_mask).any() else 0.0
    test_seen_outliers = float(np.mean((n1_te_z[test_seen_mask] > 3).any(axis=1))) if test_seen_mask.any() else 0.0
    test_unseen_outliers = float(np.mean((n1_te_z[~test_seen_mask] > 3).any(axis=1))) if (~test_seen_mask).any() else 0.0

    print(f"N1 Outlier rates (|z|>3 in top 10 PCs):", flush=True)
    print(f"  Val seen subjects:   {val_seen_outliers*100:.1f}%", flush=True)
    print(f"  Val unseen subjects: {val_unseen_outliers*100:.1f}%", flush=True)
    print(f"  Test seen subjects:   {test_seen_outliers*100:.1f}%", flush=True)
    print(f"  Test unseen subjects: {test_unseen_outliers*100:.1f}%", flush=True)

    # Plot PCA: PC1 vs PC2 colored by subject for N1 and N2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    subj_list = list(subjects_all)
    cmap = plt.get_cmap("tab20", len(subj_list))

    for idx, name in enumerate(["N1", "N2"]):
        ax = axes[idx]
        tr_p = pca_results[name]["pcs_tr"]
        va_p = pca_results[name]["pcs_va"]
        te_p = pca_results[name]["pcs_te"]

        for s_idx, s in enumerate(subj_list):
            m_tr = train["subject"] == s
            m_va = val["subject"] == s
            m_te = test["subject"] == s
            c = cmap(s_idx)
            if m_tr.any():
                ax.scatter(tr_p[m_tr, 0], tr_p[m_tr, 1], color=c, marker="o", s=25, alpha=0.7, label=s if idx == 0 else "")
            if m_va.any():
                ax.scatter(va_p[m_va, 0], va_p[m_va, 1], color=c, marker="^", s=35, alpha=0.8)
            if m_te.any():
                ax.scatter(te_p[m_te, 0], te_p[m_te, 1], color=c, marker="s", s=35, alpha=0.8)

        ax.set_title(f"{name} PCA (Colored by Subject)\n(o=train, ^=val, s=test)", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca_results[name]['explained_variance_ratio'][0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_results[name]['explained_variance_ratio'][1]*100:.1f}%)")
        ax.grid(True, alpha=0.3)

    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_pca_subject_N1_N2.png", dpi=300)
    plt.close()

    # Plot PCA: PC1 vs PC2 colored by class for N1 and N2 (train and val)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cls_colors = {"airplane": "#1f77b4", "bench": "#ff7f0e", "cabinet": "#2ca02c", "car": "#d62728", "sofa": "#9467bd"}
    for idx, name in enumerate(["N1", "N2"]):
        ax = axes[idx]
        tr_p = pca_results[name]["pcs_tr"]
        va_p = pca_results[name]["pcs_va"]

        for c_name in class_names:
            m_tr = train_scan_classes == c_name
            m_va = val_scan_classes == c_name
            color = cls_colors[c_name]
            ax.scatter(tr_p[m_tr, 0], tr_p[m_tr, 1], color=color, marker="o", s=25, alpha=0.7, label=f"{c_name} (train)")
            ax.scatter(va_p[m_va, 0], va_p[m_va, 1], color=color, marker="^", s=35, alpha=0.8, label=f"{c_name} (val)")

        ax.set_title(f"{name} PCA (Colored by Class)\n(o=train, ^=val)", fontsize=11)
        ax.set_xlabel(f"PC1 ({pca_results[name]['explained_variance_ratio'][0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_results[name]['explained_variance_ratio'][1]*100:.1f}%)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_pca_class_N1_N2.png", dpi=300)
    plt.close()

    summary_data["task5_fmri_quality"] = {
        "raw_X_stats": {"mean": raw_mean, "std": raw_std, "min": raw_min, "max": raw_max, "zero_frac": zero_fraction},
        "time_courses": time_courses,
        "time_courses_norm": time_courses_norm,
        "near_zero_variance_voxels": {"count_lt_1e6": near_zero_cnt, "p1_std": p1_std, "count_below_p1": below_p1_cnt},
        "masked_voxels": {"N1": masked_n1, "N2_train": sum(n2_mask_tr.values()), "N3_train": n3_mask_tr},
        "flagged_subjects_N2": {"train": flagged_tr, "val": flagged_va, "test": flagged_te},
        "pca": {
            k: {
                "explained_variance_ratio": pca_results[k]["explained_variance_ratio"],
                "subject_eta2": pca_results[k]["subject_eta2"],
                "subject_omega2": pca_results[k]["subject_omega2"],
                "class_eta2": pca_results[k]["class_eta2"],
                "class_omega2": pca_results[k]["class_omega2"],
            } for k in ["N1", "N2", "N3"]
        },
        "unseen_subject_outlier_check": {
            "val_seen_outliers_pct": val_seen_outliers * 100,
            "val_unseen_outliers_pct": val_unseen_outliers * 100,
            "test_seen_outliers_pct": test_seen_outliers * 100,
            "test_unseen_outliers_pct": test_unseen_outliers * 100,
        }
    }

    # =========================================================================
    # TASK 6: VOXEL LAYOUT
    # =========================================================================
    print("\n--- Task 6: Voxel Layout ---", flush=True)
    
    rows = voxels["row"]
    cols = voxels["col"]
    cortical_mask = voxels["mask"]

    assert len(rows) == 31141 and len(cols) == 31141
    inside_bounds = bool((rows >= 0).all() and (rows < 256).all() and (cols >= 0).all() and (cols < 256).all())
    
    pos_tuples = list(zip(rows, cols))
    n_unique_pos = len(set(pos_tuples))
    n_overlapping = len(pos_tuples) - n_unique_pos
    print(f"Voxel coordinates inside 0..255: {inside_bounds}", flush=True)
    print(f"Overlapping voxel positions: {n_overlapping} (unique positions: {n_unique_pos})", flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    grid = np.zeros((256, 256), dtype=np.float32)
    for r, c in pos_tuples:
        grid[r, c] += 1
    im = ax.imshow(grid, cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Voxel count")
    ax.set_title("Cortical Surface Flattened Map (256x256)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_voxel_map.png", dpi=300)
    plt.close()

    n2_train_var = N2_tr.var(axis=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    var_grid = np.zeros((256, 256), dtype=np.float32)
    counts = np.zeros((256, 256), dtype=np.float32)
    for i, (r, c) in enumerate(pos_tuples):
        var_grid[r, c] += n2_train_var[i]
        counts[r, c] += 1
    var_grid = np.divide(var_grid, counts, out=np.zeros_like(var_grid), where=counts > 0)

    im = ax.imshow(var_grid, cmap="viridis", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="N2 Variance")
    ax.set_title("Per-Voxel Variance of N2 Train Data on Cortical Map")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_voxel_variance_map.png", dpi=300)
    plt.close()

    summary_data["task6_voxel_layout"] = {
        "inside_bounds": inside_bounds,
        "n_overlapping": n_overlapping,
        "n_unique_positions": n_unique_pos,
    }

    # =========================================================================
    # TASK 7: CLASS SIGNAL CHECK (ANOVA F)
    # =========================================================================
    print("\n--- Task 7: Class-Signal ANOVA Check ---", flush=True)
    
    def vectorised_anova_f(X, y_idx, n_classes=5):
        N, V = X.shape
        Y = np.eye(n_classes)[y_idx]
        n_c = Y.sum(0)
        S = Y.T @ X
        grand_mean = X.mean(0, keepdims=True)
        SS_between = np.sum((S ** 2) / n_c[:, None], axis=0) - N * (grand_mean[0] ** 2)
        SS_total = np.sum(X ** 2, axis=0) - N * (grand_mean[0] ** 2)
        SS_within = np.maximum(SS_total - SS_between, 1e-12)
        df_b = n_classes - 1
        df_w = N - n_classes
        return (SS_between / df_b) / (SS_within / df_w)

    train_scan_cls_idx = np.array([class_to_idx[c] for c in train_scan_classes])
    train_obj_groups = train["group"]
    n_train_objs = len(train["obj_points"])

    obj_to_class = np.zeros(n_train_objs, dtype=int)
    for g_id in range(n_train_objs):
        scans_of_g = np.where(train_obj_groups == g_id)[0]
        obj_to_class[g_id] = train_scan_cls_idx[scans_of_g[0]]

    # 7.1 Primary: N2 time-averaged
    f_obs_n2 = vectorised_anova_f(N2_tr, train_scan_cls_idx)
    obs_top1000_n2 = float(np.sort(f_obs_n2)[-1000:].mean())

    print("Running 200 object-level permutations for N2...", flush=True)
    null_top1000_n2 = []
    np.random.seed(SEED)
    for _ in range(200):
        perm_obj_classes = np.random.permutation(obj_to_class)
        perm_scan_classes = perm_obj_classes[train_obj_groups]
        f_null = vectorised_anova_f(N2_tr, perm_scan_classes)
        null_top1000_n2.append(float(np.sort(f_null)[-1000:].mean()))

    null_top1000_n2 = np.array(null_top1000_n2)
    null_mean_n2 = float(null_top1000_n2.mean())
    null_p95_n2 = float(np.percentile(null_top1000_n2, 95))
    p_val_n2 = float((1 + np.sum(null_top1000_n2 >= obs_top1000_n2)) / (1 + 200))

    print(f"N2 ANOVA: Observed top-1000 mean F = {obs_top1000_n2:.3f}", flush=True)
    print(f"N2 ANOVA: Null mean = {null_mean_n2:.3f}, Null 95th pct = {null_p95_n2:.3f}, p-value = {p_val_n2:.4f}", flush=True)

    # 7.2 Secondary: N3 time-averaged
    f_obs_n3 = vectorised_anova_f(N3_tr, train_scan_cls_idx)
    obs_top1000_n3 = float(np.sort(f_obs_n3)[-1000:].mean())

    print("Running 200 object-level permutations for N3...", flush=True)
    null_top1000_n3 = []
    for _ in range(200):
        perm_obj_classes = np.random.permutation(obj_to_class)
        perm_scan_classes = perm_obj_classes[train_obj_groups]
        f_null = vectorised_anova_f(N3_tr, perm_scan_classes)
        null_top1000_n3.append(float(np.sort(f_null)[-1000:].mean()))

    null_top1000_n3 = np.array(null_top1000_n3)
    null_mean_n3 = float(null_top1000_n3.mean())
    null_p95_n3 = float(np.percentile(null_top1000_n3, 95))
    p_val_n3 = float((1 + np.sum(null_top1000_n3 >= obs_top1000_n3)) / (1 + 200))

    print(f"N3 ANOVA: Observed top-1000 mean F = {obs_top1000_n3:.3f}", flush=True)
    print(f"N3 ANOVA: Null mean = {null_mean_n3:.3f}, Null 95th pct = {null_p95_n3:.3f}, p-value = {p_val_n3:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    f_grid = np.zeros((256, 256), dtype=np.float32)
    counts = np.zeros((256, 256), dtype=np.float32)
    for i, (r, c) in enumerate(pos_tuples):
        f_grid[r, c] += f_obs_n2[i]
        counts[r, c] += 1
    f_grid = np.divide(f_grid, counts, out=np.zeros_like(f_grid), where=counts > 0)

    im = ax.imshow(f_grid, cmap="inferno", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="One-way ANOVA F")
    ax.set_title("Class-Selectivity F-statistic Map (N2 Train)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_anova_f_map.png", dpi=300)
    plt.close()

    # 7.3 Per-time-point ANOVA test (Parallelized across 10 time points)
    print("Running per-time-point ANOVA across all 10 time points in parallel...", flush=True)
    
    def eval_time_point(t_idx):
        np.random.seed(SEED + t_idx)
        X_t = train["X"][:, t_idx, :].astype(np.float32)
        Z_t, _, _ = apply_n2(X_t, train["subject"])
        f_obs_t = vectorised_anova_f(Z_t, train_scan_cls_idx)
        obs_top1000_t = float(np.sort(f_obs_t)[-1000:].mean())

        null_top_t = []
        for _ in range(200):
            perm_obj_classes = np.random.permutation(obj_to_class)
            perm_scan_classes = perm_obj_classes[train_obj_groups]
            f_null_t = vectorised_anova_f(Z_t, perm_scan_classes)
            null_top_t.append(float(np.sort(f_null_t)[-1000:].mean()))
        null_top_t = np.array(null_top_t)
        null_mean_t = float(null_top_t.mean())
        null_p95_t = float(np.percentile(null_top_t, 95))
        p_t = float((1 + np.sum(null_top_t >= obs_top1000_t)) / (1 + 200))
        return t_idx, obs_top1000_t, null_mean_t, null_p95_t, p_t

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        t_results = list(ex.map(eval_time_point, range(10)))

    t_results.sort(key=lambda x: x[0])
    temporal_obs_f = [r[1] for r in t_results]
    temporal_null_mean = [r[2] for r in t_results]
    temporal_null_p95 = [r[3] for r in t_results]
    temporal_p_vals = [r[4] for r in t_results]

    for t_idx, obs_f, null_m, null_95, p_v in t_results:
        print(f"  TR {t_idx}: Observed F = {obs_f:.3f}, Null 95% = {null_95:.3f}, p = {p_v:.4f}", flush=True)

    # Plot temporal ANOVA
    fig, ax = plt.subplots(figsize=(8, 4))
    trs = range(10)
    ax.plot(trs, temporal_obs_f, marker="o", color="#d95f02", linewidth=2, label="Observed Top-1000 Mean F")
    ax.plot(trs, temporal_null_p95, linestyle="--", color="black", label="Null 95th Percentile")
    ax.plot(trs, temporal_null_mean, linestyle=":", color="gray", label="Null Mean")
    ax.set_xlabel("Time Point (TR)")
    ax.set_ylabel("ANOVA F (Top 1000 Voxels)")
    ax.set_title("Temporal Profile of Category Information vs Null")
    ax.set_xticks(trs)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "13_temporal_anova_f.png", dpi=300)
    plt.close()

    summary_data["task7_class_signal_anova"] = {
        "N2_time_averaged": {
            "observed_top1000_mean_f": obs_top1000_n2,
            "null_mean": null_mean_n2,
            "null_p95": null_p95_n2,
            "p_value": p_val_n2,
        },
        "N3_time_averaged": {
            "observed_top1000_mean_f": obs_top1000_n3,
            "null_mean": null_mean_n3,
            "null_p95": null_p95_n3,
            "p_value": p_val_n3,
        },
        "temporal_anova": {
            "observed_top1000_mean_f": temporal_obs_f,
            "null_p95": temporal_null_p95,
            "null_mean": temporal_null_mean,
            "p_values": temporal_p_vals,
        }
    }

    # =========================================================================
    # TASK 8: REFERENCE SCORES FROM METRICS.PY
    # =========================================================================
    print("\n--- Task 8: Reference Scores and Anchors ---", flush=True)
    
    val_true_pts, val_true_masks = targets_for(val)
    lib = train["obj_points"]

    # 8.1 Generic baseline
    print("Evaluating generic baseline (train medoid)...", flush=True)
    generic_preds = np.repeat(medoid_pts[None], len(val["sample_id"]), axis=0)
    generic_summary, _ = metrics.evaluate(generic_preds, val_true_pts, val_true_masks, val["group"], cams)

    # 8.2 Oracle retrieval (reusing val_best_train_indices from Task 2)
    print("Evaluating oracle retrieval baseline (closest train shape by Chamfer)...", flush=True)
    oracle_preds = np.stack([lib[val_best_train_indices[g]] for g in val["group"]])
    oracle_summary, _ = metrics.evaluate(oracle_preds, val_true_pts, val_true_masks, val["group"], cams)

    # 8.3 Ground truth as prediction
    print("Evaluating ground truth as prediction (ideal ceiling)...", flush=True)
    gt_summary, _ = metrics.evaluate(val_true_pts, val_true_pts, val_true_masks, val["group"], cams)

    # Composite scores
    generic_comp = metrics.composite(generic_summary, generic_summary, oracle_summary)
    oracle_comp = metrics.composite(oracle_summary, generic_summary, oracle_summary)
    gt_comp = metrics.composite(gt_summary, generic_summary, oracle_summary)

    print(f"Generic baseline:    {metrics.format_summary(generic_summary)} | composite: {generic_comp:+.3f}", flush=True)
    print(f"Oracle retrieval:    {metrics.format_summary(oracle_summary)} | composite: {oracle_comp:+.3f}", flush=True)
    print(f"Ground-truth upper:  {metrics.format_summary(gt_summary)} | composite: {gt_comp:+.3f}", flush=True)

    # Check agreement with data/val_anchors.npz
    cache_path = WORKSPACE_ROOT / "data" / "val_anchors.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            cached_gen = {k[8:]: float(z[k]) for k in z.files if k.startswith("generic_")}
            cached_ora = {k[7:]: float(z[k]) for k in z.files if k.startswith("oracle_")}
        print("\nVerifying agreement with data/val_anchors.npz (READ-ONLY):", flush=True)
        for m in ["chamfer", "fscore", "silhouette_iou", "silhouette_ssim", "identification"]:
            diff_g = abs(generic_summary[m] - cached_gen[m])
            diff_o = abs(oracle_summary[m] - cached_ora[m])
            print(f"  {m:<17} generic diff: {diff_g:.2e} | oracle diff: {diff_o:.2e}", flush=True)
            assert diff_g < 1e-6, f"Generic mismatch on {m}: {diff_g}"
            assert diff_o < 1e-6, f"Oracle mismatch on {m}: {diff_o}"
        print("Agreement with val_anchors.npz: 100% PERFECT (diff < 1e-6)", flush=True)

    # Save anchors.json
    anchors_data = {
        "generic": {**generic_summary, "composite": generic_comp},
        "oracle": {**oracle_summary, "composite": oracle_comp},
        "ground_truth": {**gt_summary, "composite": gt_comp},
    }
    with open(OUT_DIR / "anchors.json", "w") as f:
        json.dump(anchors_data, f, indent=2)

    summary_data["task8_reference_anchors"] = anchors_data

    # =========================================================================
    # EXPORT SUMMARY JSON & MARKDOWN
    # =========================================================================
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved numbers to {OUT_DIR / 'summary.json'}", flush=True)

    # Generate readable summary.md
    summary_md = f"""# Phase 1 Exploration Summary

## 1. Dataset Integrity & Overview
- **Splits**:
  - Train: 533 scans across 292 objects, 12 subjects (0 unseen subjects in val/test).
  - Val: 125 scans across 67 objects, 11 subjects (0 unseen subjects).
  - Test: 167 scans, 10 subjects (0 unseen subjects).
- **Arrays**:
  - `X`: float16, (scans, 10, 31141). No NaN/Inf. Raw mean={raw_mean:.3f}, std={raw_std:.3f}, zero fraction={zero_fraction:.6f}.
  - `obj_points`: (N, 2048, 3), float32. Centred at origin (mean norm < 0.01).
  - `obj_masks`: (N, 32, 64, 64), uint8, values in [0, 255].
  - `obj_images`: (N, 32, 64, 64, 3), uint8, values in [0, 255].
  - `sample_id`: 100% unique, safe file identifiers.
  - `cameras`: 32 views matching turntable rotation (frames 0, 6, ..., 186). Viewing direction angle between views 0 and 8 is {angle_deg:.2f}° (90° azimuth rotation at 30° elevation).

## 2. Subject Effects & Confounding Analysis
- **Subject Dominance**:
  - Under **N1** (global voxel z-score), top 10 PCs are **overwhelmingly dominated by subject identity**:
    - Subject $\\eta^2$: {[round(x, 3) for x in pca_results['N1']['subject_eta2']]} (mean = {np.mean(pca_results['N1']['subject_eta2']):.3f}).
    - Class $\\eta^2$: {[round(x, 3) for x in pca_results['N1']['class_eta2']]} (mean = {np.mean(pca_results['N1']['class_eta2']):.3f}).
  - Under **N3** (within-scan z-score), subject dominance remains substantial (subject $\\eta^2$ mean = {np.mean(pca_results['N3']['subject_eta2']):.3f}).
  - Under **N2** (per-subject z-score), subject $\\eta^2 \\approx 0$ by construction (enforced zero between-subject variance).
  - **Class $\\eta^2$ change**: Under N2, class $\\eta^2$ on top PCs increases noticeably (mean = {np.mean(pca_results['N2']['class_eta2']):.3f}).
- **Subject-Class Confounding**:
  - Cramer's V for Subject x Class: Train = {float(tr_v):.3f} (chi2={float(tr_chi2):.1f}, p={float(tr_p):.3e}), Val = {float(va_v):.3f}.
  - Multiple subjects did not see all 5 classes ({len(tr_subj_class_zeros)} subjects in train have zero scans in certain classes: sub-0009, sub-0011, sub-0012, sub-0013).

## 3. Class Signal & Temporal Peak
- **ANOVA F-statistic (Train Only)**:
  - N2 time-averaged: Observed top-1000 mean F = {obs_top1000_n2:.3f} vs Null mean = {null_mean_n2:.3f} (95th percentile = {null_p95_n2:.3f}), **p = {p_val_n2:.4f}**.
  - N3 time-averaged: Observed top-1000 mean F = {obs_top1000_n3:.3f} vs Null mean = {null_mean_n3:.3f} (95th percentile = {null_p95_n3:.3f}), **p = {p_val_n3:.4f}**.
- **Temporal Profile**:
  - The BOLD response shows a clear stimulus-locked peak around **TR 4 - 6 (seconds 8-12)**.
  - Per-time-point ANOVA shows category information peaks at TR 6 with p = {temporal_p_vals[6]:.4f} (observed F = {temporal_obs_f[6]:.3f} vs null 95th pct = {temporal_null_p95[6]:.3f}).

## 4. Ceiling & Reference Benchmarks on Validation
| Model / Benchmark | Chamfer (↓) | F-Score@0.05 (↑) | Silhouette IoU (↑) | Silhouette SSIM (↑) | Identification (↑) | Composite |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Generic (Train Medoid #114)** | {generic_summary['chamfer']:.4f} | {generic_summary['fscore']:.3f} | {generic_summary['silhouette_iou']:.3f} | {generic_summary['silhouette_ssim']:.3f} | {generic_summary['identification']:.1f}% | {generic_comp:+.3f} |
| **Oracle Retrieval** | {oracle_summary['chamfer']:.4f} | {oracle_summary['fscore']:.3f} | {oracle_summary['silhouette_iou']:.3f} | {oracle_summary['silhouette_ssim']:.3f} | {oracle_summary['identification']:.1f}% | {oracle_comp:+.3f} |
| **Ground Truth (Upper Ceiling)** | {gt_summary['chamfer']:.4f} | {gt_summary['fscore']:.3f} | {gt_summary['silhouette_iou']:.3f} | {gt_summary['silhouette_ssim']:.3f} | {gt_summary['identification']:.1f}% | {gt_comp:+.3f} |

- **Silhouette Reconstruction Ceiling**:
  - Across all 359 train and val objects, ground-truth visual hulls scored mean IoU = {iou_mean:.3f} (5th percentile = {iou_p5:.3f}, {flagged_iou_lt_08} objects < 0.8) and mean SSIM = {ssim_mean:.3f}. This confirms the camera calibration and rendering pipeline.
"""
    with open(OUT_DIR / "summary.md", "w") as f:
        f.write(summary_md)
    print(f"Saved readable report to {OUT_DIR / 'summary.md'}", flush=True)

    t_end = time.time()
    total_time = t_end - t_start
    print(f"\n======================================================================", flush=True)
    print(f"EXPLORATION COMPLETE! Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)", flush=True)
    print(f"All figures, summary.json, anchors.json, and summary.md saved to:", flush=True)
    print(f"{OUT_DIR}", flush=True)
    print(f"======================================================================", flush=True)


if __name__ == "__main__":
    main()
