"""
Reusable, leakage-safe evaluation and data-handling toolkit for fMRI 3D Reconstruction.
Phase 2 / Phase 3 common utilities. Follows RULES.md strictly.
"""
from pathlib import Path
import sys

# Ensure repository root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import json
import hashlib
import datetime
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib
matplotlib.use("Agg")

import starter
import metrics


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all():
    """
    Loads train, val, test, cams, voxels.
    Test set is strictly restricted to (X, subject, sample_id) - no shapes or labels exist.
    """
    train = starter.load_split("train")
    val = starter.load_split("val")
    test_raw = starter.load_split("test")
    test = {
        "X": test_raw["X"],
        "subject": test_raw["subject"],
        "sample_id": test_raw["sample_id"],
    }
    cams = starter.load_cameras()
    voxels = starter.load_voxels()
    return train, val, test, cams, voxels


# =============================================================================
# GEOMETRY & IDENTIFICATION METRICS
# =============================================================================

def chamfer_matrix(pred, true, verbose=False):
    """
    Computes pairwise Chamfer distance matrix D of shape (n, m) between
    pred (n, 2048, 3) and true (m, 2048, 3).
    Follows metrics.py's definition exactly:
      cKDTree built on the original float32 arrays, d_pg.mean() + d_gp.mean().
    """
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    n = len(pred)
    m = len(true)
    
    is_symmetric = (pred is true) or (n == m and np.array_equal(pred, true))
    
    if is_symmetric:
        trees = [cKDTree(p) for p in pred]
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            if verbose and (i % 25 == 0 or i == n - 1):
                print(f"  [chamfer_matrix] computing row {i+1}/{n} (symmetric)...", flush=True)
            p_i = pred[i]
            tree_i = trees[i]
            for j in range(i + 1, n):
                p_j = pred[j]
                tree_j = trees[j]
                d_i2j = tree_j.query(p_i)[0].mean()
                d_j2i = tree_i.query(p_j)[0].mean()
                dist = float(d_i2j + d_j2i)
                D[i, j] = dist
                D[j, i] = dist
        return D
    else:
        true_trees = [cKDTree(t) for t in true]
        pred_trees = [cKDTree(p) for p in pred]
        D = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            if verbose and (i % 25 == 0 or i == n - 1):
                print(f"  [chamfer_matrix] computing row {i+1}/{n}...", flush=True)
            p_i = pred[i]
            tree_p = pred_trees[i]
            for j in range(m):
                t_j = true[j]
                tree_t = true_trees[j]
                d_p2t = tree_t.query(p_i)[0].mean()
                d_t2p = tree_p.query(t_j)[0].mean()
                D[i, j] = float(d_p2t + d_t2p)
        return D



def per_scan_identification(D, scan_object_index):
    """
    Computes per-scan identification scores from distance matrix D (n_scans, n_objects)
    and scan_object_index (n_scans,).
    Replicates metrics.evaluate exactly:
      For each scan k:
        own = D[k, scan_object_index[k]]
        others = [D[k, o] for o in unique_objects if o != scan_object_index[k]]
        score = 100.0 * mean(own < others)
    Vectorized implementation: O(n_scans * n_objects) in pure numpy.
    """
    D = np.asarray(D, dtype=np.float64)
    scan_object_index = np.asarray(scan_object_index, dtype=int)
    n_scans, n_objects = D.shape
    
    if n_objects <= 1:
        return np.full(n_scans, np.nan, dtype=np.float64)
        
    own = D[np.arange(n_scans), scan_object_index] # (n_scans,)
    # Strict inequality own < other: own < own is False, so summing over all columns
    # and dividing by (n_objects - 1) gives exact mean over all other objects.
    n_better = np.sum(own[:, None] < D, axis=1) # (n_scans,)
    return (100.0 * n_better / (n_objects - 1)).astype(np.float64)


def identification_from_matrix(D, scan_object_index, return_per_scan=False):
    """
    Replicates metrics.evaluate's identification score.
    Returns scalar summary mean, or (summary, per_scan) if return_per_scan=True.
    """
    per_scan = per_scan_identification(D, scan_object_index)
    summary = float(np.nanmean(per_scan))
    if return_per_scan:
        return summary, per_scan
    return summary


def bootstrap_ci(per_scan_values, object_index, B=2000, seed=42):
    """
    Clustered bootstrap for scan-level metrics.
    Resamples OBJECTS with replacement (preserving all scans of each selected object),
    and computes the 95% percentile interval (2.5%, 97.5%) of the scan-level mean.
    """
    per_scan_values = np.asarray(per_scan_values, dtype=np.float64)
    object_index = np.asarray(object_index)
    unique_objs = np.unique(object_index)
    m = len(unique_objs)
    
    obj_to_indices = {o: np.where(object_index == o)[0] for o in unique_objs}
    
    rng = np.random.RandomState(seed)
    boot_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        sampled_objs = rng.choice(unique_objs, size=m, replace=True)
        sampled_indices = np.concatenate([obj_to_indices[o] for o in sampled_objs])
        boot_means[b] = np.mean(per_scan_values[sampled_indices])
        
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return ci_lower, ci_upper


def identification_permutation_test(D, scan_object_index, n=2000, seed=42):
    """
    Permutation test for identification:
    Shuffles which scan receives which prediction (permutes rows of D).
    Returns (observed, null_mean, null_95th_pct, p_value).
      p = (1 + count(null >= obs)) / (1 + n)
    """
    D = np.asarray(D, dtype=np.float32)
    scan_object_index = np.asarray(scan_object_index, dtype=int)
    n_scans = len(D)
    
    obs = identification_from_matrix(D, scan_object_index)
    
    rng = np.random.RandomState(seed)
    null_scores = np.empty(n, dtype=np.float32)
    for i in range(n):
        perm = rng.permutation(n_scans)
        null_scores[i] = identification_from_matrix(D[perm], scan_object_index)
        
    null_mean = float(np.mean(null_scores))
    null_95 = float(np.percentile(null_scores, 95))
    p_val = float((1 + np.sum(null_scores >= obs)) / (1 + n))
    return obs, null_mean, null_95, p_val


def score_val(pred_points):
    """
    Evaluates predictions on the validation set using metrics.evaluate,
    then computes metrics.composite using reference anchors from outputs/phase1/anchors.json.
    Returns (summary, per). summary contains 'composite'.
    """
    val = starter.load_split("val")
    val_true_pts, val_true_masks = starter.targets_for(val)
    cams = starter.load_cameras()
    
    anchors_path = WORKSPACE_ROOT / "outputs" / "phase1" / "anchors.json"
    with open(anchors_path, "r") as f:
        anchors = json.load(f)
        
    summary, per = metrics.evaluate(pred_points, val_true_pts, val_true_masks, val["group"], cams)
    comp = metrics.composite(summary, anchors["generic"], anchors["oracle"])
    summary["composite"] = comp
    return summary, per


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def group_folds(split, k=5, seed=0, verbose=True):
    """
    Generates k grouped and stratified folds using StratifiedGroupKFold:
      y = scan class
      groups = object group
    Ensures no object leaks across folds.
    Returns list of (train_idx, val_idx) tuples.
    """
    scan_classes = split["obj_class"][split["group"]]
    groups = split["group"]
    
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = list(sgkf.split(split["X"], scan_classes, groups))
    
    all_classes = np.unique(scan_classes)
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        tr_classes = np.unique(scan_classes[tr_idx])
        va_classes = np.unique(scan_classes[va_idx])
        has_all_tr = len(tr_classes) == len(all_classes)
        has_all_va = len(va_classes) == len(all_classes)
        if verbose:
            print(f"  Fold {fold_idx}: train scans={len(tr_idx)} (all classes: {has_all_tr}), "
                  f"val scans={len(va_idx)} (all classes: {has_all_va})", flush=True)
        assert has_all_tr, f"Fold {fold_idx} training set missing classes!"
        assert has_all_va, f"Fold {fold_idx} validation set missing classes!"
        
    return folds


# =============================================================================
# FEATURE BUILDERS & NORMALIZATION
# =============================================================================

def window(X, trs):
    """
    Mean of BOLD signal over chosen time indices (float32).
    X shape: (N, 10, V) or (N, T, V).
    trs: list or iterable of integer time indices.
    """
    return np.mean(X[:, trs, :], axis=1).astype(np.float32)


def N3(X):
    """
    Per-scan z-score across voxels.
    Preserves relative spatial patterns while equalizing overall scan amplitude.
    """
    X = np.asarray(X, dtype=np.float32)
    m = np.mean(X, axis=1, keepdims=True)
    s = np.std(X, axis=1, keepdims=True)
    s_safe = np.where(s < 1e-6, 1.0, s)
    return np.where(s < 1e-6, 0.0, (X - m) / s_safe).astype(np.float32)


def fit_n4(X_train, subjects_train, k=0):
    """
    Fits N4 normalization statistics on training data:
      mean_s_shrunk = (n_s * m_s + k * m_global) / (n_s + k), for k in {0, 5, 20}
      scale = per-voxel std of subject-centred training data (pooled).
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    subjects_train = np.asarray(subjects_train)
    n_train, n_voxels = X_train.shape
    
    m_global = np.mean(X_train, axis=0)
    
    unique_subjs = np.unique(subjects_train)
    m_s_dict = {}
    n_s_dict = {}
    mean_s_shrunk_dict = {}
    
    # Compute per-subject mean and shrinkage
    R_train = np.empty_like(X_train)
    for s in unique_subjs:
        idx_s = np.where(subjects_train == s)[0]
        n_s = len(idx_s)
        m_s = np.mean(X_train[idx_s], axis=0)
        m_s_dict[s] = m_s
        n_s_dict[s] = n_s
        
        # Shrinkage towards global mean
        m_shrunk = (n_s * m_s + k * m_global) / (n_s + k)
        mean_s_shrunk_dict[s] = m_shrunk
        
        # Residuals for pooled variance
        R_train[idx_s] = X_train[idx_s] - m_s
        
    # Pooled per-voxel std of subject-centred data
    scale = np.std(R_train, axis=0)
    zero_var_mask = scale < 1e-6
    scale_safe = np.where(zero_var_mask, 1.0, scale)
    
    return {
        "m_global": m_global,
        "mean_s_shrunk": mean_s_shrunk_dict,
        "m_s": m_s_dict,
        "n_s": n_s_dict,
        "scale": scale,
        "scale_safe": scale_safe,
        "zero_var_mask": zero_var_mask,
        "k": k,
        "unique_subjs": set(unique_subjs)
    }


def transform_n4(X, subjects, stats):
    """
    Applies fitted N4 statistics to data using subject IDs.
    Unseen subjects fall back to global mean.
    Zero-variance voxels are mapped to 0.0, never NaN.
    """
    X = np.asarray(X, dtype=np.float32)
    subjects = np.asarray(subjects)
    n = len(X)
    
    Z = np.empty_like(X)
    m_global = stats["m_global"]
    scale_safe = stats["scale_safe"]
    zero_var_mask = stats["zero_var_mask"]
    
    for s in np.unique(subjects):
        idx = np.where(subjects == s)[0]
        if s in stats["mean_s_shrunk"]:
            mu = stats["mean_s_shrunk"][s]
        else:
            mu = m_global
        Z[idx] = (X[idx] - mu) / scale_safe
        
    Z[:, zero_var_mask] = 0.0
    return Z.astype(np.float32)


def N4(X_train, subjects_train, X_heldout=None, subjects_heldout=None, k=0, verify_shrinkage=False):
    """
    Fits N4 on training data and transforms X_train (and optionally X_heldout).
    
    NOTE: N4 statistics are in-sample for training rows and out-of-sample for
    held-out rows, so CV is slightly pessimistic for N4.
    """
    stats = fit_n4(X_train, subjects_train, k=k)
    Z_train = transform_n4(X_train, subjects_train, stats)
    
    # Unit test assertions on Z_train
    assert np.isfinite(Z_train).all(), "N4 transformed train data contains NaN or Inf!"
    
    # Check shrinkage formula assertion when requested by unit tests
    if verify_shrinkage:
        unique_subjs = np.unique(subjects_train)
        for s in unique_subjs:
            idx_s = np.where(subjects_train == s)[0]
            obs_mean = np.mean(Z_train[idx_s], axis=0)
            n_s = stats["n_s"][s]
            m_s = stats["m_s"][s]
            m_global = stats["m_global"]
            scale_safe = stats["scale_safe"]
            
            # Expected post-transformation mean:
            # (m_s - m_shrunk) / scale = (m_s - (n_s*m_s + k*m_global)/(n_s + k)) / scale
            #                          = k * (m_s - m_global) / ((n_s + k) * scale)
            expected_mean = np.where(stats["zero_var_mask"], 0.0,
                                     (k * (m_s - m_global)) / ((n_s + k) * scale_safe))
            diff = np.abs(obs_mean - expected_mean)
            max_diff = np.max(diff)
            if k == 0:
                assert max_diff < 1e-5, f"N4 k=0 subject {s} mean not zero! Max diff: {max_diff}"
            else:
                assert max_diff < 1e-5, f"N4 k={k} subject {s} shrinkage formula mismatch! Max diff: {max_diff}"

    if X_heldout is not None:
        assert subjects_heldout is not None, "subjects_heldout must be provided when X_heldout is given"
        Z_heldout = transform_n4(X_heldout, subjects_heldout, stats)
        assert np.isfinite(Z_heldout).all(), "N4 transformed heldout data contains NaN or Inf!"
        return Z_train, Z_heldout, stats
        
    return Z_train, stats


def N2w(X, subjects, fallback_n4_stats=None, min_scans=3):
    """
    Per (subject, voxel) z-score within the set being transformed (TRANSDUCTIVE).
    For any subject with < min_scans in that set:
      FALLS BACK to N4 for that subject.
    Returns: (Z, affected_scans_count, affected_subjects_list)
    """
    X = np.asarray(X, dtype=np.float32)
    subjects = np.asarray(subjects)
    n, v = X.shape
    Z = np.zeros_like(X)
    
    unique_subjs = np.unique(subjects)
    affected_subjects = []
    affected_scans_count = 0
    
    for s in unique_subjs:
        idx_s = np.where(subjects == s)[0]
        n_s = len(idx_s)
        if n_s < min_scans:
            affected_subjects.append(s)
            affected_scans_count += n_s
            if fallback_n4_stats is not None:
                # Use N4 stats for this subject
                mu = fallback_n4_stats["mean_s_shrunk"].get(s, fallback_n4_stats["m_global"])
                scale_safe = fallback_n4_stats["scale_safe"]
                row_z = (X[idx_s] - mu) / scale_safe
                row_z[:, fallback_n4_stats["zero_var_mask"]] = 0.0
                Z[idx_s] = row_z
            else:
                # Standard global fallback if no N4 provided
                Z[idx_s] = 0.0 # Zero out if < 3 and no fallback available
        else:
            s_mean = np.mean(X[idx_s], axis=0)
            s_std = np.std(X[idx_s], axis=0)
            s_safe = np.where(s_std < 1e-6, 1.0, s_std)
            Z[idx_s] = np.where(s_std < 1e-6, 0.0, (X[idx_s] - s_mean) / s_safe)
            
    assert np.isfinite(Z).all(), "N2w output contains NaN or Inf!"
    return Z.astype(np.float32), affected_scans_count, affected_subjects


# =============================================================================
# STATUS & AUDIT HELPERS
# =============================================================================

def get_config_hash():
    """
    Computes a deterministic hash over all input data arrays and common.py.
    """
    h = hashlib.sha256()
    data_dir = WORKSPACE_ROOT / "data"
    for p in sorted(data_dir.glob("*.npz")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    common_p = Path(__file__).resolve()
    if common_p.exists():
        h.update(common_p.read_bytes())
    return h.hexdigest()[:16]


def update_status(item, status="DONE", evidence_file="", reason=""):
    """
    Updates outputs/phase2/status.json with the item status, timestamp, and config hash.
    """
    status_file = WORKSPACE_ROOT / "outputs" / "phase2" / "status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                records = json.load(f)
        except Exception:
            records = {}
    records[item] = {
        "status": status,
        "evidence_file": str(evidence_file),
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
        "config_hash": get_config_hash()
    }
    with open(status_file, "w") as f:
        json.dump(records, f, indent=2)

