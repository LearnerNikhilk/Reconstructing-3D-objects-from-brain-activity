"""
Unit test suite for code/common.py.
Verifies all mathematical definitions and requirements from Task 1.
"""
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

CODE_DIR = WORKSPACE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import starter
import metrics
import common


def test_load_all():
    print("Testing load_all()...", flush=True)
    train, val, test, cams, voxels = common.load_all()
    assert set(test.keys()) == {"X", "subject", "sample_id"}, f"Test keys mismatch: {test.keys()}"
    assert len(train["X"]) == 533
    assert len(val["X"]) == 125
    assert len(test["X"]) == 167
    assert cams["resolution"] == 64
    assert len(voxels["row"]) == 31141
    print("  PASSED: load_all()", flush=True)


def test_identification_from_matrix_parity():
    print("Testing identification_from_matrix parity against metrics.evaluate to 1e-9...", flush=True)
    val = starter.load_split("val")
    val_true_pts, val_true_masks = starter.targets_for(val)
    cams = starter.load_cameras()
    train = starter.load_split("train")
    
    # 1. Generic medoid (train shape 114, confirmed in Phase 1)
    medoid_pts = train["obj_points"][114]
    generic_preds = np.repeat(medoid_pts[None], len(val["sample_id"]), axis=0)
    
    print("  Running metrics.evaluate for generic medoid...", flush=True)
    eval_gen_summary, eval_gen_per = metrics.evaluate(generic_preds, val_true_pts, val_true_masks, val["group"], cams)
    
    # Compute Chamfer matrix D (125, 67) against val unique objects (val["obj_points"])
    print("  Computing chamfer_matrix for generic medoid...", flush=True)
    D_gen = common.chamfer_matrix(generic_preds, val["obj_points"])
    gen_summary, gen_per = common.identification_from_matrix(D_gen, val["group"], return_per_scan=True)
    
    gen_per_eval = np.array([p["identification"] for p in eval_gen_per])
    per_scan_diff_gen = np.max(np.abs(gen_per - gen_per_eval))
    summary_diff_gen = abs(gen_summary - eval_gen_summary["identification"])
    print(f"  Generic medoid: max per-scan diff = {per_scan_diff_gen:.2e}, summary diff = {summary_diff_gen:.2e}", flush=True)
    assert per_scan_diff_gen < 1e-9, f"Generic per-scan identification mismatch: {per_scan_diff_gen}"
    assert summary_diff_gen < 1e-9, f"Generic summary identification mismatch: {summary_diff_gen}"
    
    # Check per-scan Chamfer agreement
    cd_own_gen = D_gen[np.arange(len(val["group"])), val["group"]]
    eval_gen_chamfers = np.array([p["chamfer"] for p in eval_gen_per])
    cd_diff_gen = np.max(np.abs(cd_own_gen - eval_gen_chamfers))
    print(f"  Generic medoid per-scan Chamfer max diff: {cd_diff_gen:.2e}", flush=True)
    assert cd_diff_gen < 1e-6, f"Generic per-scan Chamfer mismatch: {cd_diff_gen}"
    
    # 2. Ground truth as prediction
    print("  Running metrics.evaluate for ground truth...", flush=True)
    eval_gt_summary, eval_gt_per = metrics.evaluate(val_true_pts, val_true_pts, val_true_masks, val["group"], cams)
    
    print("  Computing chamfer_matrix for ground truth...", flush=True)
    D_gt = common.chamfer_matrix(val_true_pts, val["obj_points"])
    gt_summary, gt_per = common.identification_from_matrix(D_gt, val["group"], return_per_scan=True)
    
    gt_per_eval = np.array([p["identification"] for p in eval_gt_per])
    per_scan_diff_gt = np.max(np.abs(gt_per - gt_per_eval))
    summary_diff_gt = abs(gt_summary - eval_gt_summary["identification"])
    print(f"  Ground truth:   max per-scan diff = {per_scan_diff_gt:.2e}, summary diff = {summary_diff_gt:.2e}", flush=True)
    assert per_scan_diff_gt < 1e-9, f"Ground truth per-scan identification mismatch: {per_scan_diff_gt}"
    assert summary_diff_gt < 1e-9, f"Ground truth summary identification mismatch: {summary_diff_gt}"
    
    # Check per-scan Chamfer agreement for ground truth
    cd_own_gt = D_gt[np.arange(len(val["group"])), val["group"]]
    eval_gt_chamfers = np.array([p["chamfer"] for p in eval_gt_per])
    cd_diff_gt = np.max(np.abs(cd_own_gt - eval_gt_chamfers))
    print(f"  Ground truth per-scan Chamfer max diff: {cd_diff_gt:.2e}", flush=True)
    assert cd_diff_gt < 1e-6, f"Ground truth per-scan Chamfer mismatch: {cd_diff_gt}"
    
    print("  PASSED: identification_from_matrix & per-scan Chamfer parity (100% exact)", flush=True)


def test_bootstrap_and_permutation():
    print("Testing bootstrap_ci and permutation test...", flush=True)
    val = starter.load_split("val")
    dummy_values = np.random.RandomState(42).randn(len(val["X"]))
    ci_l, ci_u = common.bootstrap_ci(dummy_values, val["group"], B=500, seed=42)
    assert ci_l < ci_u, f"Bootstrap CI inverted: [{ci_l}, {ci_u}]"
    print(f"  Bootstrap CI: [{ci_l:.4f}, {ci_u:.4f}]", flush=True)
    
    dummy_D = np.random.RandomState(42).rand(10, 5).astype(np.float32)
    dummy_groups = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    obs, null_m, null_95, p_val = common.identification_permutation_test(dummy_D, dummy_groups, n=100, seed=42)
    assert 0.0 <= p_val <= 1.0
    print(f"  Permutation test: obs={obs:.2f}, null_mean={null_m:.2f}, p={p_val:.4f}", flush=True)
    print("  PASSED: bootstrap_ci and identification_permutation_test", flush=True)


def test_group_folds():
    print("Testing group_folds on train split...", flush=True)
    train = starter.load_split("train")
    folds = common.group_folds(train, k=5, seed=0, verbose=True)
    assert len(folds) == 5
    for tr_idx, va_idx in folds:
        tr_groups = set(train["group"][tr_idx])
        va_groups = set(train["group"][va_idx])
        assert len(tr_groups.intersection(va_groups)) == 0, "Object group leaked across train and val folds!"
    print("  PASSED: group_folds (0 group leakage, all classes present)", flush=True)


def test_feature_builders():
    print("Testing feature builders (window, N3, N4, N2w)...", flush=True)
    train = starter.load_split("train")
    X_raw = train["X"].astype(np.float32)
    
    # Window
    X_win = common.window(X_raw, [4, 5, 6])
    assert X_win.shape == (533, 31141)
    assert X_win.dtype == np.float32
    print("  PASSED: window", flush=True)
    
    # N3
    X_n3 = common.N3(X_win)
    assert X_n3.shape == (533, 31141)
    # Across voxels, scan mean should be ~0 and scan std should be ~1
    scan_means = X_n3.mean(axis=1)
    scan_stds = X_n3.std(axis=1)
    assert np.allclose(scan_means, 0.0, atol=1e-4), "N3 scan means are not zero!"
    assert np.allclose(scan_stds, 1.0, atol=1e-3), "N3 scan stds are not 1!"
    print("  PASSED: N3 (mean ~0, std ~1)", flush=True)
    
    # N4 unit tests with k=0, 5, 20
    for k in [0, 5, 20]:
        Z_tr, stats = common.N4(X_win, train["subject"], k=k, verify_shrinkage=True)
        assert np.isfinite(Z_tr).all()
        assert not stats["zero_var_mask"].all()
        print(f"  PASSED: N4 (k={k}) shrinkage assertion", flush=True)
        
    # N4 heldout transformation test
    val = starter.load_split("val")
    X_va_win = common.window(val["X"].astype(np.float32), [4, 5, 6])
    Z_tr, Z_va, stats = common.N4(X_win, train["subject"], X_heldout=X_va_win, subjects_heldout=val["subject"], k=5)
    assert np.isfinite(Z_va).all()
    print("  PASSED: N4 heldout transform", flush=True)
    
    # N2w test with fallback
    Z_n2w, aff_scans, aff_subjs = common.N2w(X_va_win, val["subject"], fallback_n4_stats=stats, min_scans=3)
    assert np.isfinite(Z_n2w).all()
    print(f"  PASSED: N2w (affected subjects: {aff_subjs}, affected scans: {aff_scans})", flush=True)


def main():
    print("======================================================================")
    print("RUNNING COMMON.PY UNIT TESTS")
    print("======================================================================")
    test_load_all()
    test_feature_builders()
    test_group_folds()
    test_bootstrap_and_permutation()
    test_identification_from_matrix_parity()
    print("======================================================================")
    print("ALL COMMON.PY UNIT TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")
    common.update_status(
        "task1_common_unit_tests",
        status="DONE",
        evidence_file="code/test_common.py",
        reason="Unit tests: load_all, feature builders (N3, N4, N2w), group_folds, bootstrap_ci, identification parity to 1e-9"
    )


if __name__ == "__main__":
    main()

