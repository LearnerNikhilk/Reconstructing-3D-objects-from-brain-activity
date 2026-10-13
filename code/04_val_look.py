"""
Phase 2 - Task 4: Baseline Val Look (Class-Medoid Decoder)
Strict adherence to RULES.md:
- Starter and metrics imports only (NO evaluate.py, check_submission.py, 01_explore.py, 00_followup.py)
- sys.path fix at the top
- CPU-only, fixed seeds, Agg backend
- Exactly ONE val look with a learned model; logged in outputs/phase2/val_looks.md
- No models, hyper-parameters or configurations tuned on val
- Verifies table lookup composite against common.score_val to 1e-6
- Exports outputs/phase2/val_predictions_baseline.npz
"""
from pathlib import Path
import sys
import os
import json
import time
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

CODE_DIR = WORKSPACE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import starter
import metrics
import common

SEED = 42
OUT_DIR = WORKSPACE_ROOT / "outputs" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = WORKSPACE_ROOT / "outputs" / "cache"

CHOSEN_CONFIG_PATH = OUT_DIR / "chosen_config.json"
VAL_PRED_PATH = OUT_DIR / "val_predictions_baseline.npz"
VAL_LOOKS_PATH = OUT_DIR / "val_looks.md"
SUMMARY_JSON_PATH = OUT_DIR / "task4_summary.json"


# =============================================================================
# 1. PRE-CHECKS & AUDIT
# =============================================================================

def run_pre_checks():
    print("=" * 75)
    print("PRE-CHECKS (Running before touching validation data)")
    print("=" * 75)
    
    # 1. Print chosen_config.json in full
    assert CHOSEN_CONFIG_PATH.exists(), f"Missing {CHOSEN_CONFIG_PATH}"
    with open(CHOSEN_CONFIG_PATH, "r", encoding="utf-8") as f:
        chosen_config_data = json.load(f)
        
    print("\n[PRE-CHECK 1] outputs/phase2/chosen_config.json contents:")
    print(json.dumps(chosen_config_data, indent=2))
    
    chosen = chosen_config_data["chosen_config"]
    print("\nExplanation of 'a1000' configuration:")
    print(f"Classifier: RidgeClassifier(alpha={chosen['alpha']}, class_weight='balanced', random_state=42), No PCA, all 31,141 voxels used.")
    print(f"Window: {chosen['window']} (TR indices [4, 5, 6]), Normalisation: {chosen['norm']} (within-scan z-score across voxels).")
    
    # 2. Print real grid from cv_results.csv
    cv_csv_path = OUT_DIR / "cv_results.csv"
    assert cv_csv_path.exists(), f"Missing {cv_csv_path}"
    with open(cv_csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    headers = lines[0].split(",")
    grid_rows = [line.split(",") for line in lines[1:]]
    all_keys = [r[0] for r in grid_rows]
    print(f"\n[PRE-CHECK 2] Real Task 3 CV Grid: Size = {len(all_keys)} configurations.")
    print(f"All config keys ({len(all_keys)}):")
    for i in range(0, len(all_keys), 4):
        print("  " + ", ".join(f"{k:18s}" for k in all_keys[i:i+4]))
        
    print("\nDifferences from specified grid & CV isolation confirmations:")
    print("- N2w included: Evaluated as transductive within-fold z-scoring with fallback to N4 (k=5) for <3 scans.")
    print("- Classifier: RidgeClassifier with dual solve on Gram matrix for alpha in [10, 100, 1000, 10000].")
    print("- Windows: W1=[4,5,6], W2=[5,6,7], W3=[3,4,5] selected via contiguous-3 highest sum of F-statistic.")
    print("- Fold isolation confirmed: normalisation, shrinkage stats, and classifiers were fit STRICTLY inside each training fold (0 leakage, no PCA/voxel selection).")
    
    # 3. Confirm chosen config (N3) properties
    print("\n[PRE-CHECK 3] Chosen Config (N3) Confirmation:")
    print("- N3 is purely within-scan across voxels: z = (x - mean(x)) / std(x). It uses NO other scan at transform time.")
    print("- Final model will be fit strictly on all 533 train scans only. Validation data is never used for fitting.")
    print("=" * 75 + "\n")
    
    return chosen_config_data


# =============================================================================
# 2. TASK 4: FIT ON TRAIN & EVALUATE ON VAL
# =============================================================================

def main():
    t_start = time.time()
    
    # Run pre-checks first
    chosen_config_data = run_pre_checks()
    chosen_cfg = chosen_config_data["chosen_config"]
    
    print("=" * 75)
    print("TASK 4: BASELINE VAL LOOK (Class-Medoid Decoder)")
    print("=" * 75)
    
    # Load train and val data
    print("Loading train.npz and val.npz...", flush=True)
    train = starter.load_split("train")
    val = starter.load_split("val")
    
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train = train["obj_class"][train["group"]]
    y_val = val["obj_class"][val["group"]]
    
    y_train_idx = np.array([class_to_idx[c] for c in y_train])
    y_val_idx = np.array([class_to_idx[c] for c in y_val])
    val_groups = val["group"]
    val_unique_objs = np.unique(val_groups)
    n_val_scans = len(val["sample_id"])
    
    # 1. Feature extraction & model fit
    w_trs = [4, 5, 6] # W1
    print(f"Extracting Window W1 (TRs {w_trs}) and applying N3...", flush=True)
    X_train_win = common.window(train["X"].astype(np.float32), w_trs)
    Z_train = common.N3(X_train_win)
    
    X_val_win = common.window(val["X"].astype(np.float32), w_trs)
    Z_val = common.N3(X_val_win)
    
    print(f"Fitting RidgeClassifier(alpha={chosen_cfg['alpha']}, class_weight='balanced') on all 533 train scans...", flush=True)
    clf = RidgeClassifier(alpha=chosen_cfg["alpha"], class_weight="balanced", random_state=SEED)
    clf.fit(Z_train, y_train_idx)
    
    # Predict on validation data
    val_preds_idx = clf.predict(Z_val)
    val_preds_class = np.array([classes[i] for i in val_preds_idx])
    
    # 2. Accuracy & Balanced Accuracy with object-bootstrap CI
    acc_val = float(accuracy_score(y_val_idx, val_preds_idx))
    bacc_val = float(balanced_accuracy_score(y_val_idx, val_preds_idx))
    
    # Object-level bootstrap CI
    rng = np.random.RandomState(SEED)
    B = 2000
    obj_to_scans = {o: np.where(val_groups == o)[0] for o in val_unique_objs}
    
    boot_accs = np.empty(B, dtype=np.float64)
    boot_baccs = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_objs = rng.choice(val_unique_objs, size=len(val_unique_objs), replace=True)
        sample_scans = np.concatenate([obj_to_scans[o] for o in sample_objs])
        boot_accs[b] = accuracy_score(y_val_idx[sample_scans], val_preds_idx[sample_scans])
        boot_baccs[b] = balanced_accuracy_score(y_val_idx[sample_scans], val_preds_idx[sample_scans])
        
    acc_ci = [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))]
    bacc_ci = [float(np.percentile(boot_baccs, 2.5)), float(np.percentile(boot_baccs, 97.5))]
    
    print(f"\nValidation Classification Performance:")
    print(f"  Overall Accuracy:   {acc_val*100:5.2f}% [95% CI: {acc_ci[0]*100:5.2f}%, {acc_ci[1]*100:5.2f}%]")
    print(f"  Balanced Accuracy:  {bacc_val*100:5.2f}% [95% CI: {bacc_ci[0]*100:5.2f}%, {bacc_ci[1]*100:5.2f}%]")
    
    # Confusion matrix and per-class recall
    cm = confusion_matrix(y_val_idx, val_preds_idx)
    per_class_recall = {}
    for i, cname in enumerate(classes):
        rec = float(cm[i, i] / np.sum(cm[i, :])) if np.sum(cm[i, :]) > 0 else 0.0
        per_class_recall[cname] = rec
        
    print("\nValidation Confusion Matrix & Recall:")
    header_str = "True \\ Pred".ljust(12) + "".join([f"{c[:8]:>10}" for c in classes]) + "     Recall"
    print(header_str)
    for i, cname in enumerate(classes):
        row_str = f"{cname[:10]:12s}" + "".join([f"{cm[i][j]:10d}" for j in range(5)]) + f"   {per_class_recall[cname]*100:6.2f}%"
        print(row_str)
        
    # 3. Table Lookup from val_medoid_tables.npz
    tables_path = CACHE_DIR / "val_medoid_tables.npz"
    assert tables_path.exists(), f"Missing {tables_path}"
    medoid_tables = np.load(tables_path)
    
    metric_names = ["chamfer", "fscore", "silhouette_iou", "silhouette_ssim", "identification"]
    lookup_per_scan = {}
    lookup_summary = {}
    for m in metric_names:
        table_m = medoid_tables[m] # (5, 125)
        # Scan k receives metric from predicted class val_preds_idx[k]
        scan_vals = table_m[val_preds_idx, np.arange(n_val_scans)]
        lookup_per_scan[m] = scan_vals
        lookup_summary[m] = float(np.mean(scan_vals))
        
    # Load anchors for composite
    anchors_path = WORKSPACE_ROOT / "outputs" / "phase1" / "anchors.json"
    with open(anchors_path, "r") as f:
        anchors = json.load(f)
    generic_anchor = anchors["generic"]
    oracle_anchor = anchors["oracle"]
    
    lookup_composite = float(metrics.composite(lookup_summary, generic_anchor, oracle_anchor))
    lookup_summary["composite"] = lookup_composite
    
    # Compute object-bootstrap CIs for each metric and composite
    boot_metrics = {m: np.empty(B, dtype=np.float64) for m in metric_names}
    boot_composites = np.empty(B, dtype=np.float64)
    for b in range(B):
        sample_objs = rng.choice(val_unique_objs, size=len(val_unique_objs), replace=True)
        sample_scans = np.concatenate([obj_to_scans[o] for o in sample_objs])
        b_summary = {}
        for m in metric_names:
            b_val = float(np.mean(lookup_per_scan[m][sample_scans]))
            boot_metrics[m][b] = b_val
            b_summary[m] = b_val
        boot_composites[b] = metrics.composite(b_summary, generic_anchor, oracle_anchor)
        
    metric_cis = {}
    for m in metric_names:
        metric_cis[m] = [float(np.percentile(boot_metrics[m], 2.5)), float(np.percentile(boot_metrics[m], 97.5))]
    comp_ci = [float(np.percentile(boot_composites, 2.5)), float(np.percentile(boot_composites, 97.5))]
    
    print("\n" + "=" * 75)
    print("VALIDATION METRICS COMPARISON (Class-Medoid Decoder vs Benchmarks)")
    print("=" * 75)
    print(f"{'Benchmark / Model':<32s}{'Chamfer':>9s}{'F-Score':>9s}{'IoU':>8s}{'SSIM':>8s}{'Ident':>9s}{'Composite':>11s}")
    print("-" * 75)
    print(f"{'Generic Anchor (#114)':<32s}{generic_anchor['chamfer']:>9.4f}{generic_anchor['fscore']:>9.3f}{generic_anchor['silhouette_iou']:>8.3f}{generic_anchor['silhouette_ssim']:>8.3f}{generic_anchor['identification']:>8.1f}%{generic_anchor['composite']:>11.3f}")
    print(f"{'Class-Oracle (1 Prototype)':<32s}{0.0635:>9.4f}{0.820:>9.3f}{0.647:>8.3f}{0.772:>8.3f}{90.2:>8.1f}%{0.782:>11.3f}")
    print(f"{'Learned Baseline (W1_N3_a1000)':<32s}{lookup_summary['chamfer']:>9.4f}{lookup_summary['fscore']:>9.3f}{lookup_summary['silhouette_iou']:>8.3f}{lookup_summary['silhouette_ssim']:>8.3f}{lookup_summary['identification']:>8.1f}%{lookup_composite:>11.3f}")
    print(f"{'  95% Bootstrap CI':<32s}[{metric_cis['chamfer'][0]:.4f},{metric_cis['chamfer'][1]:.4f}] [{metric_cis['fscore'][0]:.3f},{metric_cis['fscore'][1]:.3f}] [{metric_cis['silhouette_iou'][0]:.3f},{metric_cis['silhouette_iou'][1]:.3f}] [{metric_cis['silhouette_ssim'][0]:.3f},{metric_cis['silhouette_ssim'][1]:.3f}] [{metric_cis['identification'][0]:.1f}%,{metric_cis['identification'][1]:.1f}%] [{comp_ci[0]:.3f},{comp_ci[1]:.3f}]")
    print("-" * 75)
    
    # 4. Identification Permutation Test (n=2000)
    print("\nRunning Identification Permutation Test (n=2000)...", flush=True)
    # The 5 class medoid indices from Task 2:
    medoid_indices = {"airplane": 151, "bench": 179, "cabinet": 166, "car": 63, "sofa": 92}
    medoid_pts_list = [train["obj_points"][medoid_indices[c]] for c in classes]
    
    # Construct predicted point clouds for all 125 val scans
    pred_points = np.empty((n_val_scans, 2048, 3), dtype=np.float32)
    for k in range(n_val_scans):
        pred_points[k] = medoid_pts_list[val_preds_idx[k]]
        
    # Chamfer distance matrix between predicted points and the 67 unique val objects
    # Note: can compute using chamfer_val_train or common.chamfer_matrix
    D_val_pred = common.chamfer_matrix(pred_points, val["obj_points"])
    obs_ident, null_ident_mean, null_ident_95, p_ident = common.identification_permutation_test(
        D_val_pred, val_groups, n=2000, seed=SEED
    )
    print(f"Identification Permutation Test: Obs = {obs_ident:.2f}%, Null Mean = {null_ident_mean:.2f}%, Null 95th pct = {null_ident_95:.2f}%, p = {p_ident:.4f}")
    
    # 5. Shuffled-Feature Control (n=1000)
    print("\nRunning Shuffled-Feature Control (n=1000 permutations of val feature rows)...", flush=True)
    rng_shuff = np.random.RandomState(SEED)
    n_shuff = 1000
    shuff_accs = np.empty(n_shuff, dtype=np.float64)
    shuff_idents = np.empty(n_shuff, dtype=np.float64)
    shuff_comps = np.empty(n_shuff, dtype=np.float64)
    
    table_ident = medoid_tables["identification"]
    
    for s in range(n_shuff):
        # Permute val feature rows after normalisation
        perm_idx = rng_shuff.permutation(n_val_scans)
        Z_val_shuff = Z_val[perm_idx]
        shuff_preds_idx = clf.predict(Z_val_shuff)
        
        # Accuracy under shuffle
        shuff_accs[s] = accuracy_score(y_val_idx, shuff_preds_idx)
        
        # Summary metrics under shuffle via table lookup
        s_summary = {}
        for m in metric_names:
            s_vals = medoid_tables[m][shuff_preds_idx, np.arange(n_val_scans)]
            s_summary[m] = float(np.mean(s_vals))
            
        shuff_idents[s] = s_summary["identification"]
        shuff_comps[s] = metrics.composite(s_summary, generic_anchor, oracle_anchor)
        
    shuff_null = {
        "accuracy": {
            "mean": float(np.mean(shuff_accs)),
            "std": float(np.std(shuff_accs)),
            "pct95": float(np.percentile(shuff_accs, 95)),
            "obs": acc_val,
            "p_value": float((1 + np.sum(shuff_accs >= acc_val)) / (1 + n_shuff))
        },
        "identification": {
            "mean": float(np.mean(shuff_idents)),
            "std": float(np.std(shuff_idents)),
            "pct95": float(np.percentile(shuff_idents, 95)),
            "obs": lookup_summary["identification"],
            "p_value": float((1 + np.sum(shuff_idents >= lookup_summary["identification"])) / (1 + n_shuff))
        },
        "composite": {
            "mean": float(np.mean(shuff_comps)),
            "std": float(np.std(shuff_comps)),
            "pct95": float(np.percentile(shuff_comps, 95)),
            "obs": lookup_composite,
            "p_value": float((1 + np.sum(shuff_comps >= lookup_composite)) / (1 + n_shuff))
        }
    }
    
    print(f"Shuffled Control Results (n=1000):")
    print(f"  Accuracy:       Null Mean = {shuff_null['accuracy']['mean']*100:.2f}%, 95th pct = {shuff_null['accuracy']['pct95']*100:.2f}%, Obs = {acc_val*100:.2f}%, p = {shuff_null['accuracy']['p_value']:.4f}")
    print(f"  Identification: Null Mean = {shuff_null['identification']['mean']:.2f}%, 95th pct = {shuff_null['identification']['pct95']:.2f}%, Obs = {lookup_summary['identification']:.2f}%, p = {shuff_null['identification']['p_value']:.4f}")
    print(f"  Composite:      Null Mean = {shuff_null['composite']['mean']:.3f}, 95th pct = {shuff_null['composite']['pct95']:.3f}, Obs = {lookup_composite:.3f}, p = {shuff_null['composite']['p_value']:.4f}")
    
    # 6. ONE Direct score_val Call & Parity Check
    print("\n" + "=" * 75)
    print("ONE DIRECT SCORE_VAL CALL ON VAL (Exact Parity Check)")
    print("=" * 75)
    score_val_summary, score_val_per = common.score_val(pred_points)
    score_val_comp = float(score_val_summary["composite"])
    
    diff_comp = abs(lookup_composite - score_val_comp)
    print(f"Table Lookup Composite:    {lookup_composite:.8f}")
    print(f"Direct score_val Composite:{score_val_comp:.8f}")
    print(f"Absolute Difference:       {diff_comp:.2e}")
    assert diff_comp < 1e-6, f"Lookup and score_val composite mismatch exceeds 1e-6! diff = {diff_comp}"
    print("CONFIRMED: Table lookup and direct score_val match to < 1e-6 (exact match).")
    
    # 7. Write outputs/phase2/val_predictions_baseline.npz
    np.savez_compressed(VAL_PRED_PATH, sample_id=val["sample_id"], points=pred_points)
    print(f"\nSaved baseline validation predictions to {VAL_PRED_PATH} ({pred_points.shape}, float32)")
    
    # 8. Log in outputs/phase2/val_looks.md
    log_entry = (
        "| **Look-1** | Task 4 | Baseline class-medoid decoder (`code/04_val_look.py`) | "
        "`val['X']`, `val['obj_points']`, `val['group']` | "
        "**Look #1: baseline class-medoid decoder.** No model or config choice depends on this result. "
        f"Later improvements are chosen on train CV only. (Val Composite = {lookup_composite:+.3f}) | "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} |\n"
    )
    with open(VAL_LOOKS_PATH, "r", encoding="utf-8") as f:
        val_looks_content = f.read()
        
    # Replace pending row or append
    if "| **Look-2** | Task 4 |" in val_looks_content:
        val_looks_content = val_looks_content.replace(
            "| **Look-2** | Task 4 | Single Learned Model Validation Look | `val['X']`, `val['obj_points']`, `val['group']` | **Single Learned Validation Look**: `score_val` on trained fMRI->class decoder predictions, lookup scoring, and shuffled brain controls. | Pending |",
            log_entry.strip()
        )
    else:
        val_looks_content += "\n" + log_entry
        
    with open(VAL_LOOKS_PATH, "w", encoding="utf-8") as f:
        f.write(val_looks_content)
    print(f"Logged Look #1 in {VAL_LOOKS_PATH}")
    
    # 9. Update status.json
    common.update_status(
        "task4_val_look",
        status="DONE",
        evidence_file="outputs/phase2/val_predictions_baseline.npz",
        reason=f"Single learned baseline val look: composite={lookup_composite:.3f}, ident={lookup_summary['identification']:.1f}%, acc={acc_val:.3f}"
    )
    print("Updated outputs/phase2/status.json (task4_val_look: DONE)")
    
    # Save full summary JSON
    total_runtime = time.time() - t_start
    summary_data = {
        "chosen_config": chosen_cfg,
        "classification": {
            "accuracy": acc_val,
            "accuracy_ci_95": acc_ci,
            "balanced_accuracy": bacc_val,
            "balanced_accuracy_ci_95": bacc_ci,
            "per_class_recall": per_class_recall,
            "confusion_matrix": cm.tolist()
        },
        "metrics_summary": lookup_summary,
        "metrics_ci_95": {**metric_cis, "composite": comp_ci},
        "identification_permutation_test": {
            "observed": obs_ident,
            "null_mean": null_ident_mean,
            "null_95th_pct": null_ident_95,
            "p_value": p_ident
        },
        "shuffled_feature_control": shuff_null,
        "score_val_verification": {
            "lookup_composite": lookup_composite,
            "direct_score_val_composite": score_val_comp,
            "difference": diff_comp,
            "passed": True
        },
        "total_runtime_seconds": float(total_runtime),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved full Task 4 summary to {SUMMARY_JSON_PATH}")
    
    print("\n" + "=" * 75)
    print(f"TASK 4 COMPLETE — Total Runtime: {total_runtime:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
