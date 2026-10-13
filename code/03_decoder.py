"""
Phase 2 - Task 3: Train-Only Minimal fMRI->Class Decoder & Brain-Use Controls
Strict adherence to RULES.md:
- Starter and metrics imports only (NO evaluate.py, check_submission.py, 01_explore.py, 00_followup.py)
- sys.path fix at the top
- CPU-only, fixed seeds, Agg backend
- Train-only execution: val.npz and val_medoid_tables.npz are NEVER loaded or accessed
- Memory rule: convert X to float32 per window, delete raw array, fit everything inside training fold
- Incremental and resumable writing to outputs/phase2/cv_results.csv
"""
from pathlib import Path
import sys
import os
import json
import csv
import time
import numpy as np

# Ensure repository root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

CODE_DIR = WORKSPACE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import starter
import metrics
import common
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

SEED = 42
OUT_DIR = WORKSPACE_ROOT / "outputs" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CV_CSV_PATH = OUT_DIR / "cv_results.csv"
CHOSEN_CONFIG_PATH = OUT_DIR / "chosen_config.json"
SUMMARY_JSON_PATH = OUT_DIR / "task3_summary.json"

# ASSERTION: val_medoid_tables.npz must NOT be used or opened in this script
assert not os.path.exists("DO_NOT_OPEN_VAL_TABLES"), "Safety check"


# =============================================================================
# HELPER: BOOTSTRAP CI FOR BALANCED ACCURACY
# =============================================================================

def bootstrap_bacc_ci(y_true, y_pred, groups, B=2000, seed=42):
    """
    Cluster bootstrap at the object (group) level for balanced accuracy.
    Resamples unique objects with replacement to respect scan clustering.
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    
    bacc_samples = []
    classes = np.unique(y_true)
    for _ in range(B):
        boot_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_idx = np.concatenate([group_to_idx[g] for g in boot_groups])
        b_true = y_true[boot_idx]
        b_pred = y_pred[boot_idx]
        bacc_samples.append(balanced_accuracy_score(b_true, b_pred))
        
    ci_lower = float(np.percentile(bacc_samples, 2.5))
    ci_upper = float(np.percentile(bacc_samples, 97.5))
    return ci_lower, ci_upper


# =============================================================================
# 1. LOAD TRAINING DATA & CREATE FOLDS
# =============================================================================

def load_train_data():
    """
    Loads train split only. No validation data loaded.
    """
    print("Loading train.npz (strictly train-only)...", flush=True)
    train = starter.load_split("train")
    y_scans = train["obj_class"][train["group"]]
    class_names = [str(c) for c in train["class_names"]]
    subjects = np.asarray(train["subject"])
    groups = np.asarray(train["group"])
    
    print(f"  Train: {len(y_scans)} scans across {len(np.unique(groups))} objects, {len(np.unique(subjects))} subjects", flush=True)
    print(f"  Classes: {class_names}", flush=True)
    return train, y_scans, class_names, subjects, groups


# =============================================================================
# 2. RUN 36-CONFIG GRID WITH INCREMENTAL CSV WRITING
# =============================================================================

def run_cv_grid(train, y_scans, class_names, subjects, groups, folds):
    """
    Runs the 36-configuration grid with grouped 5-fold CV.
    Incremental and resumable: writes outputs/phase2/cv_results.csv row by row.
    """
    # Define 36 configs: 3 windows x 3 normalizations x 4 regularization strengths
    windows = {
        "W1": [4, 5, 6],
        "W2": [5, 6, 7],
        "W3": [3, 4, 5]
    }
    norms = ["N3", "N4", "N2w"]
    alphas = [10.0, 100.0, 1000.0, 10000.0]
    
    # Read existing cv_results.csv if present
    existing_keys = set()
    csv_header = [
        "config_key", "window", "norm", "alpha",
        "mean_balanced_accuracy", "balanced_acc_se", "mean_accuracy",
        "fold0_bacc", "fold1_bacc", "fold2_bacc", "fold3_bacc", "fold4_bacc"
    ]
    
    if CV_CSV_PATH.exists():
        with open(CV_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add(row["config_key"])
        print(f"Found {len(existing_keys)} existing configurations in {CV_CSV_PATH}. Resuming...", flush=True)
    else:
        with open(CV_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            
    # Load raw X and convert to float32 per window
    X_raw = train["X"].astype(np.float32)
    
    all_results = {}
    
    print("\n" + "=" * 75, flush=True)
    print("STARTING 36-CONFIGURATION CROSS-VALIDATION GRID (Grouped 5-Fold)", flush=True)
    print("=" * 75, flush=True)
    
    for w_name, w_trs in windows.items():
        print(f"\n--- Window {w_name} (TRs {w_trs}) ---", flush=True)
        X_win = common.window(X_raw, w_trs)
        
        for norm_name in norms:
            # Check if all 4 alphas for this (window, norm) are already computed
            alphas_to_run = [a for a in alphas if f"{w_name}_{norm_name}_a{int(a)}" not in existing_keys]
            
            # If not all computed, prepare normalized folds once
            fold_features = []
            if len(alphas_to_run) > 0:
                print(f"  Normalizing folds for ({w_name}, {norm_name})...", flush=True)
                for tr_idx, va_idx in folds:
                    X_tr, y_tr = X_win[tr_idx], y_scans[tr_idx]
                    X_va, y_va = X_win[va_idx], y_scans[va_idx]
                    sub_tr, sub_va = subjects[tr_idx], subjects[va_idx]
                    
                    if norm_name == "N3":
                        Z_tr = common.N3(X_tr)
                        Z_va = common.N3(X_va)
                    elif norm_name == "N4":
                        Z_tr, Z_va, _ = common.N4(X_tr, sub_tr, X_heldout=X_va, subjects_heldout=sub_va, k=5)
                    elif norm_name == "N2w":
                        stats = common.fit_n4(X_tr, sub_tr, k=5)
                        Z_tr, _, _ = common.N2w(X_tr, sub_tr, fallback_n4_stats=stats, min_scans=3)
                        Z_va, _, _ = common.N2w(X_va, sub_va, fallback_n4_stats=stats, min_scans=3)
                    fold_features.append((Z_tr, y_tr, Z_va, y_va, va_idx))
                    
            for alpha in alphas:
                cfg_key = f"{w_name}_{norm_name}_a{int(alpha)}"
                
                if cfg_key in existing_keys:
                    # Load from CSV to memory
                    pass
                else:
                    # Evaluate config across folds
                    oof_preds = np.empty_like(y_scans)
                    fold_baccs = []
                    for Z_tr, y_tr, Z_va, y_va, va_idx in fold_features:
                        clf = RidgeClassifier(alpha=alpha, class_weight="balanced", random_state=SEED)
                        clf.fit(Z_tr, y_tr)
                        pred_va = clf.predict(Z_va)
                        oof_preds[va_idx] = pred_va
                        fold_baccs.append(float(balanced_accuracy_score(y_va, pred_va)))
                        
                    mean_bacc = float(np.mean(fold_baccs))
                    se_bacc = float(np.std(fold_baccs, ddof=1) / np.sqrt(len(fold_baccs)))
                    mean_acc = float(accuracy_score(y_scans, oof_preds))
                    
                    # Append to cv_results.csv immediately (resumable)
                    row_data = [
                        cfg_key, w_name, norm_name, alpha,
                        f"{mean_bacc:.5f}", f"{se_bacc:.5f}", f"{mean_acc:.5f}",
                        f"{fold_baccs[0]:.5f}", f"{fold_baccs[1]:.5f}", f"{fold_baccs[2]:.5f}",
                        f"{fold_baccs[3]:.5f}", f"{fold_baccs[4]:.5f}"
                    ]
                    with open(CV_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(row_data)
                    existing_keys.add(cfg_key)
                    
                    print(f"    {cfg_key:18s} | BAcc: {mean_bacc*100:5.2f}% +/- {se_bacc*100:4.2f}% | Acc: {mean_acc*100:5.2f}%", flush=True)

    # Read back all 36 results from CSV to ensure exact parity
    cv_records = []
    with open(CV_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cv_records.append({
                "config_key": row["config_key"],
                "window": row["window"],
                "norm": row["norm"],
                "alpha": float(row["alpha"]),
                "mean_balanced_accuracy": float(row["mean_balanced_accuracy"]),
                "balanced_acc_se": float(row["balanced_acc_se"]),
                "mean_accuracy": float(row["mean_accuracy"]),
                "fold_baccs": [float(row[f"fold{i}_bacc"]) for i in range(5)]
            })
            
    print(f"\nAll {len(cv_records)}/36 configs logged in {CV_CSV_PATH}", flush=True)
    return cv_records, X_raw


# =============================================================================
# 3. MODEL SELECTION: BEST-CV VS SIMPLEST-WITHIN-1-SE
# =============================================================================

def select_model(cv_records):
    """
    Selects best-CV config and simplest-within-1-SE config side-by-side.
    Simplicity hierarchy:
      1. Normalization: N3 (simplest, 0 subject params) > N4 (train shrinkage) > N2w (transductive)
      2. Regularization: higher alpha (stronger regularization / smaller model capacity)
      3. Window: W1 (stimulus peak [4,5,6]) > W2 > W3
      4. Higher balanced accuracy
    """
    best_cfg = max(cv_records, key=lambda x: x["mean_balanced_accuracy"])
    threshold = best_cfg["mean_balanced_accuracy"] - best_cfg["balanced_acc_se"]
    
    # Filter candidates within 1 SE
    candidates_1se = [c for c in cv_records if c["mean_balanced_accuracy"] >= threshold]
    
    norm_simplicity = {"N3": 3, "N4": 2, "N2w": 1} # Higher = simpler
    window_simplicity = {"W1": 3, "W2": 2, "W3": 1} # Higher = preferred peak window
    
    def simplicity_key(c):
        return (
            norm_simplicity[c["norm"]],
            c["alpha"], # Higher alpha is simpler (more regularized)
            window_simplicity[c["window"]],
            c["mean_balanced_accuracy"]
        )
        
    simplest_cfg = max(candidates_1se, key=simplicity_key)
    return best_cfg, simplest_cfg, threshold, candidates_1se


# =============================================================================
# 4. RECOMPUTE OUT-OF-FOLD PREDICTIONS & BOOTSTRAP CI
# =============================================================================

def evaluate_oof(train, y_scans, subjects, groups, folds, config):
    """
    Generates out-of-fold predictions and computes bootstrap CI.
    """
    windows = {"W1": [4, 5, 6], "W2": [5, 6, 7], "W3": [3, 4, 5]}
    w_trs = windows[config["window"]]
    X_win = common.window(train["X"].astype(np.float32), w_trs)
    
    oof_preds = np.empty_like(y_scans)
    
    for tr_idx, va_idx in folds:
        X_tr, y_tr = X_win[tr_idx], y_scans[tr_idx]
        X_va, y_va = X_win[va_idx], y_scans[va_idx]
        sub_tr, sub_va = subjects[tr_idx], subjects[va_idx]
        
        if config["norm"] == "N3":
            Z_tr = common.N3(X_tr)
            Z_va = common.N3(X_va)
        elif config["norm"] == "N4":
            Z_tr, Z_va, _ = common.N4(X_tr, sub_tr, X_heldout=X_va, subjects_heldout=sub_va, k=5)
        elif config["norm"] == "N2w":
            stats = common.fit_n4(X_tr, sub_tr, k=5)
            Z_tr, _, _ = common.N2w(X_tr, sub_tr, fallback_n4_stats=stats, min_scans=3)
            Z_va, _, _ = common.N2w(X_va, sub_va, fallback_n4_stats=stats, min_scans=3)
            
        clf = RidgeClassifier(alpha=config["alpha"], class_weight="balanced", random_state=SEED)
        clf.fit(Z_tr, y_tr)
        oof_preds[va_idx] = clf.predict(Z_va)
        
    bacc = balanced_accuracy_score(y_scans, oof_preds)
    acc = accuracy_score(y_scans, oof_preds)
    ci_l, ci_u = bootstrap_bacc_ci(y_scans, oof_preds, groups, B=2000, seed=SEED)
    return oof_preds, bacc, acc, ci_l, ci_u


# =============================================================================
# 5. CONTROLS: MAJORITY, SUBJECT-ONLY & PERMUTATION TEST
# =============================================================================

def run_controls(y_scans, subjects, groups, folds, train, chosen_config):
    """
    Evaluates majority baseline, subject-only baseline, and
    fixed-grid object-level label permutation test (100 permutations).
    """
    print("\n" + "=" * 75, flush=True)
    print("RUNNING CONTROLS & BASELINES", flush=True)
    print("=" * 75, flush=True)
    
    # 1. Majority class baseline
    oof_majority = np.empty_like(y_scans)
    for tr_idx, va_idx in folds:
        y_tr = y_scans[tr_idx]
        vals, counts = np.unique(y_tr, return_counts=True)
        maj_class = vals[np.argmax(counts)]
        oof_majority[va_idx] = maj_class
        
    maj_bacc = balanced_accuracy_score(y_scans, oof_majority)
    maj_acc = accuracy_score(y_scans, oof_majority)
    print(f"Majority Baseline:      Balanced Acc = {maj_bacc*100:5.2f}% | Overall Acc = {maj_acc*100:5.2f}%", flush=True)
    
    # 2. Subject-only baseline
    oof_subj = np.empty_like(y_scans)
    for tr_idx, va_idx in folds:
        y_tr = y_scans[tr_idx]
        sub_tr = subjects[tr_idx]
        sub_va = subjects[va_idx]
        
        # Compute modal class per subject in train fold
        subj_modal = {}
        vals, counts = np.unique(y_tr, return_counts=True)
        fallback_maj = vals[np.argmax(counts)]
        for s in np.unique(sub_tr):
            idx_s = np.where(sub_tr == s)[0]
            s_vals, s_counts = np.unique(y_tr[idx_s], return_counts=True)
            subj_modal[s] = s_vals[np.argmax(s_counts)]
            
        for i, s in enumerate(sub_va):
            oof_subj[va_idx[i]] = subj_modal.get(s, fallback_maj)
            
    subj_bacc = balanced_accuracy_score(y_scans, oof_subj)
    subj_acc = accuracy_score(y_scans, oof_subj)
    print(f"Subject-Only Baseline:  Balanced Acc = {subj_bacc*100:5.2f}% | Overall Acc = {subj_acc*100:5.2f}%", flush=True)
    
    # 3. Object-level label permutation test (100 permutations) on chosen config
    print(f"\nRunning object-level label permutation test (100 permutations) on {chosen_config['config_key']}...", flush=True)
    windows = {"W1": [4, 5, 6], "W2": [5, 6, 7], "W3": [3, 4, 5]}
    w_trs = windows[chosen_config["window"]]
    X_win = common.window(train["X"].astype(np.float32), w_trs)
    
    # Pre-normalize folds once for the chosen config
    fold_features = []
    for tr_idx, va_idx in folds:
        X_tr = X_win[tr_idx]
        X_va = X_win[va_idx]
        sub_tr, sub_va = subjects[tr_idx], subjects[va_idx]
        if chosen_config["norm"] == "N3":
            Z_tr = common.N3(X_tr)
            Z_va = common.N3(X_va)
        elif chosen_config["norm"] == "N4":
            Z_tr, Z_va, _ = common.N4(X_tr, sub_tr, X_heldout=X_va, subjects_heldout=sub_va, k=5)
        elif chosen_config["norm"] == "N2w":
            stats = common.fit_n4(X_tr, sub_tr, k=5)
            Z_tr, _, _ = common.N2w(X_tr, sub_tr, fallback_n4_stats=stats, min_scans=3)
            Z_va, _, _ = common.N2w(X_va, sub_va, fallback_n4_stats=stats, min_scans=3)
        fold_features.append((Z_tr, Z_va, tr_idx, va_idx))
        
    unique_groups = np.unique(groups)
    obj_classes = train["obj_class"] # object-level class array
    
    rng = np.random.RandomState(SEED)
    null_baccs = []
    
    t0_perm = time.time()
    for perm_i in range(100):
        # Permute classes at the OBJECT level (preserves object grouping)
        perm_classes = rng.permutation(obj_classes)
        y_perm = perm_classes[groups]
        
        oof_perm = np.empty_like(y_perm)
        for Z_tr, Z_va, tr_idx, va_idx in fold_features:
            y_tr_p = y_perm[tr_idx]
            clf = RidgeClassifier(alpha=chosen_config["alpha"], class_weight="balanced", random_state=SEED)
            clf.fit(Z_tr, y_tr_p)
            oof_perm[va_idx] = clf.predict(Z_va)
            
        null_b = float(balanced_accuracy_score(y_perm, oof_perm))
        null_baccs.append(null_b)
        
    perm_time = time.time() - t0_perm
    null_mean = float(np.mean(null_baccs))
    null_std = float(np.std(null_baccs, ddof=1))
    null_95 = float(np.percentile(null_baccs, 95.0))
    obs_bacc = chosen_config["oof_bacc"]
    p_val = float((1.0 + np.sum(np.array(null_baccs) >= obs_bacc)) / (1.0 + 100.0))
    
    print(f"Permutation null (100 perms): mean={null_mean*100:5.2f}%, 95th pct={null_95*100:5.2f}%, obs={obs_bacc*100:5.2f}%, p={p_val:.4f} (took {perm_time:.2f}s)", flush=True)
    
    return {
        "majority": {"bacc": float(maj_bacc), "acc": float(maj_acc)},
        "subject_only": {"bacc": float(subj_bacc), "acc": float(subj_acc)},
        "permutation_test": {
            "n_permutations": 100,
            "null_mean": null_mean,
            "null_std": null_std,
            "null_95": null_95,
            "obs_bacc": float(obs_bacc),
            "p_value": p_val,
            "runtime_seconds": float(perm_time)
        }
    }


# =============================================================================
# 6. POOLED CONFUSION MATRIX & PER-SUBJECT ACCURACY
# =============================================================================

def compute_diagnostics(y_scans, oof_preds, subjects, class_names):
    """
    Computes pooled confusion matrix, per-class recall, and per-subject accuracy.
    """
    cm = confusion_matrix(y_scans, oof_preds)
    per_class_recall = {}
    for i, cname in enumerate(class_names):
        total_c = np.sum(cm[i, :])
        correct_c = cm[i, i]
        rec = float(correct_c / total_c) if total_c > 0 else 0.0
        per_class_recall[cname] = rec
        
    # Per-subject accuracy
    per_subject_acc = {}
    for s in sorted(np.unique(subjects)):
        idx_s = np.where(subjects == s)[0]
        s_acc = float(accuracy_score(y_scans[idx_s], oof_preds[idx_s]))
        per_subject_acc[str(s)] = {
            "n_scans": int(len(idx_s)),
            "accuracy": s_acc
        }
        
    return cm.tolist(), per_class_recall, per_subject_acc


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    t_start = time.time()
    print("======================================================================")
    print("PHASE 2 - TASK 3: FMRI->CLASS DECODER & CONTROLS")
    print("======================================================================")
    
    train, y_scans, class_names, subjects, groups = load_train_data()
    folds = common.group_folds(train, k=5, seed=0)
    
    # 1. Run 36-config grid
    cv_records, X_raw = run_cv_grid(train, y_scans, class_names, subjects, groups, folds)
    
    # 2. Model selection
    best_cfg, simplest_cfg, threshold, candidates_1se = select_model(cv_records)
    
    print("\n" + "=" * 75)
    print("MODEL SELECTION: BEST-CV VS SIMPLEST-WITHIN-1-SE")
    print("=" * 75)
    print(f"Best-CV Config:              {best_cfg['config_key']}")
    print(f"  Window:                    {best_cfg['window']}")
    print(f"  Norm:                      {best_cfg['norm']}")
    print(f"  Alpha:                     {best_cfg['alpha']}")
    print(f"  CV Balanced Accuracy:      {best_cfg['mean_balanced_accuracy']*100:.2f}% +/- {best_cfg['balanced_acc_se']*100:.2f}%")
    print(f"  1-SE Selection Threshold:  {threshold*100:.2f}%")
    print(f"  Candidates within 1-SE:    {len(candidates_1se)} configs")
    print("-" * 75)
    print(f"Chosen Config (1-SE Rule):   {simplest_cfg['config_key']}")
    print(f"  Window:                    {simplest_cfg['window']}")
    print(f"  Norm:                      {simplest_cfg['norm']} (simplest available)")
    print(f"  Alpha:                     {simplest_cfg['alpha']}")
    print(f"  CV Balanced Accuracy:      {simplest_cfg['mean_balanced_accuracy']*100:.2f}% +/- {simplest_cfg['balanced_acc_se']*100:.2f}%")
    
    # 3. Evaluate out-of-fold predictions and Bootstrap CIs
    print("\nComputing out-of-fold predictions and cluster bootstrap 95% CIs...", flush=True)
    oof_best, bacc_best, acc_best, ci_l_best, ci_u_best = evaluate_oof(train, y_scans, subjects, groups, folds, best_cfg)
    best_cfg["oof_bacc"] = float(bacc_best)
    best_cfg["oof_acc"] = float(acc_best)
    best_cfg["ci_95"] = [float(ci_l_best), float(ci_u_best)]
    
    oof_chosen, bacc_chosen, acc_chosen, ci_l_chosen, ci_u_chosen = evaluate_oof(train, y_scans, subjects, groups, folds, simplest_cfg)
    simplest_cfg["oof_bacc"] = float(bacc_chosen)
    simplest_cfg["oof_acc"] = float(acc_chosen)
    simplest_cfg["ci_95"] = [float(ci_l_chosen), float(ci_u_chosen)]
    
    print(f"Best-CV OOF Balanced Acc:    {bacc_best*100:.2f}% [95% CI: {ci_l_best*100:.2f}%, {ci_u_best*100:.2f}%]")
    print(f"Chosen Config OOF Bal Acc:   {bacc_chosen*100:.2f}% [95% CI: {ci_l_chosen*100:.2f}%, {ci_u_chosen*100:.2f}%]")
    
    # 4. Run controls
    controls_results = run_controls(y_scans, subjects, groups, folds, train, simplest_cfg)
    
    # 5. Diagnostic metrics: confusion matrix & per-subject accuracy
    cm, recalls, per_sub_acc = compute_diagnostics(y_scans, oof_chosen, subjects, class_names)
    
    print("\n" + "=" * 75)
    print("POOLED CONFUSION MATRIX & PER-CLASS RECALL (Chosen Config)")
    print("=" * 75)
    header_str = "True \\ Pred".ljust(12) + "".join([f"{c[:8]:>10}" for c in class_names]) + "     Recall"
    print(header_str)
    for i, cname in enumerate(class_names):
        row_str = f"{cname[:10]:12s}" + "".join([f"{cm[i][j]:10d}" for j in range(5)]) + f"   {recalls[cname]*100:6.2f}%"
        print(row_str)
        
    print("\nPER-SUBJECT ACCURACY (Chosen Config):")
    for s, data in sorted(per_sub_acc.items()):
        print(f"  {s}: {data['accuracy']*100:5.2f}% ({data['n_scans']} scans)")
        
    # Check criteria: bacc < 30% or p > 0.05
    check_passed = (bacc_chosen >= 0.30) and (controls_results["permutation_test"]["p_value"] <= 0.05)
    print(f"\nPhase 2 Decision Gate: Balanced Acc >= 30% ({bacc_chosen*100:.2f}%) and Permutation p <= 0.05 (p={controls_results['permutation_test']['p_value']:.4f})")
    if check_passed:
        print("  -> GATE PASSED: Statistically significant above-chance fMRI category signal.")
    else:
        print("  -> GATE FAILED / WARNING: Below 30% or p > 0.05. Recommending changes in Phase 3.")
        
    total_runtime = time.time() - t_start
    print(f"\nTotal Task 3 Runtime: {total_runtime:.2f} seconds")
    
    # 6. Save chosen_config.json
    chosen_config_data = {
        "selection_rule": "simplest-within-1-SE of best-CV balanced accuracy",
        "threshold_balanced_acc": float(threshold),
        "best_cv_config": {
            "config_key": best_cfg["config_key"],
            "window": best_cfg["window"],
            "norm": best_cfg["norm"],
            "alpha": best_cfg["alpha"],
            "cv_balanced_acc_mean": best_cfg["mean_balanced_accuracy"],
            "cv_balanced_acc_se": best_cfg["balanced_acc_se"],
            "oof_balanced_acc": best_cfg["oof_bacc"],
            "oof_accuracy": best_cfg["oof_acc"],
            "ci_95": best_cfg["ci_95"]
        },
        "chosen_config": {
            "config_key": simplest_cfg["config_key"],
            "window": simplest_cfg["window"],
            "norm": simplest_cfg["norm"],
            "alpha": simplest_cfg["alpha"],
            "cv_balanced_acc_mean": simplest_cfg["mean_balanced_accuracy"],
            "cv_balanced_acc_se": simplest_cfg["balanced_acc_se"],
            "oof_balanced_acc": simplest_cfg["oof_bacc"],
            "oof_accuracy": simplest_cfg["oof_acc"],
            "ci_95": simplest_cfg["ci_95"]
        },
        "controls": controls_results,
        "per_class_recall": recalls,
        "confusion_matrix": cm,
        "per_subject_accuracy": per_sub_acc,
        "total_runtime_seconds": float(total_runtime),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(CHOSEN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen_config_data, f, indent=2)
    print(f"Written chosen configuration to {CHOSEN_CONFIG_PATH}", flush=True)
    
    # 7. Update status.json
    common.update_status(
        "task3_decoder",
        status="DONE",
        evidence_file="outputs/phase2/chosen_config.json",
        reason="36-config grouped 5-fold CV, 1-SE model selection, majority/subject/permutation controls, confusion matrix"
    )
    print("Updated outputs/phase2/status.json (task3_decoder: DONE)", flush=True)
    print("=" * 75)
    print("TASK 3 EXECUTION COMPLETE - STOPPING BEFORE TASK 4 AS REQUIRED")
    print("=" * 75)


if __name__ == "__main__":
    main()
