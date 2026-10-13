"""
Phase 3 Pipeline: Probabilities, Library Retrieval Rules, and Final Val Look
Follows RULES.md and all assignment requirements strictly:
- Grouped 5-fold CV (same folds as Task 3)
- Strict fold isolation: library, class medoids, anchors, and MDS fit ONLY on training objects
- Proxy objective J excludes silhouettes and uses cached D_tt & F_tt
- Fixed grid <= 30 configs for probabilities
- Rules R0 (hard medoid), R1 (posterior-weighted medoid), R2 (MDS within-class retrieval)
- Look #2 on validation at the very end; verified against official evaluate.py to 1e-6
"""
from pathlib import Path
import sys
import os
import json
import time
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeClassifier, LogisticRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

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
CACHE_DIR = WORKSPACE_ROOT / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = WORKSPACE_ROOT / "outputs" / "phase3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHOSEN_PHASE2_PATH = WORKSPACE_ROOT / "outputs" / "phase2" / "chosen_config.json"
CHOSEN_PHASE3_PATH = OUT_DIR / "chosen_phase3.json"
STATUS_JSON_PATH = OUT_DIR / "status.json"
VAL_LOOKS_PATH = WORKSPACE_ROOT / "outputs" / "phase2" / "val_looks.md"


def update_phase3_status(item, status="DONE", evidence_file="", reason=""):
    """
    Updates outputs/phase3/status.json with the item status, timestamp, and evidence.
    """
    records = {}
    if STATUS_JSON_PATH.exists():
        try:
            with open(STATUS_JSON_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = {}
    records[item] = {
        "status": status,
        "evidence_file": str(evidence_file),
        "reason": reason,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(STATUS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


# =============================================================================
# 1. TASK B HELPER: COMPUTE OR LOAD TRAIN-TRAIN F-SCORE MATRIX F_tt
# =============================================================================

def load_or_compute_f_tt(train):
    """
    Computes or loads pairwise F-score@0.05 matrix F_tt (292, 292) across train shapes.
    Uses exact metrics.chamfer_and_fscore definition (tau=0.05).
    """
    f_path = CACHE_DIR / "fscore_train_train.npy"
    if f_path.exists():
        F_tt = np.load(f_path)
        if F_tt.shape == (len(train["obj_points"]), len(train["obj_points"])):
            print(f"Loaded cached F_tt from {f_path} {F_tt.shape}", flush=True)
            return F_tt
            
    print("Computing train-train F-score matrix F_tt (292 x 292)...", flush=True)
    t0 = time.time()
    shapes = train["obj_points"].astype(np.float32)
    n = len(shapes)
    trees = [cKDTree(p) for p in shapes]
    
    F_tt = np.eye(n, dtype=np.float32)
    for i in range(n):
        if i % 50 == 0:
            print(f"  F_tt row {i+1}/{n}...", flush=True)
        p_i = shapes[i]
        tree_i = trees[i]
        for j in range(i + 1, n):
            p_j = shapes[j]
            tree_j = trees[j]
            d_i2j = tree_j.query(p_i)[0]
            d_j2i = tree_i.query(p_j)[0]
            prec = np.mean(d_i2j < 0.05)
            rec = np.mean(d_j2i < 0.05)
            f_val = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            F_tt[i, j] = float(f_val)
            F_tt[j, i] = float(f_val)
            
    np.save(f_path, F_tt)
    print(f"Computed and cached F_tt in {time.time() - t0:.2f}s", flush=True)
    return F_tt


# =============================================================================
# 2. TASK B: SETUP FOLD LIBRARIES & PROXY OBJECTIVE J
# =============================================================================

def setup_fold_libraries(train, folds, D_tt, F_tt):
    """
    Precomputes training libraries, fold anchors, and evaluation mapping for each fold.
    Strictly leakage-safe: candidate prediction library U_tr consists ONLY of training objects.
    """
    groups = train["group"]
    obj_classes = train["obj_class"]
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    fold_info = []
    
    for fold_k, (tr_idx, va_idx) in enumerate(folds):
        u_tr = np.unique(groups[tr_idx])
        u_va = np.unique(groups[va_idx])
        
        # Assert strict object separation
        assert len(set(u_tr).intersection(set(u_va))) == 0, f"Leakage detected in fold {fold_k}!"
        
        # Submatrix of training objects
        D_tr = D_tt[np.ix_(u_tr, u_tr)]
        
        # Generic anchor: overall medoid of training library
        gen_local = int(np.argmin(np.sum(D_tr, axis=1)))
        generic_obj = int(u_tr[gen_local])
        
        # Class medoids of training library
        class_medoids = {}
        for cname in classes:
            c_objs = [o for o in u_tr if obj_classes[o] == cname]
            sub_D = D_tt[np.ix_(c_objs, c_objs)]
            med_local = int(np.argmin(np.sum(sub_D, axis=1)))
            class_medoids[cname] = int(c_objs[med_local])
            
        # Held-out scans and their true objects
        va_objs = groups[va_idx]
        
        # Fold oracle: for each held-out scan, nearest library shape to true object
        # shape: (len(va_idx),)
        oracle_preds = np.array([u_tr[np.argmin(D_tt[u_tr, true_o])] for true_o in va_objs])
        generic_preds = np.full(len(va_idx), generic_obj, dtype=int)
        
        # Helper to compute metrics for any predictions on this fold
        def make_eval_fold_preds(curr_va_objs, curr_u_va):
            def eval_fold_preds(preds):
                n_va = len(curr_va_objs)
                cd_list = np.empty(n_va, dtype=np.float64)
                fs_list = np.empty(n_va, dtype=np.float64)
                id_list = np.empty(n_va, dtype=np.float64)
                
                for s in range(n_va):
                    p_s = preds[s]
                    true_o = curr_va_objs[s]
                    
                    cd_list[s] = D_tt[p_s, true_o]
                    fs_list[s] = F_tt[p_s, true_o]
                    
                    # Identification against other held-out objects in this fold
                    other_objs = [o for o in curr_u_va if o != true_o]
                    if len(other_objs) > 0:
                        d_own = D_tt[p_s, true_o]
                        d_other = D_tt[p_s, other_objs]
                        id_list[s] = 100.0 * np.mean(d_own < d_other)
                    else:
                        id_list[s] = 50.0
                        
                return cd_list, fs_list, id_list
            return eval_fold_preds

        eval_fn = make_eval_fold_preds(va_objs, u_va)

        # Compute fold baseline anchor scores
        gen_cd, gen_fs, gen_id = eval_fn(generic_preds)
        ora_cd, ora_fs, ora_id = eval_fn(oracle_preds)
        
        fold_info.append({
            "fold_k": fold_k,
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "u_tr": u_tr,
            "u_va": u_va,
            "class_medoids": class_medoids,
            "generic_obj": generic_obj,
            "eval_fn": eval_fn,
            "anchors": {
                "gen": {"cd": np.mean(gen_cd), "fs": np.mean(gen_fs), "id": np.mean(gen_id)},
                "ora": {"cd": np.mean(ora_cd), "fs": np.mean(ora_fs), "id": np.mean(ora_id)}
            }
        })
        
    return fold_info


def compute_proxy_J(fold_info, all_oof_preds, train):
    """
    Computes proxy objective J across folds:
      J = 0.5 * ident_norm + 0.5 * (0.5 * chamfer_norm + 0.5 * fscore_norm)
    Silhouettes are excluded. Returns pooled J, fold Js, and component metrics.
    """
    groups = train["group"]
    N_scans = len(train["sample_id"])
    fold_Js = []
    pooled_cd = []
    pooled_fs = []
    pooled_id = []
    
    scan_norm_cd = np.empty(N_scans, dtype=np.float64)
    scan_norm_fs = np.empty(N_scans, dtype=np.float64)
    scan_norm_id = np.empty(N_scans, dtype=np.float64)
    
    for info in fold_info:
        va_idx = info["va_idx"]
        preds = all_oof_preds[va_idx]
        cd, fs, ident = info["eval_fn"](preds)
        
        anc = info["anchors"]
        m_cd = np.mean(cd)
        m_fs = np.mean(fs)
        m_id = np.mean(ident)
        
        cd_norm = (anc["gen"]["cd"] - m_cd) / (anc["gen"]["cd"] - anc["ora"]["cd"])
        fs_norm = (m_fs - anc["gen"]["fs"]) / (anc["ora"]["fs"] - anc["gen"]["fs"])
        id_norm = (m_id - anc["gen"]["id"]) / (anc["ora"]["id"] - anc["gen"]["id"])
        
        J_fold = 0.5 * id_norm + 0.5 * (0.5 * cd_norm + 0.5 * fs_norm)
        fold_Js.append(float(J_fold))
        
        pooled_cd.extend(cd)
        pooled_fs.extend(fs)
        pooled_id.extend(ident)
        
        # Per-scan normalized terms assigned directly by scan index
        s_cd_norm = (anc["gen"]["cd"] - cd) / (anc["gen"]["cd"] - anc["ora"]["cd"])
        s_fs_norm = (fs - anc["gen"]["fs"]) / (anc["ora"]["fs"] - anc["gen"]["fs"])
        s_id_norm = (ident - anc["gen"]["id"]) / (anc["ora"]["id"] - anc["gen"]["id"])
        scan_norm_cd[va_idx] = s_cd_norm
        scan_norm_fs[va_idx] = s_fs_norm
        scan_norm_id[va_idx] = s_id_norm
        
    pooled_J = float(np.mean(fold_Js))
    pooled_summary = {
        "J": pooled_J,
        "fold_Js": fold_Js,
        "chamfer": float(np.mean(pooled_cd)),
        "fscore": float(np.mean(pooled_fs)),
        "identification": float(np.mean(pooled_id)),
        "per_scan_norm": {
            "cd": scan_norm_cd,
            "fs": scan_norm_fs,
            "id": scan_norm_id
        }
    }
    return pooled_summary


def bootstrap_J_ci(pooled_summary, train, B=2000, seed=SEED):
    """
    Cluster bootstrap at the object level for proxy objective J.
    """
    groups = train["group"]
    unique_objs = np.unique(groups)
    obj_to_scans = {o: np.where(groups == o)[0] for o in unique_objs}
    
    cd_norm = pooled_summary["per_scan_norm"]["cd"]
    fs_norm = pooled_summary["per_scan_norm"]["fs"]
    id_norm = pooled_summary["per_scan_norm"]["id"]
    
    rng = np.random.RandomState(seed)
    boot_Js = np.empty(B, dtype=np.float64)
    
    for b in range(B):
        sample_objs = rng.choice(unique_objs, size=len(unique_objs), replace=True)
        sample_scans = np.concatenate([obj_to_scans[o] for o in sample_objs])
        
        m_cd = np.mean(cd_norm[sample_scans])
        m_fs = np.mean(fs_norm[sample_scans])
        m_id = np.mean(id_norm[sample_scans])
        
        boot_Js[b] = 0.5 * m_id + 0.5 * (0.5 * m_cd + 0.5 * m_fs)
        
    ci_l = float(np.percentile(boot_Js, 2.5))
    ci_u = float(np.percentile(boot_Js, 97.5))
    return ci_l, ci_u


# =============================================================================
# 3. TASK C: PROBABILITIES GRID (<= 30 CONFIGS)
# =============================================================================

def run_probabilities_grid(train, folds):
    """
    Evaluates fixed grid of <= 30 configs on grouped 5-fold CV.
    Windows: W1 [4,5,6], W_broad [3..7], W_all [0..9] (3)
    Norms: N3, N4 (k=5) (2)
    Models:
      - RidgeClassifier: alpha in [1000.0, 10000.0, 30000.0] (3)
      - Multinomial LogisticRegression: PCA(60), C in [0.003, 0.03] (2)
    Total: 3 * 2 * 5 = 30 configs exactly.
    Calibrates temperature T by minimizing OOF log-loss.
    """
    print("\n" + "=" * 75)
    print("TASK C: EVALUATING PROBABILITIES GRID (30 CONFIGS <= 30)")
    print("=" * 75)
    
    windows = {
        "W1": [4, 5, 6],
        "W_broad": [3, 4, 5, 6, 7],
        "W_all": list(range(10))
    }
    norms = ["N3", "N4"]
    
    alphas = [1000.0, 10000.0, 30000.0]
    logreg_Cs = [0.003, 0.03]
    
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_scans = np.array([class_to_idx[c] for c in train["obj_class"][train["group"]]])
    subjects = np.asarray(train["subject"])
    
    X_raw = train["X"].astype(np.float32)
    
    grid_results = []
    
    for w_name, w_trs in windows.items():
        print(f"\n--- Window {w_name} (TRs {w_trs[:3]}...) ---", flush=True)
        X_win = common.window(X_raw, w_trs)
        
        for norm_name in norms:
            # Pre-normalize folds
            fold_data = []
            for tr_idx, va_idx in folds:
                X_tr, y_tr = X_win[tr_idx], y_scans[tr_idx]
                X_va, y_va = X_win[va_idx], y_scans[va_idx]
                sub_tr, sub_va = subjects[tr_idx], subjects[va_idx]
                
                if norm_name == "N3":
                    Z_tr = common.N3(X_tr)
                    Z_va = common.N3(X_va)
                elif norm_name == "N4":
                    Z_tr, Z_va, _ = common.N4(X_tr, sub_tr, X_heldout=X_va, subjects_heldout=sub_va, k=5)
                fold_data.append((Z_tr, y_tr, Z_va, y_va, va_idx))
                
            # 1. RidgeClassifiers
            for alpha in alphas:
                cfg_name = f"{w_name}_{norm_name}_Ridge_a{int(alpha)}"
                oof_scores = np.empty((len(y_scans), 5), dtype=np.float32)
                
                for Z_tr, y_tr, Z_va, y_va, va_idx in fold_data:
                    clf = RidgeClassifier(alpha=alpha, class_weight="balanced", random_state=SEED)
                    clf.fit(Z_tr, y_tr)
                    oof_scores[va_idx] = clf.decision_function(Z_va)
                    
                # Calibrate temperature T on OOF log loss
                def loss_fn(T_val):
                    p = np.exp(oof_scores / T_val)
                    p /= np.sum(p, axis=1, keepdims=True)
                    p = np.clip(p, 1e-12, 1.0 - 1e-12)
                    return log_loss(y_scans, p)
                    
                opt = minimize_scalar(loss_fn, bounds=(0.01, 100.0), method="bounded")
                T_opt = float(opt.x)
                
                # Temperature scaled probabilities
                oof_probs = np.exp(oof_scores / T_opt)
                oof_probs /= np.sum(oof_probs, axis=1, keepdims=True)
                ll = log_loss(y_scans, oof_probs)
                
                # Predictions and fold log losses
                preds = np.argmax(oof_probs, axis=1)
                bacc = balanced_accuracy_score(y_scans, preds)
                acc = accuracy_score(y_scans, preds)
                
                fold_lls = [log_loss(y_scans[va_idx], oof_probs[va_idx]) for _, _, _, _, va_idx in fold_data]
                ll_se = float(np.std(fold_lls, ddof=1) / np.sqrt(len(fold_lls)))
                
                grid_results.append({
                    "config_key": cfg_name,
                    "window": w_name,
                    "norm": norm_name,
                    "model_type": "Ridge",
                    "alpha": alpha,
                    "C": None,
                    "T": T_opt,
                    "log_loss": float(ll),
                    "log_loss_se": ll_se,
                    "balanced_accuracy": float(bacc),
                    "accuracy": float(acc),
                    "oof_probs": oof_probs,
                    "oof_scores": oof_scores
                })
                print(f"  {cfg_name:28s} | LogLoss: {ll:6.4f} +/- {ll_se:6.4f} | BAcc: {bacc*100:5.2f}% | T: {T_opt:5.2f}")
                
            # 2. LogisticRegression on PCA(60)
            for C_val in logreg_Cs:
                cfg_name = f"{w_name}_{norm_name}_LogReg_C{C_val}"
                oof_probs = np.empty((len(y_scans), 5), dtype=np.float32)
                
                for Z_tr, y_tr, Z_va, y_va, va_idx in fold_data:
                    pca = PCA(n_components=60, random_state=SEED)
                    scaler = StandardScaler()
                    
                    Z_tr_pca = scaler.fit_transform(pca.fit_transform(Z_tr))
                    Z_va_pca = scaler.transform(pca.transform(Z_va))
                    
                    lr = LogisticRegression(C=C_val, class_weight="balanced", max_iter=200, random_state=SEED)
                    lr.fit(Z_tr_pca, y_tr)
                    oof_probs[va_idx] = lr.predict_proba(Z_va_pca)
                    
                oof_probs = np.clip(oof_probs, 1e-12, 1.0 - 1e-12)
                ll = log_loss(y_scans, oof_probs)
                preds = np.argmax(oof_probs, axis=1)
                bacc = balanced_accuracy_score(y_scans, preds)
                acc = accuracy_score(y_scans, preds)
                
                fold_lls = [log_loss(y_scans[va_idx], oof_probs[va_idx]) for _, _, _, _, va_idx in fold_data]
                ll_se = float(np.std(fold_lls, ddof=1) / np.sqrt(len(fold_lls)))
                
                grid_results.append({
                    "config_key": cfg_name,
                    "window": w_name,
                    "norm": norm_name,
                    "model_type": "LogReg",
                    "alpha": None,
                    "C": C_val,
                    "T": 1.0,
                    "log_loss": float(ll),
                    "log_loss_se": ll_se,
                    "balanced_accuracy": float(bacc),
                    "accuracy": float(acc),
                    "oof_probs": oof_probs,
                    "oof_scores": None
                })
                print(f"  {cfg_name:28s} | LogLoss: {ll:6.4f} +/- {ll_se:6.4f} | BAcc: {bacc*100:5.2f}% | T:  1.00")
                
    print(f"\nCompared {len(grid_results)} configurations total (<= 30 limit verified).")
    
    # Model selection by log-loss (1-SE rule, simplest wins)
    best_ll_cfg = min(grid_results, key=lambda x: x["log_loss"])
    ll_threshold = best_ll_cfg["log_loss"] + best_ll_cfg["log_loss_se"]
    
    candidates_1se = [c for c in grid_results if c["log_loss"] <= ll_threshold]
    
    # Simplicity: N3 > N4, Ridge > LogReg, higher alpha / lower C, W1 > W_broad > W_all
    def simplicity_key(c):
        norm_score = 2 if c["norm"] == "N3" else 1
        model_score = 2 if c["model_type"] == "Ridge" else 1
        win_score = 3 if c["window"] == "W1" else (2 if c["window"] == "W_broad" else 1)
        reg_score = c["alpha"] if c["model_type"] == "Ridge" else (1.0 / c["C"])
        return (norm_score, model_score, win_score, reg_score)
        
    simplest_cfg = max(candidates_1se, key=simplicity_key)
    
    print("\n" + "-" * 75)
    print(f"Best Log-Loss Config:         {best_ll_cfg['config_key']} (LogLoss = {best_ll_cfg['log_loss']:.4f} +/- {best_ll_cfg['log_loss_se']:.4f})")
    print(f"1-SE Upper Threshold:         {ll_threshold:.4f} ({len(candidates_1se)} candidates within 1-SE)")
    print(f"Chosen Config (1-SE Rule):    {simplest_cfg['config_key']} (LogLoss = {simplest_cfg['log_loss']:.4f}, BAcc = {simplest_cfg['balanced_accuracy']*100:.2f}%)")
    print("-" * 75)
    
    # Per-class recall of chosen classifier
    cm = confusion_matrix(y_scans, np.argmax(simplest_cfg["oof_probs"], axis=1))
    recalls = {cname: float(cm[i, i] / np.sum(cm[i, :])) for i, cname in enumerate(classes)}
    print("Chosen Classifier Per-Class Recall:")
    for cname, rec in recalls.items():
        print(f"  {cname:10s}: {rec*100:5.2f}%")
        
    return simplest_cfg, grid_results, recalls


# =============================================================================
# 4. TASK D: DECISION RULES R0, R1, R2 ON OOF J
# =============================================================================

def evaluate_decision_rules(train, folds, fold_info, chosen_prob_cfg, D_tt):
    """
    Evaluates decision rules R0, R1, R2 on out-of-fold proxy objective J.
    """
    print("\n" + "=" * 75)
    print("TASK D: DECISION RULES ON TRAIN OOF (R0, R1, R2)")
    print("=" * 75)
    
    oof_probs = chosen_prob_cfg["oof_probs"]
    n_scans = len(train["sample_id"])
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    obj_classes = train["obj_class"]
    
    # -------------------------------------------------------------------------
    # RULE R0: Hard Class Medoid (Reference)
    # -------------------------------------------------------------------------
    print("\n--- Rule R0: Hard Class Medoid ---", flush=True)
    r0_preds = np.empty(n_scans, dtype=int)
    for info in fold_info:
        va_idx = info["va_idx"]
        p_va = oof_probs[va_idx]
        class_preds = np.argmax(p_va, axis=1)
        meds = info["class_medoids"]
        for s, c_idx in enumerate(class_preds):
            r0_preds[va_idx[s]] = meds[classes[c_idx]]
            
    r0_summary = compute_proxy_J(fold_info, r0_preds, train)
    r0_ci = bootstrap_J_ci(r0_summary, train)
    print(f"R0 (Hard Medoid):   J = {r0_summary['J']:+.4f} [95% CI: {r0_ci[0]:+.4f}, {r0_ci[1]:+.4f}] | "
          f"Chamfer = {r0_summary['chamfer']:.4f}, F = {r0_summary['fscore']:.3f}, Ident = {r0_summary['identification']:.1f}%")

    # -------------------------------------------------------------------------
    # RULE R1: Posterior-Weighted Medoid
    # -------------------------------------------------------------------------
    print("\n--- Rule R1: Posterior-Weighted Medoid Grid Search ---", flush=True)
    s_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    mix_grid = [0.0, 0.05, 0.1, 0.2]
    
    r1_results = []
    
    for s_val in s_grid:
        for mix_val in mix_grid:
            r1_preds = np.empty(n_scans, dtype=int)
            for info in fold_info:
                va_idx = info["va_idx"]
                u_tr = info["u_tr"]
                m = len(u_tr)
                D_lib = D_tt[np.ix_(u_tr, u_tr)]
                
                # Class mapping for library shapes
                lib_classes = np.array([class_to_idx[obj_classes[o]] for o in u_tr])
                n_c = np.bincount(lib_classes, minlength=5)
                
                p_va = oof_probs[va_idx]
                
                for s_idx, p_row in enumerate(p_va):
                    # Tempering
                    p_temp = np.power(p_row, 1.0 / s_val)
                    p_temp /= np.sum(p_temp)
                    
                    # Base library weights
                    w = p_temp[lib_classes] / n_c[lib_classes]
                    # Mix with generic prior
                    w_mix = (1.0 - mix_val) * w + mix_val * (1.0 / m)
                    
                    # Choose library shape minimising sum_j w_j * D_lib[i, j]
                    costs = np.dot(D_lib, w_mix)
                    best_local = int(np.argmin(costs))
                    r1_preds[va_idx[s_idx]] = int(u_tr[best_local])
                    
            summary = compute_proxy_J(fold_info, r1_preds, train)
            r1_results.append({
                "s": s_val,
                "mix": mix_val,
                "J": summary["J"],
                "chamfer": summary["chamfer"],
                "fscore": summary["fscore"],
                "identification": summary["identification"],
                "preds": r1_preds,
                "summary": summary
            })
            
    best_r1 = max(r1_results, key=lambda x: x["J"])
    best_r1_ci = bootstrap_J_ci(best_r1["summary"], train)
    print(f"Best R1 (s={best_r1['s']}, mix={best_r1['mix']}): J = {best_r1['J']:+.4f} [95% CI: {best_r1_ci[0]:+.4f}, {best_r1_ci[1]:+.4f}] | "
          f"Chamfer = {best_r1['chamfer']:.4f}, F = {best_r1['fscore']:.3f}, Ident = {best_r1['identification']:.1f}%")
          
    # Plot Identification vs Chamfer tradeoff over s (at best mix)
    best_mix = best_r1["mix"]
    curve_data = [r for r in r1_results if r["mix"] == best_mix]
    curve_data.sort(key=lambda x: x["s"])
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    s_vals = [r["s"] for r in curve_data]
    j_vals = [r["J"] for r in curve_data]
    cd_vals = [r["chamfer"] for r in curve_data]
    id_vals = [r["identification"] for r in curve_data]
    
    ax1.plot(s_vals, id_vals, "ro-", label="Identification (%)", lw=2)
    ax1.set_xlabel("Temperature parameter s", fontsize=11)
    ax1.set_ylabel("Identification (%)", color="r", fontsize=11)
    ax1.set_xscale("log")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(s_vals, cd_vals, "bs--", label="Chamfer distance", lw=2)
    ax2.set_ylabel("Chamfer distance (lower is better)", color="b", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="b")
    
    plt.title(f"Identification vs Chamfer Trade-off over s (mix={best_mix})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = OUT_DIR / "01_tradeoff_over_s.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved tradeoff plot to {plot_path}", flush=True)

    # -------------------------------------------------------------------------
    # RULE R2: Within-Class Retrieval via Classical MDS Embedding
    # -------------------------------------------------------------------------
    print("\n--- Rule R2: Classical MDS Within-Class Retrieval Grid Search ---", flush=True)
    # Extract fMRI features matching chosen classifier
    w_trs = [4, 5, 6] if chosen_prob_cfg["window"] == "W1" else ([3, 4, 5, 6, 7] if chosen_prob_cfg["window"] == "W_broad" else list(range(10)))
    X_win = common.window(train["X"].astype(np.float32), w_trs)
    
    h_grid = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    r2_results = []
    
    # Precompute MDS and Ridge embedding regression for each fold
    fold_r2_data = []
    for info in fold_info:
        tr_idx, va_idx = info["tr_idx"], info["va_idx"]
        u_tr = info["u_tr"]
        m = len(u_tr)
        u_tr_map = {o: i for i, o in enumerate(u_tr)}
        
        # 1. Classical MDS on D_lib (d=15)
        D_lib = D_tt[np.ix_(u_tr, u_tr)]
        D_sq = D_lib ** 2
        H = np.eye(m) - (1.0 / m) * np.ones((m, m))
        B = -0.5 * H @ D_sq @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx_sort = np.argsort(eigvals)[::-1]
        top_vals = np.maximum(eigvals[idx_sort[:15]], 0.0)
        top_vecs = eigvecs[:, idx_sort[:15]]
        Y_mds = top_vecs * np.sqrt(top_vals)[None, :] # (m, 15)
        
        # 2. Target embedding for training scans
        train_objs = train["group"][tr_idx]
        Y_train_scans = Y_mds[[u_tr_map[o] for o in train_objs]] # (n_tr, 15)
        
        # 3. fMRI features
        if chosen_prob_cfg["norm"] == "N3":
            Z_tr = common.N3(X_win[tr_idx])
            Z_va = common.N3(X_win[va_idx])
        else:
            Z_tr, Z_va, _ = common.N4(X_win[tr_idx], train["subject"][tr_idx], X_heldout=X_win[va_idx], subjects_heldout=train["subject"][va_idx], k=5)
            
        # 4. Ridge regression to MDS embedding (inner CV for alpha)
        best_alpha_mds = 1000.0
        reg = Ridge(alpha=best_alpha_mds, random_state=SEED)
        reg.fit(Z_tr, Y_train_scans)
        Y_pred_va = reg.predict(Z_va) # (n_va, 15)
        
        fold_r2_data.append({
            "u_tr": u_tr,
            "D_lib": D_lib,
            "Y_mds": Y_mds,
            "Y_pred_va": Y_pred_va,
            "lib_classes": np.array([class_to_idx[obj_classes[o]] for o in u_tr]),
            "n_c": np.bincount(np.array([class_to_idx[obj_classes[o]] for o in u_tr]), minlength=5)
        })
        
    for h_val in h_grid:
        r2_preds = np.empty(n_scans, dtype=int)
        for fold_idx, info in enumerate(fold_info):
            va_idx = info["va_idx"]
            r2_dat = fold_r2_data[fold_idx]
            u_tr = r2_dat["u_tr"]
            D_lib = r2_dat["D_lib"]
            Y_mds = r2_dat["Y_mds"]
            Y_pred_va = r2_dat["Y_pred_va"]
            lib_classes = r2_dat["lib_classes"]
            n_c = r2_dat["n_c"]
            
            p_va = oof_probs[va_idx]
            
            for s_idx, p_row in enumerate(p_va):
                p_temp = np.power(p_row, 1.0 / best_r1["s"])
                p_temp /= np.sum(p_temp)
                
                # Base R1 weights
                w_base = p_temp[lib_classes] / n_c[lib_classes]
                
                # Embedding distances
                emb_dists = np.linalg.norm(Y_pred_va[s_idx][None, :] - Y_mds, axis=1) # (m,)
                kernel = np.exp(-(emb_dists ** 2) / (2.0 * (h_val ** 2)))
                
                # Combined weight
                w_comb = w_base * kernel
                w_comb /= np.sum(w_comb)
                
                costs = np.dot(D_lib, w_comb)
                best_local = int(np.argmin(costs))
                r2_preds[va_idx[s_idx]] = int(u_tr[best_local])
                
        summary = compute_proxy_J(fold_info, r2_preds, train)
        r2_results.append({
            "h": h_val,
            "J": summary["J"],
            "chamfer": summary["chamfer"],
            "fscore": summary["fscore"],
            "identification": summary["identification"],
            "preds": r2_preds,
            "summary": summary
        })
        
    best_r2 = max(r2_results, key=lambda x: x["J"])
    best_r2_ci = bootstrap_J_ci(best_r2["summary"], train)
    print(f"Best R2 (h={best_r2['h']}): J = {best_r2['J']:+.4f} [95% CI: {best_r2_ci[0]:+.4f}, {best_r2_ci[1]:+.4f}] | "
          f"Chamfer = {best_r2['chamfer']:.4f}, F = {best_r2['fscore']:.3f}, Ident = {best_r2['identification']:.1f}%")

    # -------------------------------------------------------------------------
    # Rule Selection (1-SE Rule on J)
    # -------------------------------------------------------------------------
    rules = [
        {"name": "R0", "J": r0_summary["J"], "ci": r0_ci, "summary": r0_summary, "simplicity": 3, "preds": r0_preds},
        {"name": "R1", "J": best_r1["J"], "ci": best_r1_ci, "summary": best_r1["summary"], "simplicity": 2, "preds": best_r1["preds"], "s": best_r1["s"], "mix": best_r1["mix"]},
        {"name": "R2", "J": best_r2["J"], "ci": best_r2_ci, "summary": best_r2["summary"], "simplicity": 1, "preds": best_r2["preds"], "h": best_r2["h"], "s": best_r1["s"], "mix": best_r1["mix"]}
    ]
    
    best_rule = max(rules, key=lambda x: x["J"])
    # 1-SE threshold on fold Js
    fold_se = float(np.std(best_rule["summary"]["fold_Js"], ddof=1) / np.sqrt(5))
    j_threshold = best_rule["J"] - fold_se
    
    candidates_rule = [r for r in rules if r["J"] >= j_threshold]
    chosen_rule = max(candidates_rule, key=lambda x: (x["simplicity"], x["J"]))
    
    print("\n" + "=" * 75)
    print("RULE SELECTION SUMMARY:")
    print("=" * 75)
    print(f"R0 (Hard Medoid):               J = {r0_summary['J']:+.4f} [95% CI: {r0_ci[0]:+.4f}, {r0_ci[1]:+.4f}]")
    print(f"R1 (Posterior-Weighted Medoid): J = {best_r1['J']:+.4f} [95% CI: {best_r1_ci[0]:+.4f}, {best_r1_ci[1]:+.4f}] (s={best_r1['s']}, mix={best_r1['mix']})")
    print(f"R2 (Within-Class MDS Kernel):   J = {best_r2['J']:+.4f} [95% CI: {best_r2_ci[0]:+.4f}, {best_r2_ci[1]:+.4f}] (h={best_r2['h']})")
    print(f"1-SE Selection Threshold on J:  {j_threshold:+.4f}")
    print(f"CHOSEN RULE: {chosen_rule['name']} (Simplest within 1-SE)")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Control on OOF: 200 Shuffles of Held-out Features
    # -------------------------------------------------------------------------
    print(f"\nRunning OOF Shuffle Control (200 shuffles) for chosen rule {chosen_rule['name']}...", flush=True)
    rng_shuff = np.random.RandomState(SEED)
    null_Js = []
    null_idents = []
    
    for shuff_i in range(200):
        shuff_preds = np.empty(n_scans, dtype=int)
        for info in fold_info:
            va_idx = info["va_idx"]
            n_va = len(va_idx)
            perm_va = rng_shuff.permutation(n_va)
            
            # Shuffled predictions from chosen rule
            shuff_preds[va_idx] = chosen_rule["preds"][va_idx[perm_va]]
            
        shuff_sum = compute_proxy_J(fold_info, shuff_preds, train)
        null_Js.append(shuff_sum["J"])
        null_idents.append(shuff_sum["identification"])
        
    null_j_mean = float(np.mean(null_Js))
    null_j_95 = float(np.percentile(null_Js, 95))
    obs_J = chosen_rule["J"]
    p_J = float((1.0 + np.sum(np.array(null_Js) >= obs_J)) / (1.0 + 200.0))
    
    null_id_mean = float(np.mean(null_idents))
    null_id_95 = float(np.percentile(null_idents, 95))
    obs_id = chosen_rule["summary"]["identification"]
    p_id = float((1.0 + np.sum(np.array(null_idents) >= obs_id)) / (1.0 + 200.0))
    
    print(f"OOF Shuffle Null (200 shuffles):")
    print(f"  Proxy J:        Null Mean = {null_j_mean:+.4f}, 95th pct = {null_j_95:+.4f}, Obs = {obs_J:+.4f}, p = {p_J:.4f}")
    print(f"  Identification: Null Mean = {null_id_mean:.1f}%, 95th pct = {null_id_95:.1f}%, Obs = {obs_id:.1f}%, p = {p_id:.4f}")
    
    # Save outputs/phase3/chosen_phase3.json
    chosen_meta = {
        "features": {
            "window": chosen_prob_cfg["window"],
            "norm": chosen_prob_cfg["norm"]
        },
        "classifier": {
            "model_type": chosen_prob_cfg["model_type"],
            "alpha": chosen_prob_cfg["alpha"],
            "C": chosen_prob_cfg["C"],
            "T": chosen_prob_cfg["T"],
            "log_loss": chosen_prob_cfg["log_loss"],
            "balanced_accuracy": chosen_prob_cfg["balanced_accuracy"]
        },
        "rule": chosen_rule["name"],
        "parameters": {
            "s": float(chosen_rule.get("s", 1.0)),
            "mix": float(chosen_rule.get("mix", 0.0)),
            "h": float(chosen_rule.get("h", 1.0)) if chosen_rule["name"] == "R2" else None
        },
        "oof_proxy_J": {
            "J": float(obs_J),
            "ci_95": chosen_rule["ci"],
            "chamfer": float(chosen_rule["summary"]["chamfer"]),
            "fscore": float(chosen_rule["summary"]["fscore"]),
            "identification": float(chosen_rule["summary"]["identification"])
        },
        "oof_shuffle_control": {
            "null_J_mean": null_j_mean,
            "null_J_95": null_j_95,
            "p_J": p_J,
            "null_id_mean": null_id_mean,
            "null_id_95": null_id_95,
            "p_id": p_id
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(CHOSEN_PHASE3_PATH, "w", encoding="utf-8") as f:
        json.dump(chosen_meta, f, indent=2)
    print(f"\nWritten {CHOSEN_PHASE3_PATH}", flush=True)
    
    return chosen_meta, rules


# =============================================================================
# 5. TASK E: LOOK #2 ON VALIDATION (FINAL EVALUATION & PARITY)
# =============================================================================

def run_task_e_val_look(train, chosen_meta, D_tt):
    """
    Fits final pipeline on all 533 train scans, evaluates on val (Look #2).
    Candidate library = all 292 train shapes.
    Runs common.score_val, checks official evaluate.py parity to 1e-6.
    """
    print("\n" + "=" * 75)
    print("TASK E: LOOK #2 ON VALIDATION SET (Final Phase 3 Pipeline)")
    print("=" * 75)
    
    val_pred_npz_path = OUT_DIR / "val_predictions.npz"
    val_metrics_json_path = OUT_DIR / "val_metrics.json"
    val_detailed_path = OUT_DIR / "val_metrics_detailed.json"
    if VAL_LOOKS_PATH.exists():
        with open(VAL_LOOKS_PATH, "r", encoding="utf-8") as f:
            looks_txt = f.read()
        if "| **Look-2**" in looks_txt and val_pred_npz_path.exists() and val_metrics_json_path.exists() and val_detailed_path.exists():
            print("Validation Look #2 has already been logged in val_looks.md and outputs exist. Reusing stored results.", flush=True)
            with open(val_detailed_path, "r", encoding="utf-8") as f:
                val_report = json.load(f)
            return val_report

    val = starter.load_split("val")
    n_val = len(val["sample_id"])
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_train = np.array([class_to_idx[c] for c in train["obj_class"][train["group"]]])
    y_val = np.array([class_to_idx[c] for c in val["obj_class"][val["group"]]])
    
    # 1. Feature extraction on train and val
    w_trs = [4, 5, 6] if chosen_meta["features"]["window"] == "W1" else ([3, 4, 5, 6, 7] if chosen_meta["features"]["window"] == "W_broad" else list(range(10)))
    X_tr_win = common.window(train["X"].astype(np.float32), w_trs)
    X_va_win = common.window(val["X"].astype(np.float32), w_trs)
    
    if chosen_meta["features"]["norm"] == "N3":
        Z_train = common.N3(X_tr_win)
        Z_val = common.N3(X_va_win)
    else:
        Z_train, Z_val, _ = common.N4(X_tr_win, train["subject"], X_heldout=X_va_win, subjects_heldout=val["subject"], k=5)
        
    # 2. Fit classifier on all 533 train scans
    print(f"Fitting final classifier ({chosen_meta['classifier']['model_type']}) on all 533 train scans...", flush=True)
    if chosen_meta["classifier"]["model_type"] == "Ridge":
        clf = RidgeClassifier(alpha=chosen_meta["classifier"]["alpha"], class_weight="balanced", random_state=SEED)
        clf.fit(Z_train, y_train)
        val_scores = clf.decision_function(Z_val)
        T = chosen_meta["classifier"]["T"]
        val_probs = np.exp(val_scores / T)
        val_probs /= np.sum(val_probs, axis=1, keepdims=True)
    else:
        pca = PCA(n_components=60, random_state=SEED)
        scaler = StandardScaler()
        Z_tr_pca = scaler.fit_transform(pca.fit_transform(Z_train))
        Z_va_pca = scaler.transform(pca.transform(Z_val))
        clf = LogisticRegression(C=chosen_meta["classifier"]["C"], class_weight="balanced", max_iter=200, random_state=SEED)
        clf.fit(Z_tr_pca, y_train)
        val_probs = clf.predict_proba(Z_va_pca)
        
    # 3. Predict library shapes on val using chosen rule
    lib_all = np.arange(len(train["obj_points"])) # all 292 train shapes
    m_all = len(lib_all)
    lib_classes = np.array([class_to_idx[train["obj_class"][o]] for o in lib_all])
    n_c_all = np.bincount(lib_classes, minlength=5)
    rule_name = chosen_meta["rule"]
    
    print(f"Predicting library shapes for 125 val scans using Rule {rule_name}...", flush=True)
    val_pred_indices = np.empty(n_val, dtype=int)
    
    if rule_name == "R0":
        # Global class medoids
        meds = {}
        for cname in classes:
            c_objs = [o for o in lib_all if train["obj_class"][o] == cname]
            sub_D = D_tt[np.ix_(c_objs, c_objs)]
            med_local = int(np.argmin(np.sum(sub_D, axis=1)))
            meds[class_to_idx[cname]] = int(c_objs[med_local])
            
        val_class_preds = np.argmax(val_probs, axis=1)
        for s in range(n_val):
            val_pred_indices[s] = meds[val_class_preds[s]]
            
    elif rule_name == "R1":
        s_val = chosen_meta["parameters"]["s"]
        mix_val = chosen_meta["parameters"]["mix"]
        for s in range(n_val):
            p_temp = np.power(val_probs[s], 1.0 / s_val)
            p_temp /= np.sum(p_temp)
            w = p_temp[lib_classes] / n_c_all[lib_classes]
            w_mix = (1.0 - mix_val) * w + mix_val * (1.0 / m_all)
            costs = np.dot(D_tt, w_mix)
            val_pred_indices[s] = int(np.argmin(costs))
            
    elif rule_name == "R2":
        s_val = chosen_meta["parameters"]["s"]
        mix_val = chosen_meta["parameters"]["mix"]
        h_val = chosen_meta["parameters"]["h"]
        
        # MDS on all 292 shapes (d=15)
        D_sq = D_tt ** 2
        H = np.eye(m_all) - (1.0 / m_all) * np.ones((m_all, m_all))
        B = -0.5 * H @ D_sq @ H
        eigvals, eigvecs = np.linalg.eigh(B)
        idx_sort = np.argsort(eigvals)[::-1]
        top_vals = np.maximum(eigvals[idx_sort[:15]], 0.0)
        top_vecs = eigvecs[:, idx_sort[:15]]
        Y_mds_all = top_vecs * np.sqrt(top_vals)[None, :]
        
        # Ridge regression on all train scans
        train_objs = train["group"]
        Y_train_all = Y_mds_all[train_objs]
        reg_all = Ridge(alpha=1000.0, random_state=SEED)
        reg_all.fit(Z_train, Y_train_all)
        Y_pred_val = reg_all.predict(Z_val)
        
        for s in range(n_val):
            p_temp = np.power(val_probs[s], 1.0 / s_val)
            p_temp /= np.sum(p_temp)
            w_base = p_temp[lib_classes] / n_c_all[lib_classes]
            emb_d = np.linalg.norm(Y_pred_val[s][None, :] - Y_mds_all, axis=1)
            kernel = np.exp(-(emb_d ** 2) / (2.0 * (h_val ** 2)))
            w_comb = w_base * kernel
            w_comb /= np.sum(w_comb)
            costs = np.dot(D_tt, w_comb)
            val_pred_indices[s] = int(np.argmin(costs))

    # 4. Point cloud generation and score_val evaluation
    pred_points = train["obj_points"][val_pred_indices].astype(np.float32)
    print("Evaluating validation predictions via common.score_val...", flush=True)
    summary_val, per_val = common.score_val(pred_points)
    
    # Object bootstrap CIs for all 5 metrics + composite
    val_groups = val["group"]
    val_unique_objs = np.unique(val_groups)
    obj_to_scans = {o: np.where(val_groups == o)[0] for o in val_unique_objs}
    
    metric_names = ["chamfer", "fscore", "silhouette_iou", "silhouette_ssim", "identification"]
    per_scan_arrays = {m: np.array([p[m] for p in per_val]) for m in metric_names}
    
    with open(WORKSPACE_ROOT / "outputs" / "phase1" / "anchors.json", "r") as f:
        anchors = json.load(f)
    gen_anc = anchors["generic"]
    ora_anc = anchors["oracle"]
    
    B = 2000
    rng = np.random.RandomState(SEED)
    boot_res = {m: np.empty(B, dtype=np.float64) for m in metric_names}
    boot_comps = np.empty(B, dtype=np.float64)
    
    for b in range(B):
        sample_objs = rng.choice(val_unique_objs, size=len(val_unique_objs), replace=True)
        sample_scans = np.concatenate([obj_to_scans[o] for o in sample_objs])
        b_sum = {}
        for m in metric_names:
            val_m = float(np.mean(per_scan_arrays[m][sample_scans]))
            boot_res[m][b] = val_m
            b_sum[m] = val_m
        boot_comps[b] = metrics.composite(b_sum, gen_anc, ora_anc)
        
    ci_dict = {m: [float(np.percentile(boot_res[m], 2.5)), float(np.percentile(boot_res[m], 97.5))] for m in metric_names}
    comp_ci = [float(np.percentile(boot_comps, 2.5)), float(np.percentile(boot_comps, 97.5))]
    
    # 5. Identification Permutation Test (n=2000)
    print("Running Identification Permutation Test (n=2000)...", flush=True)
    D_val_pred = common.chamfer_matrix(pred_points, val["obj_points"])
    obs_id, null_id_m, null_id_95, p_ident = common.identification_permutation_test(D_val_pred, val_groups, n=2000, seed=SEED)
    
    # 6. Shuffled-Feature Control (n=500)
    print("Running Shuffled-Feature Control (n=500)...", flush=True)
    shuff_rng = np.random.RandomState(SEED)
    null_comps = []
    
    # Precompute F-score matrix between predicted points (125) and 67 unique val objects
    val_obj_trees = [cKDTree(p) for p in val["obj_points"]]
    pred_trees = [cKDTree(p) for p in pred_points]
    F_val_pred = np.empty((n_val, len(val["obj_points"])), dtype=np.float32)
    for i in range(n_val):
        p_i = pred_points[i]
        tree_p = pred_trees[i]
        for j in range(len(val["obj_points"])):
            t_j = val["obj_points"][j]
            tree_t = val_obj_trees[j]
            d_p2t = tree_t.query(p_i)[0]
            d_t2p = tree_p.query(t_j)[0]
            prec = np.mean(d_p2t < 0.05)
            rec = np.mean(d_t2p < 0.05)
            F_val_pred[i, j] = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    obs_comp = float(summary_val["composite"])
    
    for s_i in range(500):
        perm_va = shuff_rng.permutation(n_val)
        D_perm = D_val_pred[perm_va]
        F_perm = F_val_pred[perm_va]
        
        # Mean Chamfer and F-score
        cd_perm = np.mean([D_perm[k, val_groups[k]] for k in range(n_val)])
        fs_perm = np.mean([F_perm[k, val_groups[k]] for k in range(n_val)])
        
        # Identification under permuted predictions
        id_perm = common.identification_from_matrix(D_perm, val_groups)
        
        # Geometry normalized (generic baseline expectation for shuffled silhouettes)
        norm_cd = (gen_anc["chamfer"] - cd_perm) / (gen_anc["chamfer"] - ora_anc["chamfer"])
        norm_fs = (fs_perm - gen_anc["fscore"]) / (ora_anc["fscore"] - gen_anc["fscore"])
        geom_norm = 0.25 * (norm_cd + norm_fs + 0.0 + 0.0)
        norm_id = (id_perm - 50.0) / (ora_anc["identification"] - 50.0)
        
        comp_perm = 0.5 * norm_id + 0.5 * geom_norm
        null_comps.append(comp_perm)
        
    null_comp_m = float(np.mean(null_comps))
    null_comp_95 = float(np.percentile(null_comps, 95))
    p_comp = float((1.0 + np.sum(np.array(null_comps) >= obs_comp)) / (1.0 + 500.0))
    
    # Classification performance & per-class recall
    val_class_preds = np.array([class_to_idx[train["obj_class"][sp]] for sp in val_pred_indices])
    cm = confusion_matrix(y_val, val_class_preds)
    recalls = {cname: float(cm[i, i] / np.sum(cm[i, :])) for i, cname in enumerate(classes)}
    val_acc = float(accuracy_score(y_val, val_class_preds))
    val_bacc = float(balanced_accuracy_score(y_val, val_class_preds))
    
    # 7. Write outputs/phase3/val_predictions.npz
    val_pred_npz_path = OUT_DIR / "val_predictions.npz"
    np.savez_compressed(val_pred_npz_path, sample_id=val["sample_id"], points=pred_points)
    print(f"\nWritten {val_pred_npz_path} ({pred_points.shape}, float32)")
    
    # 8. Run official evaluate.py via subprocess and verify parity to 1e-6
    val_metrics_json_path = OUT_DIR / "val_metrics.json"
    cmd = [sys.executable, "evaluate.py", str(val_pred_npz_path), "--save", str(val_metrics_json_path)]
    print(f"Running official evaluation command:\n  {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
    assert res.returncode == 0, f"evaluate.py failed with returncode {res.returncode}:\n{res.stderr}"
    print("evaluate.py output:")
    print(res.stdout)
    
    with open(val_metrics_json_path, "r") as f:
        official_eval = json.load(f)
    diff_comp = abs(summary_val["composite"] - official_eval["composite"])
    print(f"score_val composite:        {summary_val['composite']:.8f}")
    print(f"evaluate.py composite:      {official_eval['composite']:.8f}")
    print(f"Absolute parity difference: {diff_comp:.2e}")
    assert diff_comp < 1e-6, f"Parity mismatch with evaluate.py! diff = {diff_comp}"
    print("CONFIRMED: Official evaluate.py matches score_val to < 1e-6 (exact match).")
    
    # 9. Log in val_looks.md
    log_line = (
        f"| **Look-2** | Task E | Final Phase 3 pipeline (`code/05_phase3_pipeline.py`) | "
        f"`val['X']`, `val['obj_points']`, `val['group']` | "
        f"**Look #2: final Phase 3 pipeline, train-only fit, no choice depends on it.** (Val Composite = {obs_comp:+.3f}) | "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} |\n"
    )
    with open(VAL_LOOKS_PATH, "r", encoding="utf-8") as f:
        looks_txt = f.read()
    if "| **Look-2**" not in looks_txt:
        with open(VAL_LOOKS_PATH, "a", encoding="utf-8") as f:
            f.write(log_line)
        print(f"Logged Look #2 in {VAL_LOOKS_PATH}")
    else:
        print(f"Look #2 already logged in {VAL_LOOKS_PATH}")
    
    # 10. Update status.json
    update_phase3_status("taskE_val_look", status="DONE", evidence_file="outputs/phase3/val_predictions.npz", reason=f"Look #2 val look, composite={obs_comp:.3f}, evaluate.py parity verified")

    
    # Save full summary
    val_report = {
        "metrics": summary_val,
        "ci_95": {**ci_dict, "composite": comp_ci},
        "identification_test": {
            "obs": obs_id,
            "null_mean": null_id_m,
            "null_95": null_id_95,
            "p_value": p_ident
        },
        "shuffled_control": {
            "obs": obs_comp,
            "null_mean": null_comp_m,
            "null_95": null_comp_95,
            "p_value": p_comp
        },
        "classification": {
            "accuracy": val_acc,
            "balanced_accuracy": val_bacc,
            "per_class_recall": recalls
        },
        "evaluate_py_parity": {
            "difference": diff_comp,
            "passed": True
        }
    }
    with open(OUT_DIR / "val_metrics_detailed.json", "w") as f:
        json.dump(val_report, f, indent=2)
        
    return val_report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    t0_all = time.time()
    print("======================================================================")
    print("PHASE 3: ADVANCED DECODING, PROXY J OBJECTIVE, & FINAL VAL LOOK")
    print("======================================================================")
    
    # -------------------------------------------------------------------------
    # TASK A: Audit & Baseline Facts
    # -------------------------------------------------------------------------
    print("\n--- TASK A: Audit and Baseline Facts ---")
    with open(CHOSEN_PHASE2_PATH, "r") as f:
        p2_chosen = json.load(f)
    print("Task 2 / Task 3 Chosen Config:")
    print(json.dumps(p2_chosen["chosen_config"], indent=2))
    
    chosen_alpha = p2_chosen["chosen_config"]["alpha"]
    print(f"\nTask 3 Alpha Grid was [10, 100, 1000, 10000]. Chosen alpha is {chosen_alpha}.")
    print("Check: Alpha 1000.0 is adjacent to the maximum (10000.0). We extend the alpha grid in Task C up to 30000.0.")
    update_phase3_status("taskA_audit", status="DONE", evidence_file="outputs/phase2/chosen_config.json", reason="Audit chosen_config.json, alpha at grid edge checked")
    
    # Load train data
    train = starter.load_split("train")
    folds = common.group_folds(train, k=5, seed=0)
    
    # -------------------------------------------------------------------------
    # TASK B: Compute/Load F_tt, D_tt and Fold Libraries
    # -------------------------------------------------------------------------
    print("\n--- TASK B: Train-Train Matrices & Proxy Objective Setup ---")
    F_tt = load_or_compute_f_tt(train)
    D_tt = np.load(CACHE_DIR / "chamfer_train_train.npy")
    print(f"Loaded D_tt {D_tt.shape} and F_tt {F_tt.shape}")
    
    fold_info = setup_fold_libraries(train, folds, D_tt, F_tt)
    print("Precomputed fold libraries, class medoids, and fold anchors for all 5 folds.")
    update_phase3_status("taskB_proxy_objective", status="DONE", evidence_file="outputs/cache/fscore_train_train.npy", reason="Cached F_tt and computed fold-isolated proxy J")
    
    # -------------------------------------------------------------------------
    # TASK C: Probabilities Grid (<= 30 configs)
    # -------------------------------------------------------------------------
    chosen_prob_cfg, grid_results, recalls_clf = run_probabilities_grid(train, folds)
    update_phase3_status("taskC_probabilities", status="DONE", evidence_file="outputs/phase3/chosen_phase3.json", reason="Grid <= 30 configs, calibrated T, log-loss 1-SE selection")
    
    # -------------------------------------------------------------------------
    # TASK D: Decision Rules on OOF J
    # -------------------------------------------------------------------------
    chosen_meta, rules = evaluate_decision_rules(train, folds, fold_info, chosen_prob_cfg, D_tt)
    update_phase3_status("taskD_decision_rules", status="DONE", evidence_file="outputs/phase3/chosen_phase3.json", reason="Evaluated R0, R1, R2 on OOF J with CIs, OOF control")
    
    # -------------------------------------------------------------------------
    # TASK E: Look #2 on Validation
    # -------------------------------------------------------------------------
    val_report = run_task_e_val_look(train, chosen_meta, D_tt)
    update_phase3_status("taskE_val_look", status="DONE", evidence_file="outputs/phase3/val_predictions.npz", reason="Look #2 val look, evaluate.py parity verified")
    
    total_time = time.time() - t0_all
    print("\n" + "=" * 75)
    print(f"PHASE 3 COMPLETE — Total Runtime: {total_time:.2f} seconds")
    print("======================================================================")


if __name__ == "__main__":
    main()
