"""
Phase 2 - Task 0: Phase 1 Follow-ups
Follows RULES.md strictly. CPU-only, fixed seeds.
"""
from pathlib import Path
import sys
import json
import time
import concurrent.futures

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

CODE_DIR = WORKSPACE_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import scipy.stats as stats
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import starter
import common

SEED = 42
OUT_DIR = WORKSPACE_ROOT / "outputs" / "phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cramers_v_both(contingency_table):
    """
    Computes both uncorrected and bias-corrected (Bergsma 2013) Cramer's V.
    """
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = np.sum(contingency_table)
    r, c = contingency_table.shape
    
    # Uncorrected
    v_uncorr = np.sqrt((chi2 / n) / min(r - 1, c - 1))
    
    # Bias-corrected
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    c_corr = c - ((c - 1) ** 2) / (n - 1)
    denom = min(r_corr - 1, c_corr - 1)
    if denom <= 0:
        v_corr = 0.0
    else:
        v_corr = np.sqrt(phi2_corr / denom)
        
    return float(chi2), float(v_uncorr), float(v_corr)


def compute_eta_omega_squared(y, group_labels):
    """
    Computes eta-squared and omega-squared for 1D variable y across group_labels.
    """
    unique_groups = np.unique(group_labels)
    k = len(unique_groups)
    n = len(y)
    
    grand_mean = np.mean(y)
    ss_total = np.sum((y - grand_mean) ** 2)
    
    ss_between = 0.0
    for g in unique_groups:
        idx = np.where(group_labels == g)[0]
        n_g = len(idx)
        g_mean = np.mean(y[idx])
        ss_between += n_g * ((g_mean - grand_mean) ** 2)
        
    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = n - k
    ms_within = ss_within / df_within
    
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    omega2 = (ss_between - df_between * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0.0
    omega2 = max(0.0, omega2) # bounded at 0
    return float(eta2), float(omega2)


def vectorised_anova_f(X, class_indices, n_classes=5):
    """
    Vectorized one-way ANOVA F-statistic across columns of X.
    """
    N, V = X.shape
    grand_mean = X.mean(axis=0)
    SS_between = np.zeros(V, dtype=np.float32)
    SS_within = np.zeros(V, dtype=np.float32)
    for c in range(n_classes):
        mask = (class_indices == c)
        n_c = np.sum(mask)
        if n_c > 0:
            c_mean = X[mask].mean(axis=0)
            SS_between += n_c * ((c_mean - grand_mean) ** 2)
            SS_within += np.sum((X[mask] - c_mean) ** 2, axis=0)
    df_b = n_classes - 1
    df_w = N - n_classes
    return (SS_between / df_b) / (SS_within / df_w)


def main():
    print("======================================================================")
    print("PHASE 2 - TASK 0: PHASE 1 FOLLOW-UPS")
    print("======================================================================")
    
    train, val, test, cams, voxels = common.load_all()
    
    # -------------------------------------------------------------------------
    # 0.1 RAW SUBJECT X CLASS SCAN TABLES
    # -------------------------------------------------------------------------
    print("\n--- 0.1 Subject x Class Scan Contingency Tables ---")
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cramers_results = {}
    
    for split_name, d in [("Train", train), ("Val", val)]:
        subjs = sorted(list(np.unique(d["subject"])))
        scan_classes = d["obj_class"][d["group"]]
        
        table = np.zeros((len(subjs), len(classes)), dtype=int)
        for i, s in enumerate(subjs):
            s_mask = (d["subject"] == s)
            for j, c in enumerate(classes):
                table[i, j] = np.sum(s_mask & (scan_classes == c))
                
        chi2, v_uncorr, v_corr = cramers_v_both(table)
        cramers_results[split_name.lower()] = {
            "chi2": chi2,
            "uncorrected": v_uncorr,
            "bias_corrected": v_corr,
            "table": table.tolist(),
            "subjects": subjs,
            "classes": classes
        }
        
        print(f"\n{split_name} Split Subject x Class Scan Table (N={len(d['X'])}):")
        header = f"{'Subject':<10}" + "".join([f"{c:>10}" for c in classes]) + f"{'Total':>8}"
        print(header)
        print("-" * len(header))
        missing_classes = {}
        for i, s in enumerate(subjs):
            row_str = f"{s:<10}" + "".join([f"{table[i, j]:>10}" for j in range(len(classes))]) + f"{table[i].sum():>8}"
            print(row_str)
            missing = [classes[j] for j in range(len(classes)) if table[i, j] == 0]
            if missing:
                missing_classes[s] = missing
                
        cramers_results[split_name.lower()]["missing_classes"] = missing_classes
        print(f"\n{split_name} Association Statistics:")
        print(f"  Chi-squared: {chi2:.2f}")
        print(f"  Cramer's V (uncorrected):      {v_uncorr:.4f}")
        print(f"  Cramer's V (bias-corrected):   {v_corr:.4f}")
        print(f"  Subjects missing any class: {len(missing_classes)}")
        for s, miss in missing_classes.items():
            print(f"    - {s}: missing {miss} (total scans = {table[subjs.index(s)].sum()})")

    # -------------------------------------------------------------------------
    # 0.2 CLASS ETA^2 AND OMEGA^2 FOR TOP 10 PCs UNDER N1, N2, N3
    # -------------------------------------------------------------------------
    print("\n--- 0.2 Class Effect Sizes (eta^2 & omega^2) for Top 10 PCs ---")
    X_tr_avg = train["X"].astype(np.float32).mean(axis=1) # (533, 31141)
    train_scan_classes = train["obj_class"][train["group"]]
    
    # N1: Global voxel z-score
    m_glob = X_tr_avg.mean(axis=0)
    s_glob = X_tr_avg.std(axis=0)
    s_safe = np.where(s_glob < 1e-6, 1.0, s_glob)
    N1_tr = np.where(s_glob < 1e-6, 0.0, (X_tr_avg - m_glob) / s_safe)
    
    # N2: Per-subject z-score
    N2_tr, _, _ = common.N2w(X_tr_avg, train["subject"], fallback_n4_stats=None, min_scans=1)
    
    # N3: Per-scan z-score
    N3_tr = common.N3(X_tr_avg)
    
    pca_results = {}
    for norm_name, Z_data in [("N1 (Global)", N1_tr), ("N2 (Per-Subject)", N2_tr), ("N3 (Per-Scan)", N3_tr)]:
        pca = PCA(n_components=10, random_state=SEED)
        pcs = pca.fit_transform(Z_data)
        
        eta2_list = []
        omega2_list = []
        for pc_idx in range(10):
            e2, w2 = compute_eta_omega_squared(pcs[:, pc_idx], train_scan_classes)
            eta2_list.append(e2)
            omega2_list.append(w2)
            
        pca_results[norm_name] = {
            "var_explained": [float(v) for v in pca.explained_variance_ratio_],
            "eta2": eta2_list,
            "omega2": omega2_list,
        }
        
        print(f"\n{norm_name} Normalization:")
        print(f"{'PC':<6}{'Var Explained':<16}{'Class eta^2':<14}{'Class omega^2':<14}")
        print("-" * 50)
        for pc_idx in range(10):
            print(f"PC{pc_idx+1:<3}{pca.explained_variance_ratio_[pc_idx]*100:>13.2f}%{eta2_list[pc_idx]:>13.4f}{omega2_list[pc_idx]:>14.4f}")
        print(f"Mean top-10 PC:      {np.mean(eta2_list):>13.4f}{np.mean(omega2_list):>14.4f}")

    # -------------------------------------------------------------------------
    # 0.3 TIME INDEX NOTATION CLARIFICATION
    # -------------------------------------------------------------------------
    print("\n--- 0.3 Time Index Notation Clarification ---")
    print("In Phase 1, 'TR 4-6' referred to 0-based time indices 4, 5, 6 (i.e. the 5th, 6th, and 7th time points).")
    print("Going forward, all scripts strictly use 0-based indexing: time indices 0..9.")

    # -------------------------------------------------------------------------
    # 0.4 1000-PERMUTATION PER-TIME-POINT ANOVA
    # -------------------------------------------------------------------------
    print("\n--- 0.4 Per-Time-Point ANOVA Permutation Test (N=1000 Object-Level Permutations) ---")
    n_train_objs = len(train["obj_points"])
    train_obj_groups = train["group"]
    train_scan_cls_idx = np.array([class_to_idx[c] for c in train_scan_classes])
    
    obj_to_class = np.zeros(n_train_objs, dtype=int)
    for g_id in range(n_train_objs):
        scans_of_g = np.where(train_obj_groups == g_id)[0]
        obj_to_class[g_id] = train_scan_cls_idx[scans_of_g[0]]
        
    def eval_time_point_1000(t_idx):
        t_start_tp = time.time()
        np.random.seed(SEED + t_idx * 100)
        X_t = train["X"][:, t_idx, :].astype(np.float32)
        Z_t, _, _ = common.N2w(X_t, train["subject"], min_scans=1)
        N_s, V = Z_t.shape
        df_b = 4
        df_w = N_s - 5
        
        grand_mean = Z_t.mean(axis=0)
        SS_total = np.sum((Z_t - grand_mean) ** 2, axis=0)
        
        # Observed ANOVA F
        SS_between_obs = np.zeros(V, dtype=np.float32)
        for c in range(5):
            mask = (train_scan_cls_idx == c)
            n_c = int(np.sum(mask))
            if n_c > 0:
                c_mean = Z_t[mask].mean(axis=0)
                SS_between_obs += n_c * ((c_mean - grand_mean) ** 2)
        SS_within_obs = SS_total - SS_between_obs
        f_obs_t = (SS_between_obs / df_b) / (SS_within_obs / df_w)
        obs_top1000 = float(np.partition(f_obs_t, -1000)[-1000:].mean())
        
        null_top = np.empty(1000, dtype=np.float32)
        for perm_i in range(1000):
            perm_obj_classes = np.random.permutation(obj_to_class)
            perm_scan_classes = perm_obj_classes[train_obj_groups]
            SS_between = np.zeros(V, dtype=np.float32)
            for c in range(5):
                mask = (perm_scan_classes == c)
                n_c = int(np.sum(mask))
                if n_c > 0:
                    c_mean = Z_t[mask].mean(axis=0)
                    SS_between += n_c * ((c_mean - grand_mean) ** 2)
            SS_within = SS_total - SS_between
            f_null = (SS_between / df_b) / (SS_within / df_w)
            null_top[perm_i] = float(np.partition(f_null, -1000)[-1000:].mean())
            
        null_mean = float(np.mean(null_top))
        null_p95 = float(np.percentile(null_top, 95))
        p_val = float((1 + np.sum(null_top >= obs_top1000)) / (1 + 1000))
        p_bonf = float(min(1.0, p_val * 10))
        elapsed = time.time() - t_start_tp
        return t_idx, obs_top1000, null_mean, null_p95, p_val, p_bonf, elapsed

    print("Running 1000 permutations across all 10 time points in parallel (max 5 threads)...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        tp_results = list(ex.map(eval_time_point_1000, range(10)))
        
    tp_results.sort(key=lambda x: x[0])
    
    print("\nPer-Time-Point ANOVA Results (1000 object-level permutations):")
    print(f"{'TR':<6}{'Observed Top-1000 F':<22}{'Null 95%':<12}{'p-value':<12}{'Bonferroni p (x10)':<20}{'Time (s)':<10}")
    print("-" * 82)
    tp_summary = []
    for t_idx, obs_f, null_m, null_95, p_v, p_bonf, el in tp_results:
        print(f"TR {t_idx:<3}{obs_f:>18.3f}{null_95:>14.3f}{p_v:>12.4f}{p_bonf:>18.4f}{el:>10.1f}s")
        tp_summary.append({
            "tr": t_idx,
            "observed_top1000_f": obs_f,
            "null_mean": null_m,
            "null_95": null_95,
            "p_value": p_v,
            "bonferroni_p": p_bonf,
            "elapsed_seconds": el
        })

    # Save results to json
    results = {
        "cramers_v": cramers_results,
        "pca_effect_sizes": pca_results,
        "temporal_anova_1000_perm": tp_summary
    }
    with open(OUT_DIR / "task0_followup_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Task 0 results to {OUT_DIR / 'task0_followup_results.json'}")
    
    # Update status.json marker
    common.update_status(
        "task0_followups",
        status="DONE",
        evidence_file="outputs/phase2/task0_followup_results.json",
        reason="1000 permutations per TR, Cramer's V (uncorr & bias-corr), PCA effect sizes (N1, N2, N3)"
    )


if __name__ == "__main__":
    main()
