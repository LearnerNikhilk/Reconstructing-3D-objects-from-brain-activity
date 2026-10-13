"""
End-to-End Pipeline for 3D Object Reconstruction from Human fMRI.

Usage:
    python code/run_all.py

Runtime: ≈ 8-12 minutes on CPU (tested on modern multi-core x86_64).
Pretrained models: None (all models fit strictly from scratch).
"""

from pathlib import Path
import sys
import os
import json
import time
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.decomposition import PCA

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
GRID_SIZE = 32
GRID_RANGE = 0.65
DELTA = (2.0 * GRID_RANGE) / GRID_SIZE

CACHE_DIR = WORKSPACE_ROOT / "outputs" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR = WORKSPACE_ROOT / "submission"
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = SUBMISSION_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. Volumetric Occupancy Representation and Point Cloud Decoding
# -----------------------------------------------------------------------------

def get_grid_coords():
    coords = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_SIZE, dtype=np.float32)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return coords, grid_pts


def load_or_compute_occupancies(shapes, cache_name):
    cache_path = CACHE_DIR / cache_name
    n_shapes = len(shapes)
    
    if cache_path.exists():
        V = np.load(cache_path)
        if V.shape == (n_shapes, GRID_SIZE, GRID_SIZE, GRID_SIZE):
            print(f"Loaded cached occupancies from {cache_path} {V.shape}", flush=True)
            return V
            
    print(f"Computing 32x32x32 occupancy grids for {n_shapes} shapes ({cache_name})...", flush=True)
    t0 = time.time()
    _, grid_pts = get_grid_coords()
    V = np.empty((n_shapes, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for i in range(n_shapes):
        if (i + 1) % 50 == 0 or i == n_shapes - 1:
            print(f"  Voxelizing shape {i+1}/{n_shapes}...", flush=True)
        tr = cKDTree(shapes[i])
        d = tr.query(grid_pts)[0]
        occ = (d < 0.045).astype(np.float32).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
        V[i] = occ
        
    np.save(cache_path, V)
    print(f"Computed and cached occupancies in {time.time() - t0:.2f}s", flush=True)
    return V


def decode_occupancy_to_points(V_volume, n_points=2048, seed=SEED, threshold=0.35):
    coords, grid_pts = get_grid_coords()
    grid_3d = grid_pts.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3)
    
    occupied = V_volume > threshold
    if np.sum(occupied) < 50:
        threshold = float(np.percentile(V_volume, 95.0))
        occupied = V_volume >= threshold

    eroded = binary_erosion(occupied)
    surface_mask = occupied & (~eroded)
    
    surf_indices = np.argwhere(surface_mask)
    if len(surf_indices) == 0:
        surf_indices = np.argwhere(occupied)
    if len(surf_indices) == 0:
        surf_indices = np.array([[GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2]])
        
    rng = np.random.RandomState(seed)
    replace = len(surf_indices) < n_points
    idx_sample = rng.choice(len(surf_indices), size=n_points, replace=replace)
    
    pts = grid_3d[surf_indices[idx_sample, 0], surf_indices[idx_sample, 1], surf_indices[idx_sample, 2]]
    jitter = rng.uniform(-DELTA / 2.0, DELTA / 2.0, size=(n_points, 3)).astype(np.float32)
    pts = (pts + jitter).astype(np.float32)
    return pts


# -----------------------------------------------------------------------------
# 2. Hemodynamic Feature Extraction
# -----------------------------------------------------------------------------

def extract_features(split):
    """Averages peak BOLD response across TRs 4..6 and applies per-scan spatial z-score."""
    X_win = common.window(split["X"].astype(np.float32), [4, 5, 6])
    return common.N3(X_win)


# -----------------------------------------------------------------------------
# 3. Model Components
# -----------------------------------------------------------------------------

class ClassMeanModel:
    def __init__(self, alpha=1000.0, T=1.22):
        self.alpha = alpha
        self.T = T
        self.clf = RidgeClassifier(alpha=self.alpha, class_weight="balanced", random_state=SEED)
        self.class_mean_volumes = {}
        
    def fit(self, Z_tr, y_tr, V_tr, obj_class_tr, classes):
        self.classes = classes
        self.clf.fit(Z_tr, y_tr)
        for c_idx, cname in enumerate(classes):
            c_objs = [i for i, c in enumerate(obj_class_tr) if c == cname]
            if len(c_objs) > 0:
                self.class_mean_volumes[c_idx] = np.mean(V_tr[c_objs], axis=0)
            else:
                self.class_mean_volumes[c_idx] = np.mean(V_tr, axis=0)
                
    def predict_volumes(self, Z_test):
        scores = self.clf.decision_function(Z_test)
        probs = np.exp(scores / self.T)
        probs /= np.sum(probs, axis=1, keepdims=True)
        pred_classes = np.argmax(probs, axis=1)
        N = len(Z_test)
        vols = np.empty((N, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i in range(N):
            vols[i] = self.class_mean_volumes[pred_classes[i]]
        return vols, probs


class LatentRegressionModel:
    def __init__(self, n_components=35, ridge_alpha=2000.0):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha
        self.pca = PCA(n_components=self.n_components, random_state=SEED)
        self.reg = Ridge(alpha=self.ridge_alpha, random_state=SEED)
        
    def fit(self, Z_tr, scan_obj_tr, V_tr_all):
        n_objs = len(V_tr_all)
        flat_V = V_tr_all.reshape(n_objs, -1)
        self.latent_tr_objs = self.pca.fit_transform(flat_V)
        Y_scans = self.latent_tr_objs[scan_obj_tr]
        self.reg.fit(Z_tr, Y_scans)
        
    def predict_volumes(self, Z_test):
        pred_latents = self.reg.predict(Z_test)
        flat_recon = self.pca.inverse_transform(pred_latents)
        return flat_recon.reshape(-1, GRID_SIZE, GRID_SIZE, GRID_SIZE)


class SoftRetrievalModel:
    def __init__(self, top_k=5, s=0.2):
        self.top_k = top_k
        self.s = s
        
    def fit(self, V_tr_all, obj_class_tr, classes):
        self.V_tr_all = V_tr_all
        self.obj_classes = np.array([classes.index(c) for c in obj_class_tr])
        self.n_objs = len(V_tr_all)
        
    def predict_volumes(self, probs):
        N = len(probs)
        vols = np.empty((N, GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        n_c = np.bincount(self.obj_classes, minlength=5)
        for i in range(N):
            p_tempered = np.power(probs[i], 1.0 / self.s)
            p_tempered /= np.sum(p_tempered)
            w = p_tempered[self.obj_classes] / n_c[self.obj_classes]
            top_k_idx = np.argsort(w)[::-1][:self.top_k]
            top_w = w[top_k_idx]
            top_w /= np.sum(top_w)
            vols[i] = np.sum(self.V_tr_all[top_k_idx] * top_w[:, None, None, None], axis=0)
        return vols


class BlendAndShrinkPipeline:
    def __init__(self, w_a=0.25, w_b=0.45, w_c=0.30, gamma=0.15):
        self.w_a = w_a
        self.w_b = w_b
        self.w_c = w_c
        self.gamma = gamma
        self.model_a = ClassMeanModel(alpha=1000.0, T=1.22)
        self.model_b = LatentRegressionModel(n_components=35, ridge_alpha=2000.0)
        self.model_c = SoftRetrievalModel(top_k=5, s=0.2)
        
    def fit(self, Z_tr, y_tr, scan_obj_tr, V_tr_all, obj_class_tr, classes):
        self.classes = classes
        self.global_mean_volume = np.mean(V_tr_all, axis=0)
        self.model_a.fit(Z_tr, y_tr, V_tr_all, obj_class_tr, classes)
        self.model_b.fit(Z_tr, scan_obj_tr, V_tr_all)
        self.model_c.fit(V_tr_all, obj_class_tr, classes)
        
    def predict_points(self, Z_test, gamma=None):
        if gamma is None:
            gamma = self.gamma
        V_a, probs = self.model_a.predict_volumes(Z_test)
        V_b = self.model_b.predict_volumes(Z_test)
        V_c = self.model_c.predict_volumes(probs)
        V_blend = self.w_a * V_a + self.w_b * V_b + self.w_c * V_c
        V_final = (1.0 - gamma) * V_blend + gamma * self.global_mean_volume
        N = len(Z_test)
        pts = np.empty((N, 2048, 3), dtype=np.float32)
        for i in range(N):
            pts[i] = decode_occupancy_to_points(V_final[i], n_points=2048, seed=SEED + i)
        return pts


# -----------------------------------------------------------------------------
# 4. Main Pipeline Execution
# -----------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Starting 3D reconstruction pipeline...")
    print("Runtime estimate: ≈ 8-12 min on CPU")
    print("Pretrained models: None (fitted from scratch)")
    
    # 1. Load Data
    print("\n[1/5] Loading data splits...", flush=True)
    train = starter.load_split("train")
    val = starter.load_split("val")
    test = starter.load_split("test")
    print(f"Dataset summary: train={len(train['X'])}, val={len(val['X'])}, test={len(test['X'])} scans")
    
    # 2. Occupancy Grids
    print("\n[2/5] Preparing 32x32x32 shape occupancy fields...", flush=True)
    V_train_all = load_or_compute_occupancies(train["obj_points"].astype(np.float32), "occupancy_train_32.npy")
    V_val_all = load_or_compute_occupancies(val["obj_points"].astype(np.float32), "occupancy_val_32.npy")
    
    # 3. Features
    print("\n[3/5] Extracting canonical fMRI features (Window W1 + Normalization N3)...", flush=True)
    Z_train = extract_features(train)
    Z_val = extract_features(val)
    Z_test = extract_features(test)
    
    # 4. Validation Evaluation
    print("\n[4/5] Evaluating on validation split (125 scans) via evaluate.py...", flush=True)
    classes = sorted(list(train["class_names"]))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[c] for c in train["obj_class"][train["group"]]])
    
    pipeline_val = BlendAndShrinkPipeline(w_a=0.25, w_b=0.45, w_c=0.30, gamma=0.15)
    pipeline_val.fit(Z_train, y_train, train["group"], V_train_all, train["obj_class"], classes)
    val_pred_pts = pipeline_val.predict_points(Z_val)
    
    val_pred_path = WORKSPACE_ROOT / "val_predictions.npz"
    np.savez_compressed(val_pred_path, sample_id=val["sample_id"], points=val_pred_pts)
    
    val_metrics_save = METRICS_DIR / "val_metrics.json"
    cmd_eval = [sys.executable, "evaluate.py", "val_predictions.npz", "--save", str(val_metrics_save)]
    res_eval = subprocess.run(cmd_eval, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
    assert res_eval.returncode == 0, f"evaluate.py failed:\n{res_eval.stderr}"
    print(res_eval.stdout)
    
    with open(val_metrics_save) as f:
        val_metrics = json.load(f)
    print(f"Validation summary: Composite={val_metrics['composite']:+.4f}, Identification={val_metrics['identification']:.2f}%")
    
    # 5. Retrain on Train + Val & Predict Test
    print("\n[5/5] Retraining model on train + val (658 scans) and predicting test set...", flush=True)
    Z_full = np.concatenate([Z_train, Z_val], axis=0)
    V_full = np.concatenate([V_train_all, V_val_all], axis=0)
    obj_class_full = list(train["obj_class"]) + list(val["obj_class"])
    group_full = np.concatenate([train["group"], val["group"] + len(train["obj_points"])], axis=0)
    y_full = np.array([class_to_idx[obj_class_full[g]] for g in group_full])
    
    pipeline_full = BlendAndShrinkPipeline(w_a=0.25, w_b=0.45, w_c=0.30, gamma=0.15)
    pipeline_full.fit(Z_full, y_full, group_full, V_full, obj_class_full, classes)
    
    test_pred_pts = pipeline_full.predict_points(Z_test)
    starter.save_predictions(str(SUBMISSION_DIR / "predictions.npz"), test["sample_id"], test_pred_pts)
    starter.export_outputs(str(SUBMISSION_DIR), test["sample_id"], test_pred_pts)
    print(f"Exported test predictions, 167 PLY point clouds, and 668 multi-view renders.")
    
    # Generate Report PDF
    cmd_pdf = [sys.executable, str(CODE_DIR / "generate_report_pdf.py")]
    subprocess.run(cmd_pdf, cwd=str(WORKSPACE_ROOT), check=True)
    
    # Synchronize code folder into submission
    sub_code = SUBMISSION_DIR / "code"
    if sub_code.exists():
        shutil.rmtree(sub_code)
    shutil.copytree(CODE_DIR, sub_code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    
    # Verify Submission Folder
    print("\nVerifying submission directory with check_submission.py...")
    cmd_check = [sys.executable, "check_submission.py", "submission/"]
    res_check = subprocess.run(cmd_check, capture_output=True, text=True, cwd=str(WORKSPACE_ROOT))
    print(res_check.stdout)
    assert res_check.returncode == 0, f"check_submission.py failed:\n{res_check.stderr}\n{res_check.stdout}"
    
    elapsed = time.time() - t_start
    print(f"Pipeline finished in {elapsed:.1f}s ({elapsed/60:.1f} min).")


if __name__ == "__main__":
    main()
