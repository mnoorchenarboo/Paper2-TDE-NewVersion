"""
XAI Analysis System v3.2 - Smart Detection

9 XAI Methods: gradient, deep, permutation, kernel, partition, lime, sampling, tde, fastshap

Features:
- Smart detection of existing data in xai_results.db
- Adaptive user prompts based on existing configuration
- Progress tracking with summary tables
- Background change triggers full reset with confirmation
"""

# ============================
# LIBRARY IMPORTS
# ============================
import numpy as np
import pandas as pd
import sqlite3
import time
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import shap

from scipy.stats import pearsonr, kendalltau

# TDE/FastSHAP loader
try:
    from tde import load_explainer_for_inference
    TDE_AVAILABLE = True
except ImportError:
    TDE_AVAILABLE = False
    print("⚠️ TDE/FastSHAP modules not available")

# LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not installed (pip install lime)")

# ============================
# CONFIGURATION
# ============================
from pathlib import Path

PATH_DBS = Path("databases")
PATH_DBS.mkdir(parents=True, exist_ok=True)

BENCHMARK_DB = PATH_DBS / "benchmark_results.db"
ENERGY_DB    = PATH_DBS / "energy_data.db"
XAI_DB = PATH_DBS / "xai_results.db"
RESULTS_BASE_DIR = "results"

XAI_METHODS = ['gradient', 'deep', 'permutation', 'partition', 'lime', 'sampling', 'tde', 'fastshap']
BACKGROUND_TYPES = ['random', 'kmeans', 'feature_mean']

DEFAULT_BG_TYPE = 'random'
DEFAULT_BG_SIZE = 10
DEFAULT_N_SAMPLES = 10

NOISE_STD = 0.01
FIDELITY_TOPK_PCT = 10.0
SPARSITY_THRESHOLD_PCT = 1.0
RELIABILITY_TOPK_PCT = 10.0
RANDOM_SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}")


# ============================
# DATABASE UTILITIES
# ============================
def db_execute(func, max_retries=5, delay=2):
    """Execute database function with retry on lock."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise


def init_database(db_path=XAI_DB):
    """Initialize XAI database with all required tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            random_seed INTEGER NOT NULL,
            background_type TEXT NOT NULL,
            background_size INTEGER NOT NULL,
            use_same_for_noisy INTEGER NOT NULL,
            noise_std REAL NOT NULL,
            fidelity_topk_pct REAL NOT NULL,
            sparsity_threshold_pct REAL NOT NULL,
            reliability_topk_pct REAL NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS background_data (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            background_original_json TEXT NOT NULL,
            background_noisy_json TEXT,
            PRIMARY KEY (primary_use, option_number)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS test_samples (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            sample_idx INTEGER NOT NULL,
            original_sample_json TEXT NOT NULL,
            noisy_sample_json TEXT NOT NULL,
            PRIMARY KEY (primary_use, option_number, sample_idx)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS xai_results (
            primary_use TEXT NOT NULL,
            option_number INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            xai_method TEXT NOT NULL,
            fidelity REAL,
            sparsity REAL,
            complexity REAL,
            reliability_ped REAL,
            reliability_correlation REAL,
            reliability_topk_overlap REAL,
            reliability_kendall_tau REAL,
            efficiency_error REAL,
            computation_time REAL,
            shap_values_original_json TEXT,
            shap_values_noisy_json TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, sample_idx, xai_method)
        )
    ''')
    
    # Add efficiency_error column if missing (for existing databases)
    try:
        cursor.execute('ALTER TABLE xai_results ADD COLUMN efficiency_error REAL')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.commit()
    conn.close()


# ============================
# EXISTING DATA DETECTION
# ============================
def get_existing_config(primary_use, option_number, db_path=XAI_DB):
    """
    Get all existing configuration for a dataset.
    Returns dict with settings, background, samples, and results info.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    result = {
        'has_data': False,
        'settings': None,
        'background': None,
        'samples': [],
        'n_samples': 0,
        'results_summary': {},
        'total_results': 0,
        'models_with_results': [],
        'methods_with_results': []
    }
    
    # Get settings
    cursor.execute('SELECT * FROM settings WHERE id = 1')
    row = cursor.fetchone()
    if row:
        result['settings'] = {
            'random_seed': row[1],
            'background_type': row[2],
            'background_size': row[3],
            'use_same_for_noisy': bool(row[4]),
            'noise_std': row[5]
        }
    
    # Get background
    cursor.execute('''
        SELECT background_original_json, background_noisy_json 
        FROM background_data WHERE primary_use = ? AND option_number = ?
    ''', (primary_use, option_number))
    row = cursor.fetchone()
    if row:
        result['background'] = {
            'original_shape': np.array(json.loads(row[0])).shape if row[0] else None,
            'has_separate_noisy': row[1] is not None
        }
    
    # Get test samples
    cursor.execute('''
        SELECT sample_idx FROM test_samples 
        WHERE primary_use = ? AND option_number = ?
        ORDER BY sample_idx
    ''', (primary_use, option_number))
    rows = cursor.fetchall()
    result['samples'] = [r[0] for r in rows]
    result['n_samples'] = len(result['samples'])
    
    # Get results summary
    cursor.execute('''
        SELECT model_name, xai_method, COUNT(*) as count
        FROM xai_results
        WHERE primary_use = ? AND option_number = ?
        GROUP BY model_name, xai_method
    ''', (primary_use, option_number))
    rows = cursor.fetchall()
    
    for model, method, count in rows:
        if model not in result['results_summary']:
            result['results_summary'][model] = {}
        result['results_summary'][model][method] = count
        result['total_results'] += count
        if model not in result['models_with_results']:
            result['models_with_results'].append(model)
        if method not in result['methods_with_results']:
            result['methods_with_results'].append(method)
    
    result['has_data'] = result['n_samples'] > 0 or result['total_results'] > 0
    
    conn.close()
    return result


def get_progress_summary(primary_use, option_number, models, methods, db_path=XAI_DB):
    """
    Get detailed progress summary showing completed vs remaining work.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get test samples
    cursor.execute('''
        SELECT sample_idx FROM test_samples 
        WHERE primary_use = ? AND option_number = ?
    ''', (primary_use, option_number))
    sample_indices = [r[0] for r in cursor.fetchall()]
    n_samples = len(sample_indices)
    
    if n_samples == 0:
        conn.close()
        return None
    
    # Calculate totals
    total_possible = n_samples * len(models) * len(methods)
    
    # Get completed results
    cursor.execute('''
        SELECT model_name, xai_method, COUNT(*) as count
        FROM xai_results
        WHERE primary_use = ? AND option_number = ?
        GROUP BY model_name, xai_method
    ''', (primary_use, option_number))
    
    completed_by_model_method = {}
    total_completed = 0
    for model, method, count in cursor.fetchall():
        key = (model, method)
        completed_by_model_method[key] = count
        total_completed += count
    
    # Build progress matrix
    progress = {
        'n_samples': n_samples,
        'sample_indices': sample_indices,
        'models': models,
        'methods': methods,
        'total_possible': total_possible,
        'total_completed': total_completed,
        'total_remaining': total_possible - total_completed,
        'percent_complete': (total_completed / total_possible * 100) if total_possible > 0 else 0,
        'by_model': {},
        'by_method': {}
    }
    
    # Per model stats
    for model in models:
        model_total = n_samples * len(methods)
        model_done = sum(completed_by_model_method.get((model, m), 0) for m in methods)
        progress['by_model'][model] = {
            'total': model_total,
            'completed': model_done,
            'remaining': model_total - model_done,
            'percent': (model_done / model_total * 100) if model_total > 0 else 0
        }
    
    # Per method stats
    for method in methods:
        method_total = n_samples * len(models)
        method_done = sum(completed_by_model_method.get((m, method), 0) for m in models)
        progress['by_method'][method] = {
            'total': method_total,
            'completed': method_done,
            'remaining': method_total - method_done,
            'percent': (method_done / method_total * 100) if method_total > 0 else 0
        }
    
    conn.close()
    return progress


def print_progress_table(progress):
    """Print a nicely formatted progress table."""
    if progress is None:
        print("\n  📊 No test samples configured yet.")
        return
    
    print("\n" + "="*80)
    print("📊 PROGRESS SUMMARY")
    print("="*80)
    
    print(f"\n  Test Samples: {progress['n_samples']} (indices: {progress['sample_indices']})")
    print(f"  Models: {len(progress['models'])}")
    print(f"  Methods: {len(progress['methods'])}")
    
    print(f"\n  Overall Progress: {progress['total_completed']}/{progress['total_possible']} "
          f"({progress['percent_complete']:.1f}%) - {progress['total_remaining']} remaining")
    
    # Progress by model
    print(f"\n  {'Model':<12} {'Completed':>10} {'Remaining':>10} {'Progress':>12}")
    print(f"  {'-'*46}")
    for model in progress['models']:
        stats = progress['by_model'][model]
        bar_len = int(stats['percent'] / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {model:<12} {stats['completed']:>10} {stats['remaining']:>10} {bar} {stats['percent']:>5.1f}%")
    
    # Progress by method
    print(f"\n  {'Method':<12} {'Completed':>10} {'Remaining':>10} {'Progress':>12}")
    print(f"  {'-'*46}")
    for method in progress['methods']:
        stats = progress['by_method'][method]
        bar_len = int(stats['percent'] / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {method:<12} {stats['completed']:>10} {stats['remaining']:>10} {bar} {stats['percent']:>5.1f}%")
    
    print("="*80)


# ============================
# DATABASE OPERATIONS
# ============================
def save_settings(bg_type, bg_size, use_same_for_noisy, db_path=XAI_DB):
    """Save global settings."""
    def _exec():
        conn = sqlite3.connect(db_path)
        conn.execute('''
            INSERT OR REPLACE INTO settings 
            (id, random_seed, background_type, background_size, use_same_for_noisy,
             noise_std, fidelity_topk_pct, sparsity_threshold_pct, reliability_topk_pct)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (RANDOM_SEED, bg_type, bg_size, int(use_same_for_noisy),
              NOISE_STD, FIDELITY_TOPK_PCT, SPARSITY_THRESHOLD_PCT, RELIABILITY_TOPK_PCT))
        conn.commit()
        conn.close()
    db_execute(_exec)


def clear_all_data(primary_use, option_number, db_path=XAI_DB):
    """Clear all data for a dataset (background, samples, results)."""
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM xai_results WHERE primary_use = ? AND option_number = ?',
                      (primary_use, option_number))
        results_deleted = cursor.rowcount
        
        cursor.execute('DELETE FROM test_samples WHERE primary_use = ? AND option_number = ?',
                      (primary_use, option_number))
        samples_deleted = cursor.rowcount
        
        cursor.execute('DELETE FROM background_data WHERE primary_use = ? AND option_number = ?',
                      (primary_use, option_number))
        bg_deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        return results_deleted, samples_deleted, bg_deleted
    
    return db_execute(_exec)


def clear_results_only(primary_use, option_number, db_path=XAI_DB):
    """Clear only XAI results (keep samples and background)."""
    def _exec():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM xai_results WHERE primary_use = ? AND option_number = ?',
                      (primary_use, option_number))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted
    return db_execute(_exec)


# ============================
# MODEL WRAPPER
# ============================
class SingleHorizonWrapper(nn.Module):
    def __init__(self, model, horizon_idx=0):
        super().__init__()
        self.model = model
        self.horizon_idx = horizon_idx
    
    def forward(self, x):
        out = self.model(x)
        if out.ndim > 1 and out.shape[1] > self.horizon_idx:
            return out[:, self.horizon_idx:self.horizon_idx+1]
        return out


# ============================
# DATA LOADING
# ============================
def load_trained_model(primary_use, option_number, model_name):
    """Load trained model from results directory."""
    from dl import load_complete_model
    
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_complete_model(str(model_path), device=device)
    model.eval()
    
    for cfg_name in ["model_config.json", "model_metadata.json"]:
        cfg_path = model_dir / cfg_name
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                return model, json.load(f)
    
    return model, {'seq_length': 24, 'n_features': 10, 'prediction_horizon': 1}


def load_dataset(primary_use, option_number):
    """Load dataset using preprocess module."""
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path=ENERGY_DB,
        primary_use=primary_use,
        option_number=option_number,
        scaled=True,
        scale_type="both"
    )


def get_available_datasets():
    """Get all datasets with trained models."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('''
        SELECT DISTINCT primary_use, option_number
        FROM prediction_performance
        ORDER BY primary_use, option_number
    ''', conn)
    conn.close()
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} 
            for _, r in df.iterrows()]


def get_available_models(primary_use, option_number):
    """Get models available for a dataset."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('''
        SELECT DISTINCT model_name FROM prediction_performance
        WHERE primary_use = ? AND option_number = ?
        ORDER BY model_name
    ''', conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()


# ============================
# BACKGROUND & SAMPLE MANAGEMENT
# ============================
def generate_background(X_train, bg_type, bg_size, seed=RANDOM_SEED):
    """Generate background data based on type."""
    np.random.seed(seed)
    n = min(bg_size, len(X_train))
    
    if bg_type == 'random':
        indices = np.random.choice(len(X_train), n, replace=False)
        return X_train[indices].astype(np.float32)
    elif bg_type == 'kmeans':
        from sklearn.cluster import KMeans
        X_flat = X_train.reshape(len(X_train), -1)
        kmeans = KMeans(n_clusters=n, random_state=seed, n_init=10)
        kmeans.fit(X_flat)
        centroids = kmeans.cluster_centers_.reshape(n, X_train.shape[1], X_train.shape[2])
        return centroids.astype(np.float32)
    elif bg_type == 'feature_mean':
        return np.mean(X_train, axis=0, keepdims=True).astype(np.float32)
    raise ValueError(f"Unknown background type: {bg_type}")


def add_gaussian_noise(sample, noise_std=NOISE_STD):
    """Add small Gaussian noise for reliability testing."""
    return np.clip(sample + np.random.normal(0, noise_std, sample.shape), 0, 1).astype(np.float32)

def save_background_data(primary_use, option_number, bg_original, bg_noisy, db_path=XAI_DB):
    """Save background data."""
    def _exec():
        conn = sqlite3.connect(db_path)
        conn.execute('''
            INSERT OR REPLACE INTO background_data
            (primary_use, option_number, background_original_json, background_noisy_json)
            VALUES (?, ?, ?, ?)
        ''', (primary_use, option_number, 
              json.dumps(bg_original.tolist()),
              json.dumps(bg_noisy.tolist()) if bg_noisy is not None else None))
        conn.commit()
        conn.close()
    db_execute(_exec)

def get_background_data(primary_use, option_number, db_path=XAI_DB):
    """Get background data for a dataset."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT background_original_json, background_noisy_json
        FROM background_data WHERE primary_use = ? AND option_number = ?
    ''', (primary_use, option_number))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        bg_original = np.array(json.loads(result[0]), dtype=np.float32)
        bg_noisy = np.array(json.loads(result[1]), dtype=np.float32) if result[1] else None
        return {'original': bg_original, 'noisy': bg_noisy}
    return None

def get_test_samples(primary_use, option_number, db_path=XAI_DB):
    """Get all test samples for a dataset."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sample_idx, original_sample_json, noisy_sample_json
        FROM test_samples WHERE primary_use = ? AND option_number = ?
        ORDER BY sample_idx
    ''', (primary_use, option_number))
    results = cursor.fetchall()
    conn.close()
    
    if results:
        return {r[0]: {
            'original': np.array(json.loads(r[1]), dtype=np.float32),
            'noisy': np.array(json.loads(r[2]), dtype=np.float32)
        } for r in results}
    return None

def save_test_sample(primary_use, option_number, sample_idx, original, noisy, db_path=XAI_DB):
    """Save a single test sample."""
    def _exec():
        conn = sqlite3.connect(db_path)
        conn.execute('''
            INSERT OR REPLACE INTO test_samples
            (primary_use, option_number, sample_idx, original_sample_json, noisy_sample_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (primary_use, option_number, sample_idx,
              json.dumps(original.tolist()), json.dumps(noisy.tolist())))
        conn.commit()
        conn.close()
    db_execute(_exec)


def clear_test_samples(primary_use, option_number, db_path=XAI_DB):
    """Clear test samples for a dataset."""
    def _exec():
        conn = sqlite3.connect(db_path)
        conn.execute('DELETE FROM test_samples WHERE primary_use = ? AND option_number = ?',
                    (primary_use, option_number))
        conn.commit()
        conn.close()
    db_execute(_exec)


# ============================
# SHAP COMPUTATION
# ============================
def compute_shap_values(model, sample, background, xai_method, seq_len, n_features,
                        primary_use=None, option_number=None, model_name=None):
    """Compute SHAP values using specified method."""
    
    def model_predict(X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(device)
        if X.ndim == 2:
            X = X.reshape(-1, seq_len, n_features)
        with torch.no_grad():
            return model(X).cpu().numpy().mean(axis=1).flatten()
    
    def predict_first_horizon(X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if X.ndim == 2:
            X = X.reshape(-1, seq_len, n_features)
        with torch.no_grad():
            pred = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    try:
        # TDE / FastSHAP
        if xai_method in ['tde', 'fastshap']:
            if not TDE_AVAILABLE:
                print(f"      ⚠️ {xai_method}: TDE module not available")
                return None
            
            # FIX: Check for None explicitly, not truthiness (option_number=0 is valid!)
            if primary_use is None or option_number is None or model_name is None:
                print(f"      ⚠️ {xai_method}: Missing required params (primary_use={primary_use}, option_number={option_number}, model_name={model_name})")
                return None
            
            try:
                exp_path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / xai_method
                exp_file = exp_path / f"{xai_method}_explainer.pt"
                
                if not exp_file.exists():
                    print(f"      ⚠️ {xai_method}: Explainer file not found: {exp_file}")
                    return None
                
                print(f"      Loading {xai_method} from {exp_path}...")
                
                explainer = load_explainer_for_inference(
                    primary_use, option_number, model_name,
                    explainer_type=xai_method,
                    model_predict_func=predict_first_horizon,
                    device_override=device
                )
                
                shap_vals = explainer.explain(sample)
                
                if shap_vals is not None:
                    print(f"      ✓ {xai_method}: Got shape {shap_vals.shape}")
                    if shap_vals.shape == (seq_len, n_features):
                        return shap_vals.astype(np.float32)
                    else:
                        print(f"      ⚠️ {xai_method}: Shape mismatch, expected ({seq_len}, {n_features})")
                return None
                
            except FileNotFoundError as e:
                print(f"      ⚠️ {xai_method}: File not found - {e}")
                return None
            except Exception as e:
                print(f"      ❌ {xai_method}: Error - {e}")
                import traceback
                traceback.print_exc()
                return None

        # Gradient SHAP / Deep SHAP
        elif xai_method in ['gradient', 'deep']:
            sample_torch = torch.FloatTensor(sample).unsqueeze(0).to(device)
            bg_torch = torch.FloatTensor(background).to(device)
            wrapped = SingleHorizonWrapper(model).to(device)
            wrapped.eval()
            
            if xai_method == 'gradient':
                shap_vals = shap.GradientExplainer(wrapped, bg_torch).shap_values(sample_torch)
            else:
                shap_vals = shap.DeepExplainer(wrapped, bg_torch).shap_values(sample_torch, check_additivity=False)
        
        # Model-agnostic methods
        else:
            bg_flat = background.reshape(len(background), -1)
            test_flat = sample.reshape(1, -1)
            
            if xai_method == 'permutation':
                shap_vals = shap.PermutationExplainer(model_predict, bg_flat).shap_values(test_flat, npermutations=20)
            elif xai_method == 'partition':
                exp = shap.PartitionExplainer(model_predict, bg_flat)(test_flat, max_evals=50)
                shap_vals = exp.values if hasattr(exp, 'values') else exp
            elif xai_method == 'lime':
                if not LIME_AVAILABLE:
                    return None
                explainer = LimeTabularExplainer(bg_flat, mode="regression",
                    feature_names=[f"f_{i}" for i in range(bg_flat.shape[1])])
                lime_exp = explainer.explain_instance(test_flat.flatten(), model_predict,
                    num_features=bg_flat.shape[1], num_samples=100)
                shap_vals = np.zeros(bg_flat.shape[1])
                for idx, imp in lime_exp.local_exp[1]:
                    shap_vals[idx] = imp
                shap_vals = shap_vals.reshape(1, -1)
            elif xai_method == 'sampling':
                exp = shap.explainers.Sampling(model_predict, bg_flat)(test_flat)
                shap_vals = exp.values if hasattr(exp, 'values') else exp
            else:
                return None
        
        # Shape normalization
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if isinstance(shap_vals, torch.Tensor):
            shap_vals = shap_vals.cpu().numpy()
        shap_vals = np.array(shap_vals, dtype=np.float32)
        
        if shap_vals.ndim == 4:
            shap_vals = shap_vals[0, :, :, 0]
        elif shap_vals.ndim == 3:
            shap_vals = shap_vals[0] if shap_vals.shape[0] == 1 else shap_vals[:, :, 0] if shap_vals.shape[2] == 1 else shap_vals[0]
        
        if shap_vals.ndim == 1 and shap_vals.size == seq_len * n_features:
            shap_vals = shap_vals.reshape(seq_len, n_features)
        elif shap_vals.ndim == 2 and shap_vals.shape != (seq_len, n_features):
            if shap_vals.shape == (n_features, seq_len):
                shap_vals = shap_vals.T
            elif shap_vals.size == seq_len * n_features:
                shap_vals = shap_vals.reshape(seq_len, n_features)
        
        return shap_vals.astype(np.float32) if shap_vals.shape == (seq_len, n_features) else None
    
    except Exception as e:
        return None
    
# ============================
# METRICS
# ============================
def compute_fidelity(model, sample, shap_vals, baseline, seq_len, n_features):
    """Compute fidelity by perturbing top-k features (matching tde.py)."""
    if shap_vals is None:
        return None
    try:
        # Wrap model for first horizon (like tde.py)
        wrapped = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        wrapped.eval()
        
        sample_t = torch.FloatTensor(sample).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_pred = wrapped(sample_t).cpu().numpy().flatten()[0]
        
        # Use median baseline if 3D, otherwise use directly (like tde.py)
        if baseline.ndim == 3:
            baseline_2d = np.median(baseline, axis=0)
        else:
            baseline_2d = baseline
        
        k = max(1, int(shap_vals.size * FIDELITY_TOPK_PCT / 100))
        top_k = np.argsort(np.abs(shap_vals).flatten())[-k:]
        
        masked = sample.copy()
        for idx in top_k:
            masked[idx // n_features, idx % n_features] = baseline_2d[idx // n_features, idx % n_features]
        
        with torch.no_grad():
            masked_pred = wrapped(torch.FloatTensor(masked).unsqueeze(0).to(device)).cpu().numpy().flatten()[0]
        
        return float(abs(orig_pred - masked_pred))
    except:
        return None

def compute_sparsity(shap_vals):
    """Compute sparsity as percentage of near-zero values."""
    if shap_vals is None:
        return None
    try:
        max_val = np.max(np.abs(shap_vals))
        if max_val == 0:
            return 100.0
        threshold = max_val * SPARSITY_THRESHOLD_PCT / 100
        return float(np.sum(np.abs(shap_vals) < threshold) / shap_vals.size * 100)
    except:
        return None

def compute_complexity(shap_vals):
    """Compute complexity as entropy."""
    if shap_vals is None:
        return None
    try:
        shap_abs = np.abs(shap_vals).flatten() + 1e-10
        shap_norm = shap_abs / np.sum(shap_abs)
        return float(-np.sum(shap_norm * np.log(shap_norm)))
    except:
        return None

def compute_reliability_metrics(shap_original, shap_noisy):
    """Compute reliability metrics comparing original vs noisy SHAP values."""
    if shap_original is None or shap_noisy is None:
        return {'error_pct': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}
    
    try:
        orig_flat = shap_original.flatten()
        noisy_flat = shap_noisy.flatten()
        
        valid_mask = np.isfinite(orig_flat) & np.isfinite(noisy_flat)
        if np.sum(valid_mask) < 10:
            return {'error_pct': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}
        
        orig_valid, noisy_valid = orig_flat[valid_mask], noisy_flat[valid_mask]
        
        max_mag = max(np.max(np.abs(orig_valid)), np.max(np.abs(noisy_valid)), 1e-10)
        error_pct = np.mean(np.abs(orig_valid - noisy_valid)) / max_mag * 100
        
        correlation, _ = pearsonr(orig_valid, noisy_valid)
        
        k = max(1, int(len(orig_flat) * RELIABILITY_TOPK_PCT / 100))
        top_k_orig = set(np.argsort(np.abs(orig_flat))[-k:])
        top_k_noisy = set(np.argsort(np.abs(noisy_flat))[-k:])
        topk_overlap = len(top_k_orig & top_k_noisy) / k * 100
        
        kendall_tau, _ = kendalltau(orig_valid, noisy_valid)
        
        return {
            'error_pct': float(error_pct) if np.isfinite(error_pct) else None,
            'correlation': float(correlation) if np.isfinite(correlation) else None,
            'topk_overlap': float(topk_overlap) if np.isfinite(topk_overlap) else None,
            'kendall_tau': float(kendall_tau) if np.isfinite(kendall_tau) else None
        }
    except:
        return {'error_pct': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}

def compute_efficiency_error(model, sample, shap_vals, baseline, seq_len, n_features):
    """Compute efficiency error: how well SHAP values sum to prediction difference."""
    if shap_vals is None:
        return None
    try:
        wrapped = SingleHorizonWrapper(model, horizon_idx=0).to(device)
        wrapped.eval()
        
        sample_t = torch.FloatTensor(sample).unsqueeze(0).to(device)
        with torch.no_grad():
            sample_pred = wrapped(sample_t).cpu().numpy().flatten()[0]
        
        # Get baseline prediction
        if baseline.ndim == 3:
            baseline_2d = np.median(baseline, axis=0)
        else:
            baseline_2d = baseline
        
        baseline_t = torch.FloatTensor(baseline_2d).unsqueeze(0).to(device)
        with torch.no_grad():
            base_pred = wrapped(baseline_t).cpu().numpy().flatten()[0]
        
        # Efficiency: sum(shap) should equal f(x) - f(baseline)
        expected_diff = sample_pred - base_pred
        actual_sum = np.sum(shap_vals)
        
        # Return relative error
        return float(abs(actual_sum - expected_diff) / (abs(expected_diff) + 1e-10))
    except:
        return None

# ============================
# XAI RESULTS STORAGE
# ============================
def save_xai_result(primary_use, option_number, model_name, sample_idx, xai_method,
                    fidelity, sparsity, complexity, reliability, efficiency_error,
                    computation_time, shap_original, shap_noisy, db_path=XAI_DB):
    """Save XAI result."""
    def _exec():
        conn = sqlite3.connect(db_path)
        conn.execute('''
            INSERT OR REPLACE INTO xai_results
            (primary_use, option_number, model_name, sample_idx, xai_method,
             fidelity, sparsity, complexity,
             reliability_ped, reliability_correlation, reliability_topk_overlap, reliability_kendall_tau,
             efficiency_error, computation_time, shap_values_original_json, shap_values_noisy_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (primary_use, option_number, model_name, sample_idx, xai_method,
              fidelity, sparsity, complexity,
              reliability.get('error_pct'), reliability.get('correlation'),
              reliability.get('topk_overlap'), reliability.get('kendall_tau'),
              efficiency_error, computation_time,
              json.dumps(shap_original.tolist()) if shap_original is not None else None,
              json.dumps(shap_noisy.tolist()) if shap_noisy is not None else None
              ))
        conn.commit()
        conn.close()
    db_execute(_exec)


def check_result_exists(primary_use, option_number, model_name, sample_idx, xai_method, db_path=XAI_DB):
    """Check if XAI result exists."""
    conn = sqlite3.connect(db_path)
    exists = conn.execute('''
        SELECT COUNT(*) FROM xai_results
        WHERE primary_use = ? AND option_number = ? AND model_name = ? AND sample_idx = ? AND xai_method = ?
    ''', (primary_use, option_number, model_name, sample_idx, xai_method)).fetchone()[0] > 0
    conn.close()
    return exists


# ============================
# USER INPUT - SMART DETECTION
# ============================
def get_user_inputs():
    """Get user configuration with smart detection of existing data."""
    print("\n" + "="*80)
    print("🚀 XAI Analysis System v3.2 - Smart Detection")
    print("="*80)
    print(f"Methods: {', '.join(XAI_METHODS)}")
    print("="*80)
    
    # Initialize DB
    init_database()
    
    # Get datasets
    datasets = get_available_datasets()
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    # Select primary use
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\n📊 Available Primary Uses:")
    for i, use in enumerate(uses):
        print(f"  {i}: {use}")
    
    use_input = input(f"\n--> Select [0-{len(uses)-1}] or 'all' [0]: ").strip().lower()
    if use_input == 'all':
        selected_uses = uses
    else:
        try:
            idx = int(use_input) if use_input else 0
            selected_uses = [uses[idx]] if 0 <= idx < len(uses) else [uses[0]]
        except:
            selected_uses = [uses[0]]
    
    selected_datasets = [d for d in datasets if d['primary_use'] in selected_uses]
    
    # For each dataset, check existing config
    all_configs = []
    
    for ds in selected_datasets:
        primary_use, option_number = ds['primary_use'], ds['option_number']
        models = get_available_models(primary_use, option_number)
        
        print(f"\n{'='*80}")
        print(f"📊 Dataset: {primary_use} - Option {option_number}")
        print(f"   Available Models: {', '.join(models)}")
        print(f"{'='*80}")
        
        # Check existing configuration
        existing = get_existing_config(primary_use, option_number)
        
        # Filter methods by availability
        available_methods = XAI_METHODS.copy()
        if not TDE_AVAILABLE:
            available_methods = [m for m in available_methods if m not in ['tde', 'fastshap']]
        if not LIME_AVAILABLE:
            available_methods = [m for m in available_methods if m != 'lime']
        
        if existing['has_data']:
            # ========== EXISTING DATA FOUND ==========
            print(f"\n  ✅ EXISTING DATA FOUND:")
            
            # Show settings
            if existing['settings']:
                s = existing['settings']
                print(f"     Background: {s['background_type']} (size={s['background_size']})")
                print(f"     Same for noisy: {'Yes' if s['use_same_for_noisy'] else 'No'}")
            
            # Show samples
            print(f"     Test samples: {existing['n_samples']} (indices: {existing['samples']})")
            
            # Show results
            print(f"     Total results: {existing['total_results']}")
            if existing['results_summary']:
                print(f"     Models with results: {existing['models_with_results']}")
                print(f"     Methods with results: {existing['methods_with_results']}")
            
            # Show progress
            progress = get_progress_summary(primary_use, option_number, models, available_methods)
            if progress:
                print_progress_table(progress)
            
            # ========== ASK ABOUT BACKGROUND ==========
            print(f"\n  📦 BACKGROUND DATA:")
            if existing['settings']:
                s = existing['settings']
                print(f"     Current: {s['background_type']} (size={s['background_size']})")
            
            change_bg = input("\n  ⚠️  Change background settings? This will DELETE ALL existing results!\n"
                            "      Type 'yes' to change (default=no): ").strip().lower()
            
            if change_bg == 'yes':
                # Show background options
                print(f"\n     Background Types: {BACKGROUND_TYPES}")
                bg_input = input(f"     --> Select [0-{len(BACKGROUND_TYPES)-1}] [0=random]: ").strip()
                try:
                    bg_type = BACKGROUND_TYPES[int(bg_input)] if bg_input else 'random'
                except:
                    bg_type = 'random'
                
                bg_size = int(input(f"     --> Background size [50]: ").strip() or 50)
                use_same = input(f"     --> Same background for noisy? (y/n) [y]: ").strip().lower() != 'n'
                
                # Confirm deletion
                print(f"\n  ⚠️  WARNING: This will delete {existing['total_results']} existing results!")
                confirm = input("      Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    results_del, samples_del, bg_del = clear_all_data(primary_use, option_number)
                    print(f"     🗑️ Deleted: {results_del} results, {samples_del} samples, {bg_del} backgrounds")
                    existing['n_samples'] = 0
                    existing['samples'] = []
                    existing['total_results'] = 0
                else:
                    print("     ❌ Cancelled - keeping existing background")
                    bg_type = existing['settings']['background_type']
                    bg_size = existing['settings']['background_size']
                    use_same = existing['settings']['use_same_for_noisy']
            else:
                bg_type = existing['settings']['background_type']
                bg_size = existing['settings']['background_size']
                use_same = existing['settings']['use_same_for_noisy']
            
            # ========== ASK ABOUT SAMPLES ==========
            print(f"\n  📌 TEST SAMPLES:")
            print(f"     Current: {existing['n_samples']} samples (indices: {existing['samples']})")
            
            sample_action = input("\n     Options: [enter]=keep, 'add'=add more, 'replace'=replace all\n"
                                 "     --> Choice [keep]: ").strip().lower()
            
            if sample_action == 'replace':
                print(f"\n  ⚠️  WARNING: This will delete {existing['total_results']} existing results!")
                confirm = input("      Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    clear_results_only(primary_use, option_number)
                    clear_test_samples(primary_use, option_number)
                    n_samples = int(input("     --> How many NEW samples? [10]: ").strip() or 10)
                    sample_mode = 'replace'
                    new_sample_count = n_samples
                else:
                    print("     ❌ Cancelled - keeping existing samples")
                    sample_mode = 'keep'
                    new_sample_count = 0
            
            elif sample_action == 'add':
                new_sample_count = int(input("     --> How many samples to ADD? [5]: ").strip() or 5)
                sample_mode = 'add'
            
            else:
                sample_mode = 'keep'
                new_sample_count = 0
            
            # ========== ASK ABOUT METHODS ==========
            print(f"\n  🔬 XAI METHODS:")
            for i, m in enumerate(available_methods):
                status = "✅" if m in existing['methods_with_results'] else "⭕"
                print(f"     {i}: {m} {status}")
            
            methods_input = input(f"\n     --> Select (comma-sep) or 'all' [all]: ").strip().lower()
            if methods_input in ['', 'all']:
                selected_methods = available_methods
            else:
                try:
                    selected_methods = [available_methods[int(x.strip())] for x in methods_input.split(',')]
                except:
                    selected_methods = available_methods
        
        else:
            # ========== NO EXISTING DATA - FIRST RUN ==========
            print(f"\n  🆕 NO EXISTING DATA - First time setup")
            
            # Background
            print(f"\n  📦 BACKGROUND DATA:")
            print(f"     Types: {BACKGROUND_TYPES}")
            bg_input = input(f"     --> Select [0-{len(BACKGROUND_TYPES)-1}] [0=random]: ").strip()
            try:
                bg_type = BACKGROUND_TYPES[int(bg_input)] if bg_input else 'random'
            except:
                bg_type = 'random'
            
            bg_size = int(input(f"     --> Background size [50]: ").strip() or 50)
            use_same = input(f"     --> Same background for noisy? (y/n) [y]: ").strip().lower() != 'n'
            
            # Samples
            print(f"\n  📌 TEST SAMPLES:")
            n_samples = int(input(f"     --> How many test samples? [10]: ").strip() or 10)
            sample_mode = 'new'
            new_sample_count = n_samples
            
            # Methods
            print(f"\n  🔬 XAI METHODS:")
            for i, m in enumerate(available_methods):
                print(f"     {i}: {m}")
            
            methods_input = input(f"\n     --> Select (comma-sep) or 'all' [all]: ").strip().lower()
            if methods_input in ['', 'all']:
                selected_methods = available_methods
            else:
                try:
                    selected_methods = [available_methods[int(x.strip())] for x in methods_input.split(',')]
                except:
                    selected_methods = available_methods
        
        all_configs.append({
            'primary_use': primary_use,
            'option_number': option_number,
            'models': models,
            'methods': selected_methods,
            'bg_type': bg_type,
            'bg_size': bg_size,
            'use_same_for_noisy': use_same,
            'sample_mode': sample_mode,
            'new_sample_count': new_sample_count,
            'existing_samples': existing['samples'] if existing['has_data'] else []
        })
    
    return all_configs


# ============================
# MAIN ANALYSIS
# ============================
def run_xai_analysis(configs):
    """Run XAI analysis for all configured datasets."""
    print("\n" + "="*80)
    print("🚀 STARTING XAI ANALYSIS")
    print("="*80)
    
    total_start = time.time()
    
    for cfg in configs:
        primary_use = cfg['primary_use']
        option_number = cfg['option_number']
        models = cfg['models']
        methods = cfg['methods']
        
        print(f"\n{'='*80}")
        print(f"📊 Processing: {primary_use} - Option {option_number}")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            container = load_dataset(primary_use, option_number)
            X_train, X_test = container.X_train, container.X_test
            seq_len, n_features = X_train.shape[1], X_train.shape[2]
            
            print(f"  Data: Train={X_train.shape}, Test={X_test.shape}")
            
            # Setup background
            existing_bg = get_background_data(primary_use, option_number)
            if existing_bg is None or cfg['sample_mode'] in ['replace', 'new']:
                print(f"  📦 Generating background ({cfg['bg_type']}, size={cfg['bg_size']})...")
                bg_orig = generate_background(X_train, cfg['bg_type'], cfg['bg_size'])
                bg_noisy = None if cfg['use_same_for_noisy'] else generate_background(X_train, cfg['bg_type'], cfg['bg_size'], RANDOM_SEED+1)
                save_background_data(primary_use, option_number, bg_orig, bg_noisy)
                save_settings(cfg['bg_type'], cfg['bg_size'], cfg['use_same_for_noisy'])
                bg_noisy = bg_noisy if bg_noisy is not None else bg_orig
            else:
                print(f"  📦 Using existing background")
                bg_orig = existing_bg['original']
                bg_noisy = existing_bg['noisy'] if existing_bg['noisy'] is not None else existing_bg['original']
            
            # Setup samples
            existing_samples = get_test_samples(primary_use, option_number)
            
            if cfg['sample_mode'] == 'new' or cfg['sample_mode'] == 'replace':
                print(f"  📌 Creating {cfg['new_sample_count']} new test samples...")
                np.random.seed(RANDOM_SEED)
                indices = np.random.choice(len(X_test), min(cfg['new_sample_count'], len(X_test)), replace=False).tolist()
                samples = {}
                for idx in indices:
                    orig = X_test[idx].astype(np.float32)
                    noisy = add_gaussian_noise(orig)
                    save_test_sample(primary_use, option_number, idx, orig, noisy)
                    samples[idx] = {'original': orig, 'noisy': noisy}
                print(f"     Created samples: {indices}")
            
            elif cfg['sample_mode'] == 'add':
                samples = existing_samples or {}
                available = [i for i in range(len(X_test)) if i not in samples]
                if available and cfg['new_sample_count'] > 0:
                    np.random.seed(int(time.time()))
                    new_indices = np.random.choice(available, min(cfg['new_sample_count'], len(available)), replace=False).tolist()
                    print(f"  📌 Adding {len(new_indices)} new samples: {new_indices}")
                    for idx in new_indices:
                        orig = X_test[idx].astype(np.float32)
                        noisy = add_gaussian_noise(orig)
                        save_test_sample(primary_use, option_number, idx, orig, noisy)
                        samples[idx] = {'original': orig, 'noisy': noisy}
            else:
                samples = existing_samples or {}
                print(f"  📌 Using existing {len(samples)} samples")
            
            if not samples:
                print("  ❌ No samples to process!")
                continue
            
            sample_indices = list(samples.keys())
            
            # Show initial progress
            print_progress_table(get_progress_summary(primary_use, option_number, models, methods))
            
            # Process each model
            for model_idx, model_name in enumerate(models):
                print(f"\n  🤖 MODEL [{model_idx+1}/{len(models)}]: {model_name}")
                
                try:
                    model, _ = load_trained_model(primary_use, option_number, model_name)
                    
                    # Process each method
                    for method_idx, method in enumerate(methods):
                        # Count items for this method
                        to_process = [(s, method) for s in sample_indices 
                                      if not check_result_exists(primary_use, option_number, model_name, s, method)]
                        
                        if not to_process:
                            print(f"     [{method_idx+1}/{len(methods)}] {method:<12} ✅ All {len(sample_indices)} samples done")
                            continue
                        
                        # Process all samples for this method (no per-sample output)
                        done = 0
                        failed = 0
                        method_start = time.time()
                        
                        for sample_idx, _ in to_process:
                            sample_data = samples[sample_idx]
                            
                            sample_start = time.time()
                            
                            shap_orig = compute_shap_values(
                                model, sample_data['original'], bg_orig, method,
                                seq_len, n_features,
                                primary_use=primary_use,
                                option_number=option_number,
                                model_name=model_name
                            )
                            
                            if shap_orig is None:
                                failed += 1
                                continue
                            
                            shap_noisy = compute_shap_values(
                                model, sample_data['noisy'], bg_noisy, method,
                                seq_len, n_features,
                                primary_use=primary_use,
                                option_number=option_number,
                                model_name=model_name
                            )
                            
                            computation_time = time.time() - sample_start
                            
                            # Pass bg_orig directly instead of using mean internally
                            fidelity = compute_fidelity(model, sample_data['original'], shap_orig, bg_orig, seq_len, n_features)
                            sparsity = compute_sparsity(shap_orig)
                            complexity = compute_complexity(shap_orig)
                            reliability = compute_reliability_metrics(shap_orig, shap_noisy)
                            efficiency_error = compute_efficiency_error(model, sample_data['original'], shap_orig, bg_orig, seq_len, n_features)

                            save_xai_result(primary_use, option_number, model_name, sample_idx, method,
                                fidelity, sparsity, complexity, reliability, efficiency_error, computation_time, shap_orig, shap_noisy)

                            
                            done += 1
                        
                        method_time = time.time() - method_start
                        status = "✅" if failed == 0 else f"⚠️ {failed} failed"
                        print(f"     [{method_idx+1}/{len(methods)}] {method:<12} {done}/{len(to_process)} done in {method_time:.1f}s {status}")
                    
                    del model
                    torch.cuda.empty_cache()
                    
                    # Show progress after each model
                    print_progress_table(get_progress_summary(primary_use, option_number, models, methods))
                
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"  ❌ Dataset error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"✅ COMPLETED in {time.time() - total_start:.1f}s")
    print(f"{'='*80}")

# ============================
# SUMMARY
# ============================
def print_summary(db_path=XAI_DB):
    """Print comprehensive summary."""
    conn = sqlite3.connect(db_path)
    
    print("\n" + "="*90)
    print("📊 XAI RESULTS SUMMARY")
    print("="*90)
    
    # Settings
    row = conn.execute('SELECT * FROM settings WHERE id = 1').fetchone()
    if row:
        print(f"\n  ⚙️ Settings: bg={row[2]}(size={row[3]}), same_noisy={bool(row[4])}")
    
    # Total
    total = conn.execute('SELECT COUNT(*) FROM xai_results').fetchone()[0]
    print(f"\n  Total results: {total}")
    
    # By method
    df = pd.read_sql_query('''
        SELECT xai_method, COUNT(*) as n, AVG(fidelity) as fid, AVG(sparsity) as spa,
               AVG(reliability_correlation) as rel, AVG(efficiency_error) as eff, AVG(computation_time) as t
        FROM xai_results GROUP BY xai_method ORDER BY xai_method
    ''', conn)
    
    if len(df) > 0:
        print(f"\n  {'Method':<12} {'Count':>6} {'Fidelity':>10} {'Sparsity':>10} {'Reliab':>10} {'Effic':>10} {'Time':>8}")
        print(f"  {'-'*68}")
        for _, r in df.iterrows():
            fid = f"{r['fid']:.4f}" if r['fid'] else "N/A"
            spa = f"{r['spa']:.1f}%" if r['spa'] else "N/A"
            rel = f"{r['rel']:.4f}" if r['rel'] else "N/A"
            eff = f"{r['eff']:.4f}" if r['eff'] else "N/A"
            t = f"{r['t']:.2f}s" if r['t'] else "N/A"
            print(f"  {r['xai_method']:<12} {r['n']:>6} {fid:>10} {spa:>10} {rel:>10} {eff:>10} {t:>8}")
    
    conn.close()
    print("\n" + "="*90)


# ============================
# MAIN
# ============================
def main():
    """Main entry point."""
    init_database()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == 'summary':
            print_summary()
        elif cmd in ['--help', '-h']:
            print(f"\nUsage: python xai.py [summary|--help]")
            print(f"Methods: {', '.join(XAI_METHODS)}")
        else:
            print(f"Unknown: {cmd}")
    else:
        configs = get_user_inputs()
        if configs:
            confirm = input("\n--> Proceed? (y/n) [y]: ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                run_xai_analysis(configs)
                print_summary()


if __name__ == "__main__":
    main()