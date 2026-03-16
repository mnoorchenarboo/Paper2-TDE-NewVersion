# ============================
# LIBRARY IMPORTS
# ============================
import os
import sys
import json
import time
import shutil
import sqlite3
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import optuna
from optuna.samplers import TPESampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap

from dl import load_complete_model

import shap

from dl import load_complete_model
from Functions.tde_class import TemporalDeepExplainer, FastSHAPExplainer

# Execution settings
n_jobs = 1

# ============================
# CONFIGURATION
# ============================
from pathlib import Path

PATH_DBS = Path("databases")
PATH_DBS.mkdir(parents=True, exist_ok=True)

BENCHMARK_DB = PATH_DBS / "benchmark_results.db"
EXPLAINER_DB = PATH_DBS / "explainer_results.db"
ENERGY_DB    = PATH_DBS / "energy_data.db"

RESULTS_BASE_DIR = "results"

# Standardized method names
METHOD_NAMES = {
    'tde': 'TDE', 'fastshap': 'Fast_SHAP', 'gradient_shap': 'Gradient_SHAP',
    'deep_shap': 'Deep_SHAP', 'kernel_shap': 'Kernel_SHAP', 
    'permutation_shap': 'Permutation_SHAP', 'partition_shap': 'Partition_SHAP',
    'lime': 'LIME', 'sampling_shap': 'Sampling_SHAP'
}
METHOD_KEYS = {v: k for k, v in METHOD_NAMES.items()}
NEURAL_EXPLAINER_TYPES = ['tde', 'fastshap']

# Training configuration
DEBUG_MODE = True
DEBUG_TRAINING_FRACTION, DEBUG_TRIAL_EPOCHS, DEBUG_FINAL_EPOCHS, DEBUG_N_TRIALS = 0.20, 10, 50, 5
PROD_TRAINING_FRACTION, PROD_TRIAL_EPOCHS, PROD_FINAL_EPOCHS, PROD_N_TRIALS = 0.30, 20, 100, 30

PAIRED_SAMPLING = True
VALIDATION_SPLIT = 0.20
NOISE_STD = 0.01
EARLY_STOP_PATIENCE = 5
TARGET_SPARSITY = 0.70

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}")


# ============================
# MODEL WRAPPER & TRADITIONAL SHAP
# ============================
class SingleHorizonWrapper(nn.Module):
    """Wrapper to extract single prediction horizon from multi-horizon models."""
    def __init__(self, base_model, horizon_idx=0):
        super().__init__()
        self.base_model, self.horizon_idx = base_model, horizon_idx
    
    def forward(self, x):
        out = self.base_model(x)
        return out[:, self.horizon_idx:self.horizon_idx + 1] if out.ndim > 1 and out.shape[1] > self.horizon_idx else out

class TraditionalSHAPMethods:
    """Wrapper for traditional SHAP methods (Gradient SHAP, Deep SHAP)."""
    def __init__(self, model, background, time_steps, n_features, dev=device):
        self.device, self.time_steps, self.n_features = dev, time_steps, n_features
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(dev)
        self.wrapped_model.eval()
        self.background_tensor = torch.FloatTensor(background).to(dev)
    
    def _compute_shap(self, instance, explainer_cls, **kwargs):
        try:
            if isinstance(instance, np.ndarray): instance = torch.FloatTensor(instance)
            if instance.ndim == 2: instance = instance.unsqueeze(0)
            instance = instance.to(self.device)
            shap_vals = explainer_cls(self.wrapped_model, self.background_tensor).shap_values(instance, **kwargs)
            if isinstance(shap_vals, list): shap_vals = shap_vals[0]
            if isinstance(shap_vals, torch.Tensor): shap_vals = shap_vals.cpu().numpy()
            shap_vals = np.array(shap_vals)
            if shap_vals.ndim == 3 and shap_vals.shape[0] == 1: shap_vals = shap_vals[0]
            if shap_vals.shape == (self.time_steps, self.n_features): return shap_vals
            if shap_vals.size == self.time_steps * self.n_features: return shap_vals.reshape(self.time_steps, self.n_features)
            return None
        except Exception: return None
    
    def gradient_shap(self, instance): return self._compute_shap(instance, shap.GradientExplainer)
    def deep_shap(self, instance): return self._compute_shap(instance, shap.DeepExplainer, check_additivity=False)


# ============================
# HELPER FUNCTIONS
# ============================
def get_standard_method_name(method_key):
    """Convert method identifier to standardized display name."""
    return METHOD_NAMES.get(method_key.lower().replace('-', '_').replace(' ', '_'), method_key)

def get_method_key(method_name):
    """Convert standardized method name to lowercase key."""
    if method_name.lower() in METHOD_NAMES:
        return method_name.lower()
    return METHOD_KEYS.get(method_name, method_name.lower())

def is_neural_explainer(method_name):
    """Check if method is a neural explainer (TDE or FastSHAP)."""
    return get_method_key(method_name) in NEURAL_EXPLAINER_TYPES

def setup_logger(log_path):
    """Setup logger for training progress."""
    logger = logging.getLogger('Explainer')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in [logging.FileHandler(log_path, mode='w', encoding='utf-8'), 
                    logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger


# ============================
# DATABASE FUNCTIONS (CONSOLIDATED)
# ============================
def init_database():
    """Initialize SQLite database for storing explainer results."""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    
    for table, schema in [
        ('tde_hyperparameter_trials', '''primary_use TEXT, option_number INTEGER, model_name TEXT,
            trial_number INTEGER, hyperparameters TEXT, validation_loss REAL, n_training_samples INTEGER,
            timestamp TEXT, PRIMARY KEY (primary_use, option_number, model_name, trial_number)'''),
        ('fastshap_hyperparameter_trials', '''primary_use TEXT, option_number INTEGER, model_name TEXT,
            trial_number INTEGER, hyperparameters TEXT, validation_loss REAL, n_training_samples INTEGER,
            timestamp TEXT, PRIMARY KEY (primary_use, option_number, model_name, trial_number)'''),
        ('explainer_metadata', '''primary_use TEXT, option_number INTEGER, model_name TEXT, explainer_type TEXT,
            best_hyperparameters TEXT, best_validation_loss REAL, final_training_loss REAL, n_training_samples INTEGER,
            time_steps INTEGER, n_features INTEGER, optimization_time REAL, training_time REAL, n_trials INTEGER,
            explainer_path TEXT, feature_names TEXT, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, explainer_type)'''),
        ('comparison_results', '''primary_use TEXT, option_number INTEGER, model_name TEXT, sample_idx INTEGER,
            method TEXT, fidelity REAL, reliability_correlation REAL, reliability_mse REAL, sparsity REAL,
            complexity REAL, efficiency_error REAL, computation_time REAL, timestamp TEXT,
            PRIMARY KEY (primary_use, option_number, model_name, sample_idx, method)''')
    ]:
        cursor.execute(f'CREATE TABLE IF NOT EXISTS {table} ({schema})')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {EXPLAINER_DB}")

def get_optuna_db_path(primary_use, option_number, model_name, explainer_type):
    """Get path for Optuna study database."""
    optuna_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / get_method_key(explainer_type)
    optuna_dir.mkdir(parents=True, exist_ok=True)
    return str(optuna_dir / "optuna_study.db")

def save_hyperparameter_trial(explainer_type, primary_use, option_number, model_name, trial_num, params, loss, n_train):
    """Save hyperparameter trial to database (unified for TDE and FastSHAP)."""
    table = 'tde_hyperparameter_trials' if get_method_key(explainer_type) == 'tde' else 'fastshap_hyperparameter_trials'
    conn = sqlite3.connect(EXPLAINER_DB)
    conn.execute(f'INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                 (primary_use, option_number, model_name, trial_num, json.dumps(params), loss, n_train, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_explainer_metadata(primary_use, option_number, model_name, explainer_type, best_params, best_loss,
                            final_loss, n_train, time_steps, n_features, opt_time, train_time, n_trials, path, feature_names):
    """Save explainer metadata to database."""
    conn = sqlite3.connect(EXPLAINER_DB)
    conn.execute('INSERT OR REPLACE INTO explainer_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                 (primary_use, option_number, model_name, get_standard_method_name(explainer_type),
                  json.dumps(best_params), best_loss, final_loss, n_train, time_steps, n_features,
                  opt_time, train_time, n_trials, path, json.dumps(feature_names), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_comparison(primary_use, option_number, model_name, sample_idx, method,
                    fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time):
    """Save comparison results to database."""
    conn = sqlite3.connect(EXPLAINER_DB)
    conn.execute('INSERT OR REPLACE INTO comparison_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                 (primary_use, option_number, model_name, sample_idx, get_standard_method_name(method),
                  fidelity, rel_corr, rel_mse, sparsity, complexity, eff_err, comp_time, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_explainer_metadata(primary_use, option_number, model_name, explainer_type):
    """Get explainer metadata from database."""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    cursor.execute('''SELECT best_hyperparameters, best_validation_loss, final_training_loss, n_training_samples,
                      time_steps, n_features, optimization_time, training_time, n_trials, explainer_path, feature_names, timestamp
                      FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?''',
                   (primary_use, option_number, model_name, get_standard_method_name(explainer_type)))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {'best_hyperparameters': json.loads(row[0]) if row[0] else None, 'best_validation_loss': row[1],
                'final_training_loss': row[2], 'n_training_samples': row[3], 'time_steps': row[4], 'n_features': row[5],
                'optimization_time': row[6], 'training_time': row[7], 'n_trials': row[8], 'explainer_path': row[9],
                'feature_names': json.loads(row[10]) if row[10] else None, 'timestamp': row[11]}
    return None

def get_comparison_results(primary_use, option_number, model_name, method=None):
    """Get comparison results from database."""
    conn = sqlite3.connect(EXPLAINER_DB)
    cursor = conn.cursor()
    
    query = '''SELECT sample_idx, method, fidelity, reliability_correlation, reliability_mse,
               sparsity, complexity, efficiency_error, computation_time, timestamp
               FROM comparison_results WHERE primary_use=? AND option_number=? AND model_name=?'''
    params = [primary_use, option_number, model_name]
    
    if method:
        query += ' AND method=?'
        params.append(get_standard_method_name(method))
    query += ' ORDER BY method, sample_idx'
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [{'sample_idx': r[0], 'method': r[1], 'fidelity': r[2], 'reliability_correlation': r[3],
             'reliability_mse': r[4], 'sparsity': r[5], 'complexity': r[6], 'efficiency_error': r[7],
             'computation_time': r[8], 'timestamp': r[9]} for r in rows]

def check_explainer_exists(primary_use, option_number, model_name, explainer_type):
    """
    Check explainer status and classify as: complete, partial, or fresh.
    
    Returns dict with:
        - status: 'complete' | 'partial' | 'fresh'
        - db_metadata: bool (explainer_metadata table)
        - db_trials: bool (hyperparameter_trials table)
        - db_comparison: bool (comparison_results table)
        - disk_explainer: bool (explainer .pt file)
        - disk_optuna: bool (optuna study db)
        - disk_logs: bool (log files exist)
        - comparison_count: int
        - details: list of what exists
    """
    standard_type = get_standard_method_name(explainer_type)
    type_key = get_method_key(explainer_type)
    
    result = {
        'db_metadata': False, 'db_trials': False, 'db_comparison': False,
        'disk_explainer': False, 'disk_optuna': False, 'disk_logs': False,
        'comparison_count': 0, 'details': []
    }
    
    # Check database tables
    try:
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        # Check metadata
        cursor.execute('SELECT COUNT(*) FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?',
                       (primary_use, option_number, model_name, standard_type))
        result['db_metadata'] = cursor.fetchone()[0] > 0
        
        # Check trials
        trials_table = 'tde_hyperparameter_trials' if type_key == 'tde' else 'fastshap_hyperparameter_trials'
        cursor.execute(f'SELECT COUNT(*) FROM {trials_table} WHERE primary_use=? AND option_number=? AND model_name=?',
                       (primary_use, option_number, model_name))
        result['db_trials'] = cursor.fetchone()[0] > 0
        
        # Check comparison
        cursor.execute('SELECT COUNT(*) FROM comparison_results WHERE primary_use=? AND option_number=? AND model_name=? AND method=?',
                       (primary_use, option_number, model_name, standard_type))
        result['comparison_count'] = cursor.fetchone()[0]
        result['db_comparison'] = result['comparison_count'] > 0
        
        conn.close()
    except Exception:
        pass
    
    # Check disk files
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    exp_dir = model_dir / type_key
    
    result['disk_explainer'] = (exp_dir / f"{type_key}_explainer.pt").exists()
    result['disk_optuna'] = (exp_dir / "optuna_study.db").exists()
    result['disk_logs'] = any(model_dir.glob("*.log")) if model_dir.exists() else False
    
    # Build details list
    if result['db_metadata']: result['details'].append('db:metadata')
    if result['db_trials']: result['details'].append('db:trials')
    if result['db_comparison']: result['details'].append(f"db:comparison({result['comparison_count']})")
    if result['disk_explainer']: result['details'].append('disk:explainer')
    if result['disk_optuna']: result['details'].append('disk:optuna')
    if result['disk_logs']: result['details'].append('disk:logs')
    
    # Classify status
    all_complete = (result['db_metadata'] and result['db_comparison'] and result['disk_explainer'])
    nothing_exists = not any([result['db_metadata'], result['db_trials'], result['db_comparison'],
                              result['disk_explainer'], result['disk_optuna']])
    
    if all_complete:
        result['status'] = 'complete'
    elif nothing_exists:
        result['status'] = 'fresh'
    else:
        result['status'] = 'partial'
    
    result['path'] = str(exp_dir) if result['disk_explainer'] else None
    
    return result

def get_incomplete_items(primary_use, option_number, models, explainer_types):
    """
    Check what's incomplete and return items needing processing.
    
    Returns:
        items_to_process: dict {model_name: [list of exp_types to train]}
        status_info: dict with 'complete', 'partial', 'fresh' lists and 'details'
    """
    items_to_process = {}
    status_info = {
        'complete': [],
        'partial': [],
        'fresh': [],
        'details': {}
    }
    
    for model_name in models:
        items_to_process[model_name] = []
        status_info['details'][model_name] = {}
        
        for exp_type in explainer_types:
            result = check_explainer_exists(primary_use, option_number, model_name, exp_type)
            
            # Store detailed status info
            status_info['details'][model_name][exp_type] = {
                'status': result['status'],
                'details': result['details'],
                'complete': result['status'] == 'complete',
                'db_metadata': result['db_metadata'],
                'db_trials': result['db_trials'],
                'db_comparison': result['db_comparison'],
                'disk_explainer': result['disk_explainer'],
                'disk_optuna': result['disk_optuna'],
                'disk_logs': result['disk_logs'],
                'comparison_count': result['comparison_count'],
                'path': result['path']
            }
            
            key = f"{model_name}/{get_standard_method_name(exp_type)}"
            
            if result['status'] == 'complete':
                status_info['complete'].append(key)
            elif result['status'] == 'partial':
                status_info['partial'].append(key)
                items_to_process[model_name].append(exp_type)
            else:  # 'fresh'
                status_info['fresh'].append(key)
                items_to_process[model_name].append(exp_type)
    
    return items_to_process, status_info

# ============================
# LOADING FUNCTIONS (CONSOLIDATED)
# ============================
def load_explainer_for_inference(primary_use, option_number, model_name, explainer_type, model_predict_func=None, device_override=None):
    """Load TDE or FastSHAP explainer for inference (unified function)."""
    dev = device_override or device
    type_key = get_method_key(explainer_type)
    
    path = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / type_key
    if not path.exists():
        raise FileNotFoundError(f"{type_key.upper()} directory not found: {path}")
    
    explainer_file = path / f"{type_key}_explainer.pt"
    if not explainer_file.exists():
        raise FileNotFoundError(f"{type_key.upper()} explainer file not found: {explainer_file}")
    
    try:
        ExplainerClass = TemporalDeepExplainer if type_key == 'tde' else FastSHAPExplainer
        explainer = ExplainerClass.load(str(path), filename=f"{type_key}_explainer", device_override=dev)
        if model_predict_func:
            explainer.model_predict_func = model_predict_func
        return explainer
    except Exception as e:
        raise ValueError(f"Failed to load {type_key.upper()} explainer: {e}")

def load_explainer_with_model(primary_use, option_number, model_name, explainer_type='tde', device_override=None):
    """Load both model and explainer together with automatic setup."""
    dev = device_override or device
    explainer_type_lower = explainer_type.lower()
    
    # Load model
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    config_path = model_dir / "model_metadata.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    model = load_complete_model(str(model_path), device=dev)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    time_steps, n_features = config['seq_length'], config['n_features']
    
    def predict_first_horizon(X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if X.ndim == 2:
            X = X.reshape(-1, time_steps, n_features)
        X_t = torch.tensor(X, dtype=torch.float32, device=dev)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    explainer = load_explainer_for_inference(primary_use, option_number, model_name, explainer_type_lower, predict_first_horizon, dev)
    
    return model, explainer, predict_first_horizon, {
        'primary_use': primary_use, 'option_number': option_number, 'model_name': model_name,
        'explainer_type': get_standard_method_name(explainer_type), 'time_steps': time_steps, 'n_features': n_features,
        'prediction_horizon': config.get('prediction_horizon', 1),
        'feature_names': explainer.feature_names if hasattr(explainer, 'feature_names') else None,
        'model_path': str(model_path), 'explainer_path': str(model_dir / explainer_type_lower), 'device': str(dev)
    }

def get_all_trained_explainers(primary_use, option_number, model_name):
    """Get information about all trained explainers for a specific model."""
    result = {}
    for exp_type_lower in ['tde', 'fastshap']:
        exp_type_standard = METHOD_NAMES.get(exp_type_lower)
        exp_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / exp_type_lower
        exp_file = exp_dir / f"{exp_type_lower}_explainer.pt"
        exists = exp_file.exists()
        
        metadata = None
        if exists:
            try:
                conn = sqlite3.connect(EXPLAINER_DB)
                cursor = conn.cursor()
                cursor.execute('''SELECT best_hyperparameters, best_validation_loss, n_training_samples, time_steps, n_features, timestamp
                                  FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?''',
                               (primary_use, option_number, model_name, exp_type_standard))
                row = cursor.fetchone()
                conn.close()
                if row:
                    metadata = {'best_hyperparameters': json.loads(row[0]) if row[0] else None,
                                'best_validation_loss': row[1], 'n_training_samples': row[2],
                                'time_steps': row[3], 'n_features': row[4], 'timestamp': row[5]}
            except Exception:
                pass
        
        result[exp_type_standard] = {'exists': exists, 'path': str(exp_dir) if exists else None,
                                     'file': str(exp_file) if exists else None, 'metadata': metadata}
    return result

def list_all_available_explainers():
    """List all available explainers in the results directory."""
    available = []
    results_path = Path(RESULTS_BASE_DIR)
    if not results_path.exists():
        return available
    
    for primary_use_dir in results_path.iterdir():
        if not primary_use_dir.is_dir():
            continue
        for option_dir in primary_use_dir.iterdir():
            if not option_dir.is_dir() or not option_dir.name.startswith('option_'):
                continue
            try:
                option_number = int(option_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            for model_dir in option_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                for exp_type in ['tde', 'fastshap']:
                    if (model_dir / exp_type / f"{exp_type}_explainer.pt").exists():
                        available.append({'primary_use': primary_use_dir.name, 'option_number': option_number,
                                          'model_name': model_dir.name.upper(), 'explainer_type': METHOD_NAMES[exp_type],
                                          'path': str(model_dir / exp_type)})
    return available


# ============================
# EXPLAINABILITY METRICS
# ============================
def compute_topk_feature_overlap(shap_vals_orig, shap_vals_pert, top_k_pct=10):
    """Compute overlap ratio of top-K important features between original and perturbed."""
    if shap_vals_orig is None or shap_vals_pert is None:
        return None
    try:
        flat_orig, flat_pert = np.abs(shap_vals_orig).flatten(), np.abs(shap_vals_pert).flatten()
        k = max(1, int(len(flat_orig) * top_k_pct / 100))
        return float(len(set(np.argsort(flat_orig)[-k:]) & set(np.argsort(flat_pert)[-k:])) / k)
    except Exception:
        return None

class ExplainabilityMetrics:
    """Compute various explainability metrics for SHAP values."""
    
    def __init__(self, model, baseline, base_pred, time_steps, n_features, dev=device):
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(dev)
        self.wrapped_model.eval()
        self.baseline, self.base_pred = baseline, base_pred
        self.time_steps, self.n_features, self.device = time_steps, n_features, dev
    
    def _get_prediction(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.wrapped_model(x.to(self.device)).cpu().numpy().flatten()[0]
    
    def _prepare_instance(self, instance):
        if isinstance(instance, torch.Tensor):
            instance = instance.cpu().numpy()
        return instance[0] if instance.ndim == 3 else instance
    
    def fidelity(self, instance, shap_vals, top_k_pct=10):
        """Fidelity: prediction change when top-k features masked."""
        if shap_vals is None:
            return None
        instance = self._prepare_instance(instance)
        baseline = self.baseline.cpu().numpy() if isinstance(self.baseline, torch.Tensor) else self.baseline
        baseline = baseline[0] if baseline.ndim == 3 else baseline
        
        orig_pred = self._get_prediction(instance)
        k = max(1, int(np.abs(shap_vals).size * top_k_pct / 100))
        top_k_idx = np.argsort(np.abs(shap_vals).flatten())[-k:]
        
        masked = instance.copy()
        for idx in top_k_idx:
            masked[idx // self.n_features, idx % self.n_features] = baseline[idx // self.n_features, idx % self.n_features]
        
        return float(abs(orig_pred - self._get_prediction(masked)))
    
    def reliability(self, instance, shap_vals, shap_func, noise_std=NOISE_STD):
        """Reliability: SHAP stability under small perturbations."""
        if shap_vals is None:
            return None, None, None
        instance = self._prepare_instance(instance)
        perturbed = np.clip(instance + np.random.normal(0, noise_std, instance.shape), 0, 1).astype(np.float32)
        shap_pert = shap_func(perturbed)
        if shap_pert is None:
            return None, None, None
        
        orig, pert = shap_vals.flatten(), shap_pert.flatten()
        mask = np.isfinite(orig) & np.isfinite(pert)
        if np.sum(mask) < 10:
            return None, None, None
        
        try:
            corr = float(pearsonr(orig[mask], pert[mask])[0])
            corr = corr if np.isfinite(corr) else None
        except Exception:
            corr = None
        
        return corr, float(np.mean((orig[mask] - pert[mask]) ** 2)), compute_topk_feature_overlap(shap_vals, shap_pert)
    
    def sparsity(self, shap_vals, threshold_pct=1):
        """Sparsity: fraction of near-zero SHAP values."""
        if shap_vals is None:
            return None
        abs_shap, max_val = np.abs(shap_vals), np.max(np.abs(shap_vals))
        return 100.0 if max_val == 0 else float(np.sum(abs_shap < max_val * threshold_pct / 100) / abs_shap.size * 100)
    
    def complexity(self, shap_vals):
        """Complexity: entropy of SHAP distribution."""
        if shap_vals is None:
            return None
        abs_shap = np.abs(shap_vals).flatten() + 1e-10
        probs = abs_shap / np.sum(abs_shap)
        return float(-np.sum(probs * np.log(probs)))
    
    def efficiency_error(self, instance, shap_vals):
        """Efficiency Error: how well SHAP values sum to prediction difference."""
        if shap_vals is None:
            return None
        instance = self._prepare_instance(instance)
        expected = self._get_prediction(instance) - self.base_pred
        return abs(np.sum(shap_vals) - expected) / (abs(expected) + 1e-10)


# ============================
# HYPERPARAMETER OPTIMIZATION
# ============================
def create_tde_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    """Create Optuna objective for TDE hyperparameter optimization."""
    def objective(trial):
        params = {
            'window_size': trial.suggest_categorical('window_size', [3, 6, 12, 24]),
            'l1_lambda': trial.suggest_float('l1_lambda', 0.0001, 1.0, log=True),
            'l2_lambda': trial.suggest_float('l2_lambda', 0.0001, 0.1, log=True),
            'smoothness_lambda': trial.suggest_float('smoothness_lambda', 0.001, 0.3),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'sparsity_lambda': trial.suggest_float('sparsity_lambda', 0.01, 0.5),
            'target_sparsity': trial.suggest_float('target_sparsity', 0.50, 0.80),
            'sparsity_threshold': trial.suggest_float('sparsity_threshold', 0.001, 0.05, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_conv_layers': trial.suggest_int('n_conv_layers', 1, 3),
            'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
            'n_attention_heads': trial.suggest_categorical('n_attention_heads', [2, 4, 8]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'masking_mode': trial.suggest_categorical('masking_mode', ['window', 'feature']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 4),
        }
        try:
            tde = TemporalDeepExplainer(n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False, paired_sampling=True, **params)
            val_loss = tde.train(X_train, X_val, model_predict_func, feature_names)
            del tde
            torch.cuda.empty_cache()
            return val_loss
        except Exception:
            return float('inf')
    return objective

def create_fastshap_objective(X_train, X_val, model_predict_func, feature_names, n_epochs):
    """Create Optuna objective for FastSHAP hyperparameter optimization."""
    def objective(trial):
        params = {
            'l1_lambda': trial.suggest_float('l1_lambda', 0.001, 0.3, log=True),
            'efficiency_lambda': trial.suggest_float('efficiency_lambda', 0.05, 0.5),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True),
            'optimizer_type': trial.suggest_categorical('optimizer_type', ['adam', 'adamw']),
            'samples_per_feature': trial.suggest_int('samples_per_feature', 2, 4),
        }
        try:
            fs = FastSHAPExplainer(n_epochs=n_epochs, patience=EARLY_STOP_PATIENCE, verbose=False, paired_sampling=True, **params)
            val_loss = fs.train(X_train, X_val, model_predict_func, feature_names, gpu_model=None)
            del fs
            torch.cuda.empty_cache()
            return val_loss if val_loss != float('inf') else float('inf')
        except Exception:
            return float('inf')
    return objective

def run_optimization(explainer_type, X_train, X_val, model_predict_func, feature_names,
                     n_trials, n_epochs, primary_use, option_number, model_name):
    """Run hyperparameter optimization."""
    optuna_db = get_optuna_db_path(primary_use, option_number, model_name, explainer_type)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42),
                                study_name=f"{explainer_type}_{model_name}", 
                                storage=f"sqlite:///{optuna_db}", load_if_exists=True)
    
    objective_fn = create_tde_objective if explainer_type == 'tde' else create_fastshap_objective
    objective = objective_fn(X_train, X_val, model_predict_func, feature_names, n_epochs)
    callback = lambda s, t: save_hyperparameter_trial(explainer_type, primary_use, option_number, model_name, t.number, t.params, t.value, len(X_train))
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=True, n_jobs=n_jobs)
    return study, time.time() - start


# ============================
# VISUALIZATION
# ============================
def plot_convergence(history, save_path, title="Convergence"):
    """Plot training convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', alpha=0.7)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', alpha=0.7)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title(title)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['lr'], 'orange', alpha=0.7)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Learning Rate'); axes[1].set_yscale('log'); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics_data, save_path, title="XAI Methods Comparison"):
    """Plot comprehensive comparison of all XAI methods."""
    methods = [m for m in metrics_data.keys() if metrics_data[m] and any(
        len(metrics_data[m].get(k, [])) > 0 for k in ['fidelity', 'reliability', 'sparsity'])]
    if not methods:
        return False
    
    color_map = {'TDE': '#2ecc71', 'Fast_SHAP': '#3498db', 'Gradient_SHAP': '#e74c3c',
                 'Deep_SHAP': '#9b59b6', 'Kernel_SHAP': '#f39c12', 'Permutation_SHAP': '#1abc9c'}
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    plot_configs = [
        {'key': 'fidelity', 'title': 'Fidelity (Higher is Better)', 'ylabel': 'Fidelity Score', 'row': 0, 'col': 0},
        {'key': 'sparsity', 'title': 'Sparsity (Higher = Simpler)', 'ylabel': 'Sparsity (%)', 'row': 0, 'col': 1},
        {'key': 'complexity', 'title': 'Complexity (Lower is Better)', 'ylabel': 'Entropy', 'row': 1, 'col': 0},
        {'key': 'reliability', 'title': 'Reliability - Correlation (Higher is Better)', 'ylabel': 'Pearson Correlation', 'row': 1, 'col': 1},
        {'key': 'efficiency', 'title': 'Efficiency Error (Lower is Better)', 'ylabel': 'Relative Error', 'row': 2, 'col': 0},
        {'key': 'time', 'title': 'Inference Time (Lower is Better)', 'ylabel': 'Time (ms)', 'row': 2, 'col': 1, 'scale': 1000},
    ]
    
    for config in plot_configs:
        ax = axes[config['row'], config['col']]
        scale = config.get('scale', 1)
        means, stds, colors, labels = [], [], [], []
        
        for method in methods:
            vals = [v * scale for v in metrics_data[method].get(config['key'], []) if v is not None and np.isfinite(v)]
            if vals:
                means.append(np.mean(vals)); stds.append(np.std(vals))
                colors.append(color_map.get(method, '#95a5a6')); labels.append(method.replace('_', '\n'))
        
        if means:
            x_pos = np.arange(len(means))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
            for bar, mean, std in zip(bars, means, stds):
                ax.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + std),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=9)
        
        ax.set_title(config['title'], fontsize=11, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=10); ax.grid(True, alpha=0.3, axis='y'); ax.set_axisbelow(True)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    return True

def generate_shap_heatmap_pdf(shap_values, sample_data, feature_names, output_path, xai_method,
                               metrics=None, show_title=True, font_size=None, colorbar_label="SHAP Value"):
    """Generate SHAP heatmap and save as PDF."""
    try:
        # Parse SHAP values
        if isinstance(shap_values, str):
            shap_values = json.loads(shap_values)
        elif isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        
        if isinstance(shap_values, dict):
            shap_vals = np.array(shap_values.get('values', list(shap_values.values())), dtype=float)
            if shap_vals.ndim == 1 and 'shape' in shap_values:
                shap_vals = shap_vals.reshape(shap_values['shape'])
        else:
            shap_vals = np.array(shap_values, dtype=float)
        
        if shap_vals.ndim > 2:
            shap_vals = shap_vals.reshape(-1, shap_vals.shape[-1])
        if shap_vals.size == 0:
            return False
        shap_vals = np.nan_to_num(shap_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Parse sample data
        if isinstance(sample_data, str):
            sample_data = json.loads(sample_data)
        elif isinstance(sample_data, torch.Tensor):
            sample_data = sample_data.cpu().numpy()
        sample_np = np.array(sample_data, dtype=float)
        if sample_np.ndim > 2:
            sample_np = sample_np.reshape(-1, sample_np.shape[-1])
        if sample_np.shape != shap_vals.shape:
            sample_np = sample_np.reshape(shap_vals.shape) if sample_np.size == shap_vals.size else np.zeros_like(shap_vals)
        
        # Parse feature names
        if isinstance(feature_names, str):
            feature_names = json.loads(feature_names)
        feature_names = list(feature_names) if feature_names else []
        n_features = shap_vals.shape[1]
        feature_names = feature_names[:n_features] + [f"Feature_{i}" for i in range(len(feature_names), n_features)]
        
        # Create SHAP Explanation and plot
        shap_exp = shap.Explanation(values=shap_vals, base_values=np.zeros(shap_vals.shape[0]),
                                    data=sample_np, feature_names=feature_names)
        
        fig = plt.figure(figsize=(16, 8))
        shap.plots.heatmap(shap_exp, show=False, max_display=n_features)
        plt.tight_layout()
        
        fmt = 'pdf' if Path(output_path).suffix.lower() == '.pdf' else 'png'
        plt.savefig(str(output_path), format=fmt, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"      ✗ Heatmap error for {xai_method}: {e}")
        plt.close('all')
        return False


# ============================
# DATA LOADING
# ============================
def load_dataset(primary_use, option_number):
    """Load dataset using preprocess function."""
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        db_path=ENERGY_DB, primary_use=primary_use, option_number=option_number, scaled=True, scale_type="both")

def get_datasets():
    """Get all available datasets from benchmark database."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('SELECT DISTINCT primary_use, option_number FROM prediction_performance ORDER BY primary_use, option_number', conn)
    conn.close()
    return [{'primary_use': r['primary_use'], 'option_number': int(r['option_number'])} for _, r in df.iterrows()]

def get_models(primary_use, option_number):
    """Get all available models for a dataset."""
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('SELECT DISTINCT model_name FROM prediction_performance WHERE primary_use = ? AND option_number = ?',
                           conn, params=(primary_use, option_number))
    conn.close()
    return df['model_name'].tolist()


# ============================
# MAIN TRAINING FUNCTION
# ============================
def make_explainer_func(explainer):
    """Create explainer function with proper closure."""
    return lambda x: explainer.explain(x)

def train_and_compare(primary_use, option_number, model_name, container, explainer_types_to_train,
                      n_trials, n_test_samples, logger, training_fraction=None):
    """Train explainers and compare with traditional SHAP methods."""
    logger.info(f"\n[MODEL] {model_name}")
    
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        logger.error(f"  Model not found: {model_path}")
        return None, None
    
    model = load_complete_model(str(model_path), device=device)
    time_steps, n_features = container.X_train.shape[1], container.X_train.shape[2]
    
    def predict_first_horizon(X):
        if X.ndim == 2:
            X = X.reshape(-1, time_steps, n_features)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    # Select training samples
    X_all = np.concatenate([container.X_train, container.X_val], axis=0)
    frac = training_fraction or (DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION)
    n_samples_to_use = int(len(X_all) * frac)
    
    np.random.seed(42)
    X_all = X_all[np.random.choice(len(X_all), n_samples_to_use, replace=False)]
    logger.info(f"  Using {n_samples_to_use} random samples ({frac:.0%} of data)")
    
    n_val = int(len(X_all) * VALIDATION_SPLIT)
    X_train, X_val = X_all[:-n_val], X_all[-n_val:]
    logger.info(f"  Data: Train={len(X_train)} Val={len(X_val)}")
    
    trial_epochs = DEBUG_TRIAL_EPOCHS if DEBUG_MODE else PROD_TRIAL_EPOCHS
    final_epochs = DEBUG_FINAL_EPOCHS if DEBUG_MODE else PROD_FINAL_EPOCHS
    
    explainers = {}
    
    # Train explainers
    for exp_type in explainer_types_to_train:
        exp_name = get_standard_method_name(exp_type)
        logger.info(f"  [{exp_name}] Optimizing hyperparameters...")
        
        study, opt_time = run_optimization(exp_type, X_train, X_val, predict_first_horizon,
                                           container.feature_names, n_trials, trial_epochs,
                                           primary_use, option_number, model_name)
        
        logger.info(f"  [{exp_name}] Final training with best params...")
        
        if exp_type == 'tde':
            explainer = TemporalDeepExplainer(n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE,
                                              verbose=True, paired_sampling=True, **study.best_params)
        else:
            explainer = FastSHAPExplainer(n_epochs=final_epochs, patience=EARLY_STOP_PATIENCE,
                                          verbose=True, paired_sampling=True, **study.best_params)
        
        start = time.time()
        final_loss = explainer.train(X_train, X_val, predict_first_horizon, container.feature_names,
                                     gpu_model=model if exp_type == 'tde' else None)
        train_time = time.time() - start
        
        exp_dir = model_dir / exp_type
        exp_dir.mkdir(parents=True, exist_ok=True)
        explainer.save(str(exp_dir))
        
        save_explainer_metadata(primary_use, option_number, model_name, exp_name, study.best_params,
                                study.best_value, final_loss, len(X_train), time_steps, n_features,
                                opt_time, train_time, n_trials, str(exp_dir), container.feature_names)
        
        plot_convergence(explainer.history, plots_dir / f"{exp_type}_convergence.png", f"{exp_name} Convergence")
        explainers[exp_name] = explainer
        logger.info(f"  [{exp_name}] Final Loss: {final_loss:.6f}")
    
    # Load existing explainers if not trained
    for exp_type, exp_name in [('tde', 'TDE'), ('fastshap', 'Fast_SHAP')]:
        if exp_name not in explainers:
            try:
                explainers[exp_name] = load_explainer_for_inference(primary_use, option_number, model_name, exp_type, predict_first_horizon)
            except FileNotFoundError:
                pass
    
    # Comparison setup
    logger.info(f"\n  [COMPARE] Evaluating on {n_test_samples} test samples...")
    
    bg_idx = np.arange(max(0, len(X_train) - min(50, len(X_train))), len(X_train))
    trad = TraditionalSHAPMethods(model, X_train[bg_idx], time_steps, n_features, device)
    
    all_methods = {'Gradient_SHAP': trad.gradient_shap, 'Deep_SHAP': trad.deep_shap}
    for name, exp in explainers.items():
        all_methods[name] = make_explainer_func(exp)
    
    # Setup baseline
    if 'TDE' in explainers:
        baseline_np, base_pred = explainers['TDE'].baseline.cpu().numpy(), explainers['TDE'].base_pred.item()
    elif 'Fast_SHAP' in explainers:
        baseline_np = explainers['Fast_SHAP'].baseline.cpu().numpy().reshape(time_steps, n_features)
        base_pred = explainers['Fast_SHAP'].base_pred.item()
    else:
        baseline_np = np.median(X_train, axis=0)
        base_pred = predict_first_horizon(np.median(X_train, axis=0)[np.newaxis])[0]
    
    metrics = ExplainabilityMetrics(model, baseline_np, base_pred, time_steps, n_features, device)
    X_test = container.X_test[:n_test_samples]
    
    logger.info(f"\n  {'Method':<14} {'Fidelity':>10} {'Sparsity':>10} {'Complex':>10} {'Reliab':>10} {'Effic':>12} {'Time(ms)':>10}")
    logger.info(f"  {'-'*88}")
    
    all_metrics = {m: {'fidelity': [], 'sparsity': [], 'complexity': [], 'reliability': [], 
                       'reliability_mse': [], 'topk_correlation': [], 'efficiency': [], 'time': []} for m in all_methods}
    all_results = {m: [] for m in all_methods}
    
    # Evaluate each test sample
    for idx in range(len(X_test)):
        sample = X_test[idx]
        for method, func in all_methods.items():
            start = time.time()
            try:
                shap_vals = func(sample)
            except Exception:
                continue
            comp_time = time.time() - start
            
            if shap_vals is None or shap_vals.shape != (time_steps, n_features):
                continue
            
            fid = metrics.fidelity(sample, shap_vals)
            spa = metrics.sparsity(shap_vals)
            com = metrics.complexity(shap_vals)
            rel_corr, rel_mse, topk_overlap = metrics.reliability(sample, shap_vals, func)
            eff = metrics.efficiency_error(sample, shap_vals)
            
            result = {'fidelity': fid, 'sparsity': spa, 'complexity': com, 'reliability': rel_corr,
                      'reliability_mse': rel_mse, 'topk_correlation': topk_overlap, 'efficiency': eff, 'time': comp_time}
            all_results[method].append(result)
            
            for k, v in result.items():
                all_metrics[method][k].append(v)
            
            save_comparison(primary_use, option_number, model_name, idx, method, fid, rel_corr, rel_mse, spa,
                            float(com) if com and np.isfinite(com) else None,
                            float(eff) if eff and np.isfinite(eff) else None, comp_time)
            
            generate_shap_heatmap_pdf(shap_vals, sample, container.feature_names,
                                      plots_dir / f"heatmap_sample{idx}_{method.lower().replace('_', '')}.pdf",
                                      method, font_size=12)
    
    # Print summary
    for method, results in all_results.items():
        if results:
            def safe_avg(key):
                vals = [r[key] for r in results if r[key] is not None and np.isfinite(r[key])]
                return np.mean(vals) if vals else float('nan')
            logger.info(f"  {method:<14} {safe_avg('fidelity'):>10.4f} {safe_avg('sparsity'):>9.1f}% "
                       f"{safe_avg('complexity'):>10.4f} {safe_avg('reliability'):>10.4f} "
                       f"{safe_avg('efficiency'):>12.4f} {safe_avg('time')*1000:>9.2f}")
    
    if any(all_metrics[m]['fidelity'] for m in all_metrics):
        plot_metrics_comparison(all_metrics, plots_dir / "xai_methods_comparison.png", f"{model_name}: XAI Methods Comparison")
    
    del model
    torch.cuda.empty_cache()
    
    return all_results, generate_tde_comparison_from_results(all_results)

def generate_tde_comparison_from_results(all_results):
    """Generate TDE comparison data from results dict."""
    if not all_results or 'TDE' not in all_results or not all_results['TDE']:
        return None
    
    metrics_config = {
        'fidelity': {'higher_better': True}, 'sparsity': {'higher_better': True},
        'complexity': {'higher_better': False}, 'reliability': {'higher_better': True}, 'efficiency': {'higher_better': False}
    }
    
    baseline_methods = [m for m in all_results if m != 'TDE' and all_results[m]]
    if not baseline_methods:
        return None
    
    method_averages = {}
    for method, results in all_results.items():
        if results:
            method_averages[method] = {}
            for metric in metrics_config:
                vals = [float(r.get(metric)) for r in results if r.get(metric) is not None and np.isfinite(float(r.get(metric)))]
                method_averages[method][metric] = np.mean(vals) if vals else None
    
    tde_comparison = {}
    total_wins, total_losses = 0, 0
    
    for metric, config in metrics_config.items():
        tde_val = method_averages.get('TDE', {}).get(metric)
        if tde_val is None:
            tde_comparison[metric] = {'value': None, 'status': 'na'}
            continue
        
        better_count, total_compared = 0, 0
        for baseline in baseline_methods:
            baseline_val = method_averages.get(baseline, {}).get(metric)
            if baseline_val is not None:
                total_compared += 1
                if (tde_val > baseline_val) if config['higher_better'] else (tde_val < baseline_val):
                    better_count += 1
        
        if total_compared > 0:
            status = 'best' if better_count == total_compared else ('worst' if better_count == 0 else 'partial')
            if status == 'best': total_wins += 1
            elif status == 'worst': total_losses += 1
        else:
            status = 'na'
        
        tde_comparison[metric] = {'value': tde_val, 'status': status, 'better_count': better_count, 'total_compared': total_compared}
    
    return {'tde_comparison': tde_comparison, 'method_averages': method_averages,
            'overall': 'best' if total_wins > 0 and total_losses == 0 else ('worst' if total_losses > 0 and total_wins == 0 else 'partial'),
            'metrics_config': metrics_config}


# ============================
# DISPLAY & UI FUNCTIONS
# ============================
def show_tde_comparison_table(primary_use, option_number, models):
    """Show comparison table for each deep learning model."""
    print("\n" + "=" * 120)
    print("📊 XAI METHODS COMPARISON BY MODEL")
    print("=" * 120)
    
    model_indices, models_needing_improvement = {}, []
    metrics_list = ['fidelity', 'sparsity', 'complexity', 'reliability', 'efficiency', 'time']
    metrics_config = {
        'fidelity': {'higher_better': True, 'header': 'Fidelity↑'},
        'sparsity': {'higher_better': True, 'header': 'Sparsity↑'},
        'complexity': {'higher_better': False, 'header': 'Complex↓'},
        'reliability': {'higher_better': True, 'header': 'Reliab↑'},
        'efficiency': {'higher_better': False, 'header': 'Effic↓'},
        'time': {'higher_better': False, 'header': 'Time(ms)↓'}
    }
    
    for idx, model_name in enumerate(models, 1):
        model_indices[idx] = model_name
        
        try:
            conn = sqlite3.connect(EXPLAINER_DB)
            cursor = conn.cursor()
            cursor.execute('''SELECT method, AVG(fidelity), AVG(sparsity), AVG(complexity),
                              AVG(reliability_correlation), AVG(efficiency_error), AVG(computation_time)
                              FROM comparison_results WHERE primary_use=? AND option_number=? AND model_name=?
                              GROUP BY method ORDER BY method''', (primary_use, option_number, model_name))
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"\n[{idx}] {model_name}: ❌ Error: {e}")
            continue
        
        if not rows:
            print(f"\n[{idx}] {model_name}: ❌ No comparison data")
            models_needing_improvement.append(idx)
            continue
        
        method_data = {r[0]: {'fidelity': r[1], 'sparsity': r[2], 'complexity': r[3],
                             'reliability': r[4], 'efficiency': r[5], 'time': r[6]} for r in rows}
        
        # Find best method per metric
        best_method_per_metric = {}
        for metric, config in metrics_config.items():
            valid = [(m, method_data[m].get(metric)) for m in method_data 
                     if method_data[m].get(metric) is not None and np.isfinite(method_data[m].get(metric))]
            if valid:
                best_method_per_metric[metric] = (max if config['higher_better'] else min)(valid, key=lambda x: x[1])[0]
        
        # Print table
        print(f"\n┌{'─'*122}┐")
        print(f"│ [{idx}] {model_name:<114} │")
        print(f"├{'─'*122}┤")
        print(f"│ {'Method':<16}" + "".join(f" {metrics_config[m]['header']:>17}" for m in metrics_list) + " │")
        print(f"├{'─'*122}┤")
        
        tde_wins, tde_total = 0, 0
        for method_name in sorted(method_data.keys()):
            data = method_data[method_name]
            row_str = f"│ {method_name:<16}"
            
            for metric in metrics_list:
                val = data.get(metric)
                is_best = best_method_per_metric.get(metric) == method_name
                
                if val is not None and np.isfinite(val):
                    if metric == 'sparsity':
                        val_str = f"{val:.1f}%"
                    elif metric == 'time':
                        val_str = f"{val*1000:.2f}"
                    elif metric == 'efficiency':
                        val_str = f"{val:.6f}"
                    else:
                        val_str = f"{val:.4f}"
                    cell = f"{val_str:>14} ✅" if is_best else f"{val_str:>14}   "
                    if method_name == 'TDE':
                        tde_total += 1
                        if is_best: tde_wins += 1
                else:
                    cell = f"{'N/A':>14}   "
                row_str += f" {cell}"
            print(row_str + " │")
        
        print(f"└{'─'*122}┘")
        
        if 'TDE' not in method_data:
            models_needing_improvement.append(idx)
            print(f"   ⚠️ TDE not available")
        elif tde_total > 0:
            if tde_total - tde_wins > tde_wins:
                models_needing_improvement.append(idx)
                print(f"   ⚠️ TDE underperforming (wins:{tde_wins}/{tde_total})")
            else:
                print(f"   ✅ TDE performing well (wins:{tde_wins}/{tde_total})")
    
    print("\n" + "=" * 122)
    print("Legend: ↑=higher better, ↓=lower better | ✅=best method for that metric")
    print("=" * 122)
    
    return {'model_indices': model_indices, 'models_needing_improvement': models_needing_improvement}

def prompt_tde_retrain(summary_data):
    """Ask which models to retrain TDE for and get new training configuration."""
    if not summary_data:
        return [], None
    
    model_indices = summary_data['model_indices']
    needs_improvement = summary_data.get('models_needing_improvement', [])
    
    print("\n" + "-" * 60)
    print("🔄 TDE RETRAIN OPTIONS")
    print("-" * 60)
    print("\n   Available models:")
    for num, name in model_indices.items():
        print(f"     {num}: {name} {'⚠️' if num in needs_improvement else '✅'}")
    
    print(f"\n   Models needing retrain: {needs_improvement}" if needs_improvement else "\n   ✅ All models performing well!")
    print("\n   Options: [Enter]=Skip, [1,3,5]=By numbers, [LSTM,GRU]=By names, [all]=All underperforming")
    print("-" * 60)
    
    choice = input("\n👉 Models to retrain: ").strip()
    if not choice:
        print("⏭️ Skipping retrain")
        return [], None
    
    if choice.lower() == 'all':
        selected = [model_indices[i] for i in needs_improvement if i in model_indices] if needs_improvement else []
    else:
        selected = []
        if any(c.isdigit() for c in choice):
            for num in [int(x.strip()) for x in choice.replace(' ', ',').split(',') if x.strip().isdigit()]:
                if num in model_indices:
                    selected.append(model_indices[num])
        
        if not selected:
            name_lookup = {name.upper(): name for name in model_indices.values()}
            for part in [x.strip().upper() for x in choice.split(',')]:
                if part in name_lookup:
                    selected.append(name_lookup[part])
                else:
                    selected.extend([name for key, name in name_lookup.items() if part in key])
    
    selected = list(dict.fromkeys(selected))
    
    if not selected:
        print("⏭️ No valid models selected")
        return [], None
    
    # Get new training configuration
    print("\n" + "-" * 60)
    print("📊 RETRAIN CONFIGURATION")
    print("-" * 60)
    print(f"   Selected models: {', '.join(selected)}")
    print("\n   Configure training parameters (new random data will be sampled):")
    
    default_frac = DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION
    frac_input = input(f"\n👉 Training fraction (0.05-1.0) [default: {default_frac}]: ").strip()
    training_fraction = max(0.05, min(1.0, float(frac_input))) if frac_input else default_frac
    
    default_trials = DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS
    trials_input = input(f"👉 Number of optimization trials [default: {default_trials}]: ").strip()
    n_trials = int(trials_input) if trials_input.isdigit() else default_trials
    
    print("\n" + "-" * 60)
    print(f"   ✅ Training fraction: {training_fraction:.0%}")
    print(f"   ✅ Optimization trials: {n_trials}")
    print(f"   ✅ Fresh random sampling: Yes")
    print("-" * 60)
    
    retrain_config = {
        'training_fraction': training_fraction,
        'n_trials': n_trials
    }
    
    return selected, retrain_config

def show_progress_table(primary_use, option_number, models, explainer_types, current_model=None):
    """Show compact progress table with status indicators."""
    print("\n" + "=" * 90)
    print("📊 TRAINING PROGRESS")
    print("=" * 90)
    
    # Build header
    header = f"{'Model':<12}"
    for exp_type in explainer_types:
        header += f" {get_standard_method_name(exp_type):<16}"
    header += f" {'Status':<12}"
    print(header)
    print("-" * 90)
    
    total_complete, total_items = 0, len(models) * len(explainer_types)
    
    for model_name in models:
        row = f"{model_name:<12}"
        model_complete = 0
        
        for exp_type in explainer_types:
            result = check_explainer_exists(primary_use, option_number, model_name, exp_type)
            status = result['status']
            
            if status == 'complete':
                row += f" ✅ Complete     "
                model_complete += 1
                total_complete += 1
            elif status == 'partial':
                row += f" ⚠️  Partial      "
            else:  # fresh
                row += f" 🆕 Fresh        "
        
        # Model overall status
        if current_model == model_name:
            row += f" 🔄 Training  "
        elif model_complete == len(explainer_types):
            row += f" ✅ Done      "
        elif model_complete > 0:
            row += f" ⏳ Partial   "
        else:
            row += f" ⏳ Pending   "
        
        print(row)
    
    print("=" * 90)
    pct = 100 * total_complete / total_items if total_items > 0 else 0
    print(f"📈 Progress: {total_complete}/{total_items} complete ({pct:.0f}%)")
    print("=" * 90)

def delete_existing_results(primary_use, option_number, model_name, explainer_type, status_info=None):
    """
    Delete existing results for a specific explainer before retraining.
    
    Args:
        primary_use: Dataset primary use
        option_number: Dataset option number
        model_name: Name of the model
        explainer_type: Type of explainer ('tde' or 'fastshap')
        status_info: Optional dict with status details from check_explainer_exists
    
    Returns:
        list: Items that were deleted
    """
    standard_name = get_standard_method_name(explainer_type)
    type_key = get_method_key(explainer_type)
    print(f"    🗑️ Deleting existing results for {model_name}/{standard_name}...")
    
    deleted_items = []
    
    # Delete from database: explainer_metadata
    try:
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        # Check if metadata exists before deleting
        cursor.execute('SELECT COUNT(*) FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?',
                       (primary_use, option_number, model_name, standard_name))
        if cursor.fetchone()[0] > 0:
            conn.execute('DELETE FROM explainer_metadata WHERE primary_use=? AND option_number=? AND model_name=? AND explainer_type=?',
                         (primary_use, option_number, model_name, standard_name))
            deleted_items.append('db:metadata')
        
        # Delete hyperparameter trials
        table = 'tde_hyperparameter_trials' if type_key == 'tde' else 'fastshap_hyperparameter_trials'
        cursor.execute(f'SELECT COUNT(*) FROM {table} WHERE primary_use=? AND option_number=? AND model_name=?',
                       (primary_use, option_number, model_name))
        if cursor.fetchone()[0] > 0:
            conn.execute(f'DELETE FROM {table} WHERE primary_use=? AND option_number=? AND model_name=?',
                         (primary_use, option_number, model_name))
            deleted_items.append('db:trials')
        
        # Delete comparison results
        cursor.execute('SELECT COUNT(*) FROM comparison_results WHERE primary_use=? AND option_number=? AND model_name=? AND method=?',
                       (primary_use, option_number, model_name, standard_name))
        if cursor.fetchone()[0] > 0:
            conn.execute('DELETE FROM comparison_results WHERE primary_use=? AND option_number=? AND model_name=? AND method=?',
                         (primary_use, option_number, model_name, standard_name))
            deleted_items.append('db:comparison')
        
        conn.commit()
        conn.close()
        
        if any(item.startswith('db:') for item in deleted_items):
            print(f"    ✅ Database entries deleted: {[i for i in deleted_items if i.startswith('db:')]}")
    except Exception as e:
        print(f"    ⚠️ Database deletion error: {e}")
    
    # Delete Optuna study database
    try:
        optuna_db = Path(get_optuna_db_path(primary_use, option_number, model_name, type_key))
        if optuna_db.exists():
            os.remove(optuna_db)
            deleted_items.append('disk:optuna')
            print(f"    ✅ Optuna study deleted")
    except Exception as e:
        print(f"    ⚠️ Optuna deletion error: {e}")
    
    # Delete explainer files and directory
    try:
        exp_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower() / type_key
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            deleted_items.append('disk:explainer')
            print(f"    ✅ Explainer files deleted")
    except Exception as e:
        print(f"    ⚠️ File deletion error: {e}")
    
    # Delete log files for this model (optional - only if they exist)
    try:
        model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
        if model_dir.exists():
            log_files = list(model_dir.glob("*.log"))
            if log_files:
                for log_file in log_files:
                    log_file.unlink()
                deleted_items.append('disk:logs')
                print(f"    ✅ Log files deleted ({len(log_files)} files)")
    except Exception as e:
        print(f"    ⚠️ Log file deletion error: {e}")
    
    return deleted_items

# ============================
# MAIN/USER INPUT FUNCTIONS
# ============================
def get_user_inputs():
    """Get user configuration through interactive prompts."""
    print("\n" + "=" * 60)
    print("🚀 TDE & Fast_SHAP Training System")
    print("=" * 60)
    print("📊 TDE: Dilated Conv → Multi-Head Attention → Direct Input → Soft Threshold")
    print("📊 Fast_SHAP: Pure MLP with element-wise masking")
    print("📊 Window size is now a hyperparameter: [3, 6, 12, 24]")
    print("=" * 60)
    
    datasets = get_datasets()
    if not datasets:
        print("❌ No datasets found!")
        return None
    
    # Select primary use
    uses = sorted(set(d['primary_use'] for d in datasets))
    print(f"\n📁 Available Primary Uses:")
    for i, u in enumerate(uses):
        print(f"  {i}: {u} ({len([d for d in datasets if d['primary_use'] == u])} options)")
    
    use_input = input(f"\n👉 Select primary use [0-{len(uses)-1}] or 'all' [default: 0]: ").strip().lower()
    selected_uses = uses if use_input == 'all' else [uses[int(use_input)] if use_input.isdigit() and int(use_input) < len(uses) else uses[0]]
    
    filtered_datasets = [d for d in datasets if d['primary_use'] in selected_uses]
    
    # Select option
    if len(selected_uses) == 1:
        use_ds = [d for d in filtered_datasets if d['primary_use'] == selected_uses[0]]
        if len(use_ds) == 1:
            selected_ds = use_ds[0]
        else:
            print(f"\n📋 Available Options:")
            for i, d in enumerate(use_ds):
                print(f"  {i}: Option {d['option_number']}")
            opt = input(f"👉 Select [0-{len(use_ds)-1}] [default: 0]: ").strip()
            selected_ds = use_ds[int(opt)] if opt.isdigit() and int(opt) < len(use_ds) else use_ds[0]
    else:
        selected_ds = filtered_datasets[0]
    
    # Select models
    models = get_models(selected_ds['primary_use'], selected_ds['option_number'])
    print(f"\n🤖 Available Models:")
    for i, m in enumerate(models):
        print(f"  {i}: {m}")
    
    inp = input(f"\n👉 Select [0-{len(models)-1}] or 'all' [default: all]: ").strip().lower()
    selected_models = models if inp in ('', 'all') else ([models[int(inp)]] if inp.isdigit() and int(inp) < len(models) else models)
    
    # Select explainer types
    print("\n🔬 Explainer Types: 0=TDE, 1=Fast_SHAP, 2=Both")
    exp_input = input("👉 Select [0-2] [default: 2]: ").strip()
    exp_choice = int(exp_input) if exp_input.isdigit() and int(exp_input) in [0, 1, 2] else 2
    explainer_types = [['tde'], ['fastshap'], ['tde', 'fastshap']][exp_choice]
    
    # Training fraction
    default_frac = DEBUG_TRAINING_FRACTION if DEBUG_MODE else PROD_TRAINING_FRACTION
    frac_input = input(f"📊 Training fraction (0.05-1.0) [default: {default_frac}]: ").strip()
    training_fraction = max(0.05, min(1.0, float(frac_input))) if frac_input else default_frac
    
    # Trials and test samples
    default_trials = DEBUG_N_TRIALS if DEBUG_MODE else PROD_N_TRIALS
    n_trials = int(input(f"🎯 Optimization trials [default: {default_trials}]: ").strip() or default_trials)
    n_test = int(input("🧪 Test samples [default: 5]: ").strip() or 5)
    
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION")
    print(f"  Dataset: {selected_ds['primary_use']} - Option {selected_ds['option_number']}")
    print(f"  Models: {', '.join(selected_models)}")
    print(f"  Explainers: {', '.join([get_standard_method_name(e) for e in explainer_types])}")
    print(f"  Training fraction: {training_fraction:.0%}, Trials: {n_trials}, Test: {n_test}")
    print("=" * 60)
    
    return {'primary_use': selected_ds['primary_use'], 'option_number': selected_ds['option_number'],
            'models': selected_models, 'explainer_types': explainer_types, 'n_trials': n_trials,
            'n_test_samples': n_test, 'training_fraction': training_fraction}


def main():
    """Main entry point with proper status checking and cleanup."""
    init_database()
    
    config = get_user_inputs()
    if config is None:
        return
    
    primary_use, option_number = config['primary_use'], config['option_number']
    models, explainer_types = config['models'], config['explainer_types']
    training_fraction = config['training_fraction']
    
    # Check status of all items
    items_to_process, status_info = get_incomplete_items(primary_use, option_number, models, explainer_types)
    
    # Print status summary
    print("\n" + "=" * 80)
    print("📊 EXPLAINER STATUS CHECK")
    print("=" * 80)
    
    # Show complete items
    if status_info['complete']:
        print(f"\n✅ COMPLETE ({len(status_info['complete'])}): Already trained & evaluated")
        for item in status_info['complete']:
            print(f"   • {item}")
    
    # Show partial items (need cleanup + retrain)
    if status_info['partial']:
        print(f"\n⚠️  PARTIAL ({len(status_info['partial'])}): Incomplete data found - will cleanup & retrain")
        for item in status_info['partial']:
            model, exp = item.split('/')
            exp_key = get_method_key(exp)
            details = status_info['details'][model][exp_key]['details']
            print(f"   • {item} → found: [{', '.join(details)}]")
    
    # Show fresh items (no existing data)
    if status_info['fresh']:
        print(f"\n🆕 FRESH ({len(status_info['fresh'])}): No existing data - will train from scratch")
        for item in status_info['fresh']:
            print(f"   • {item}")
    
    total_to_process = len(status_info['partial']) + len(status_info['fresh'])
    print("\n" + "=" * 80)
    
    if total_to_process == 0:
        print("✅ All explainers are complete! Nothing to train.")
        
        # Ask if user wants to replace any complete items
        replace_input = input("\n👉 Replace any complete items? (Enter model numbers or 'no') [default: no]: ").strip().lower()
        
        if replace_input and replace_input != 'no' and replace_input != 'n':
            # Parse user input for models to replace
            models_to_replace = []
            if replace_input == 'all':
                models_to_replace = models
            else:
                # Try to parse as comma-separated model names or indices
                for part in replace_input.replace(' ', ',').split(','):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part)
                        if 0 <= idx < len(models):
                            models_to_replace.append(models[idx])
                    else:
                        # Try to match model name
                        for m in models:
                            if part.upper() in m.upper():
                                models_to_replace.append(m)
            
            if models_to_replace:
                models_to_replace = list(set(models_to_replace))  # Remove duplicates
                print(f"\n🔄 Will replace: {', '.join(models_to_replace)}")
                
                # Update items_to_process for replacement
                for model_name in models_to_replace:
                    items_to_process[model_name] = explainer_types.copy()
                    # Move from complete to partial (since we're replacing)
                    for exp_type in explainer_types:
                        key = f"{model_name}/{get_standard_method_name(exp_type)}"
                        if key in status_info['complete']:
                            status_info['complete'].remove(key)
                            status_info['partial'].append(key)
                
                total_to_process = sum(len(v) for v in items_to_process.values())
        
        if total_to_process == 0:
            print("=" * 80)
            # Show comparison table for complete items
            summary_data = show_tde_comparison_table(primary_use, option_number, models)
            
            # Prompt for TDE retrain if needed
            if summary_data:
                models_to_retrain, retrain_config = prompt_tde_retrain(summary_data)
                
                if models_to_retrain and retrain_config:
                    # Load dataset for retrain
                    try:
                        container = load_dataset(primary_use, option_number)
                        print(f"\n📦 Dataset loaded for retrain: {container.X_train.shape[0]} train samples")
                    except Exception as e:
                        print(f"❌ Failed to load dataset: {e}")
                        return
                    
                    for model_name in models_to_retrain:
                        print(f"\n{'='*70}")
                        print(f"🔄 RETRAINING TDE: {model_name}")
                        print(f"   Training fraction: {retrain_config['training_fraction']:.0%}")
                        print(f"   Optimization trials: {retrain_config['n_trials']}")
                        print(f"{'='*70}")
                        
                        # Get current status and cleanup
                        result = check_explainer_exists(primary_use, option_number, model_name, 'tde')
                        print(f"  Current status: {result['status'].upper()}")
                        if result['details']:
                            print(f"  Found: [{', '.join(result['details'])}]")
                        
                        print(f"  🗑️ Cleaning up for fresh retrain...")
                        deleted = delete_existing_results(primary_use, option_number, model_name, 'tde', result)
                        if deleted:
                            print(f"  Removed: [{', '.join(deleted)}]")
                        
                        # Setup logger
                        model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
                        log_path = model_dir / f"tde_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                        logger = setup_logger(str(log_path))
                        
                        try:
                            train_and_compare(primary_use, option_number, model_name, container, ['tde'],
                                              retrain_config['n_trials'], config['n_test_samples'], 
                                              logger, retrain_config['training_fraction'])
                        except Exception as e:
                            logger.error(f"❌ Error retraining TDE for {model_name}: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        # Cleanup logger
                        for h in logger.handlers[:]:
                            h.close()
                            logger.removeHandler(h)
                    
                    # Show final comparison after retrain
                    print("\n" + "=" * 80)
                    print("📊 RESULTS AFTER RETRAIN")
                    print("=" * 80)
                    show_tde_comparison_table(primary_use, option_number, models)
            
            print("\n" + "=" * 80)
            print("✅ COMPLETE")
            print("=" * 80)
            return
    else:
        print(f"🔧 Will process {total_to_process} item(s): {len(status_info['partial'])} partial + {len(status_info['fresh'])} fresh")
        print("=" * 80)
    
    # Confirm before proceeding
    confirm = input("\n👉 Proceed with training? (y/n) [default: y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled.")
        return
    
    # Load dataset once for all models
    container = None
    try:
        container = load_dataset(primary_use, option_number)
        print(f"\n📦 Dataset loaded: {container.X_train.shape[0]} train samples, using {training_fraction:.0%}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    # Show initial progress table
    show_progress_table(primary_use, option_number, models, explainer_types)
    
    # Process each model
    for model_name in models:
        exp_types = items_to_process.get(model_name, [])
        if not exp_types:
            continue
        
        exp_display = [get_standard_method_name(e) for e in exp_types]
        print(f"\n{'='*70}")
        print(f"🔄 PROCESSING: {model_name} - {', '.join(exp_display)}")
        print(f"{'='*70}")
        
        # Handle each explainer type - cleanup partial data first
        for exp_type in exp_types:
            standard_name = get_standard_method_name(exp_type)
            exp_key = get_method_key(exp_type)
            status = status_info['details'][model_name][exp_key]
            
            if status['status'] == 'partial':
                print(f"\n  ⚠️  {standard_name}: Partial data detected - cleaning up...")
                print(f"      Found: [{', '.join(status['details'])}]")
                deleted = delete_existing_results(primary_use, option_number, model_name, exp_type, status)
                if deleted:
                    print(f"      🗑️  Removed: [{', '.join(deleted)}]")
                print(f"      ✅ Cleanup complete - ready for fresh training")
            elif status['status'] == 'complete':
                # User requested replacement of complete item
                print(f"\n  🔄 {standard_name}: Replacing existing complete data...")
                print(f"      Found: [{', '.join(status['details'])}]")
                deleted = delete_existing_results(primary_use, option_number, model_name, exp_type, status)
                if deleted:
                    print(f"      🗑️  Removed: [{', '.join(deleted)}]")
                print(f"      ✅ Cleanup complete - ready for fresh training")
            else:  # fresh
                print(f"\n  🆕 {standard_name}: No existing data - starting fresh training")
        
        # Setup logger for this model
        model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        log_path = model_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger(str(log_path))
        
        logger.info("=" * 60)
        logger.info(f"🤖 Training: {model_name}")
        logger.info(f"📊 Explainers: {', '.join(exp_display)}")
        logger.info(f"📍 Dataset: {primary_use} - Option {option_number}")
        logger.info("=" * 60)
        
        try:
            train_and_compare(primary_use, option_number, model_name, container, exp_types,
                              config['n_trials'], config['n_test_samples'], logger, training_fraction)
        except Exception as e:
            logger.error(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup logger
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
        
        # Update progress table after each model
        show_progress_table(primary_use, option_number, models, explainer_types, model_name)
    
    # Final summary - show comparison table
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    
    summary_data = show_tde_comparison_table(primary_use, option_number, models)
    
    # Prompt for TDE retrain if needed
    if summary_data:
        models_to_retrain, retrain_config = prompt_tde_retrain(summary_data)
        
        if models_to_retrain and retrain_config:
            for model_name in models_to_retrain:
                print(f"\n{'='*70}")
                print(f"🔄 RETRAINING TDE: {model_name}")
                print(f"   Training fraction: {retrain_config['training_fraction']:.0%}")
                print(f"   Optimization trials: {retrain_config['n_trials']}")
                print(f"{'='*70}")
                
                # Get current status and cleanup
                result = check_explainer_exists(primary_use, option_number, model_name, 'tde')
                print(f"  Current status: {result['status'].upper()}")
                if result['details']:
                    print(f"  Found: [{', '.join(result['details'])}]")
                
                print(f"  🗑️ Cleaning up for fresh retrain...")
                deleted = delete_existing_results(primary_use, option_number, model_name, 'tde', result)
                if deleted:
                    print(f"  Removed: [{', '.join(deleted)}]")
                
                # Setup logger
                model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
                log_path = model_dir / f"tde_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
                logger = setup_logger(str(log_path))
                
                try:
                    train_and_compare(primary_use, option_number, model_name, container, ['tde'],
                                      retrain_config['n_trials'], config['n_test_samples'], 
                                      logger, retrain_config['training_fraction'])
                except Exception as e:
                    logger.error(f"❌ Error retraining TDE for {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Cleanup logger
                for h in logger.handlers[:]:
                    h.close()
                    logger.removeHandler(h)
            
            # Show final comparison after retrain
            print("\n" + "=" * 80)
            print("📊 RESULTS AFTER RETRAIN")
            print("=" * 80)
            show_tde_comparison_table(primary_use, option_number, models)
    
    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
