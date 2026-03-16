"""
================================================================================
FILENAME: tde_ablation.py (OPTIMIZED - SKIP EXISTING + EARLY EXIT + AUTO-FIX)
PURPOSE: Ablation studies with per-sample results (matching xai_results.db)

KEY FEATURES:
  ✅ Skip existing results by default (resume capability)
  ✅ Check NULL metrics (re-compute if incomplete) - FIXED!
  ✅ Load training info from explainer_metadata for full_tde
  ✅ EARLY EXIT: Check all results BEFORE loading model
  ✅ Database lock retry with exponential backoff
  ✅ Atomic transactions (all-or-nothing saves)
  ✅ Progress tracking with NULL validation
  ✅ Force re-run option (user choice)
  ✅ Per-sample error handling (continue on failure)
  ✅ AUTO-FIX: Automatically fix incomplete results with NULL values
  
OPTIMIZED FLOW:
  1. Load test samples (lightweight JSON from database)
  2. Check ALL variants for existing results WITH NULL validation
  3. IF all complete → EARLY EXIT (no model loading) ✅
  4. IF work needed → Load model only then
  5. Process only incomplete variants/samples
  6. At end: Auto-detect and offer to fix NULL values
  
FIX NULL VALUES:
  - Automatically detects rows with NULL training metadata
  - Loads correct values from explainer_metadata table
  - Updates all incomplete rows in one batch
  - Verifies completion after fix
  
USAGE:
  python tde_ablation.py
  - Select datasets/models
  - Choose: Skip existing (default) or Force re-run
  - System automatically resumes from where it left off
  - At end: Option to auto-fix any NULL values
  
MANUAL FIX:
  from tde_ablation import fix_incomplete_results
  fix_incomplete_results('health', 0, 'BGRU')
  
SPEED OPTIMIZATION:
  - If all results exist: ~5 seconds (just database checks)
  - If partial results: Only loads model when needed
  - Skips all completed samples automatically
  - NULL validation integrated into progress check
================================================================================
"""

import os
import sys


import json
import time
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from tqdm import tqdm

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn

# Disable PyTorch 2.0 dynamic compilation to avoid Triton issues
os.environ['TORCHDYNAMO_DISABLE'] = '1'
torch.set_float32_matmul_precision('high')  # Optional: fallback to high precision

from Functions.tde_class import TemporalDeepExplainer as TemporalDeepExplainerAblation
from Functions import preprocess
from dl import load_complete_model

# ============================
# CONFIGURATION
# ============================
PATH_DBS = Path("databases")
PATH_DBS.mkdir(parents=True, exist_ok=True)

ABLATION_DB = PATH_DBS / "ablation_studies.db"
XAI_DB = PATH_DBS / "xai_results.db"
EXPLAINER_DB = PATH_DBS / "explainer_results.db"
ENERGY_DB = PATH_DBS / "energy_data.db"
BENCHMARK_DB = PATH_DBS / "benchmark_results.db"
RESULTS_BASE_DIR = "results"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_SHAP_VALUES = True
RANDOM_SEED = 42

# Database retry settings
MAX_DB_RETRIES = 10
DB_RETRY_DELAY = 0.5  # seconds
DB_RETRY_BACKOFF = 1.5  # exponential backoff multiplier

# Global config (set by user in main)
class Config:
    epochs = 5
    use_best_hyperparams = False
    best_hyperparams_cache = {}
    force_rerun = False  # NEW: Force overwrite existing results


# ============================
# DATABASE RETRY WRAPPER
# ============================
def db_execute_with_retry(func, operation_name="DB operation", max_retries=MAX_DB_RETRIES):
    """
    Execute database operation with retry on lock.
    Uses exponential backoff.
    """
    delay = DB_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = delay * (DB_RETRY_BACKOFF ** attempt)
                    print(f"      ⏳ DB locked, retry {attempt+1}/{max_retries} after {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"      ❌ DB locked after {max_retries} retries: {operation_name}")
                    raise
            else:
                print(f"      ❌ DB error during {operation_name}: {e}")
                raise
        except Exception as e:
            print(f"      ❌ Unexpected error during {operation_name}: {e}")
            raise
    
    return None


# ============================
# DATABASE INITIALIZATION
# ============================
def init_ablation_database():
    """Initialize database with PER-SAMPLE structure"""
    
    def _init():
        conn = sqlite3.connect(ABLATION_DB)
        cursor = conn.cursor()
        
        # Per-sample results (EXACT match to xai_results table structure)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ablation_results (
                primary_use TEXT NOT NULL,
                option_number INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                ablation_category TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                sample_idx INTEGER NOT NULL,
                
                -- Metrics (per sample - matching xai_results columns)
                fidelity REAL,
                sparsity REAL,
                complexity REAL,
                reliability_ped REAL,
                reliability_correlation REAL,
                reliability_topk_overlap REAL,
                reliability_kendall_tau REAL,
                completeness_error REAL,
                computation_time REAL,
                
                -- Training info (same for all samples of this variant)
                best_validation_loss REAL,
                final_training_loss REAL,
                training_time REAL,
                n_parameters INTEGER,
                
                -- Configuration
                config_json TEXT,
                
                timestamp TEXT NOT NULL,
                
                PRIMARY KEY (primary_use, option_number, model_name, ablation_category, variant_name, sample_idx)
            )
        ''')
        
        # SHAP values (per sample)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ablation_shap_values (
                primary_use TEXT NOT NULL,
                option_number INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                variant_name TEXT NOT NULL,
                sample_idx INTEGER NOT NULL,
                
                shap_values_json TEXT NOT NULL,
                
                PRIMARY KEY (primary_use, option_number, model_name, variant_name, sample_idx)
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_ablation_lookup 
            ON ablation_results(primary_use, option_number, model_name, variant_name)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_shap_lookup 
            ON ablation_shap_values(primary_use, option_number, model_name, variant_name)
        ''')
        
        conn.commit()
        conn.close()
    
    db_execute_with_retry(_init, "Database initialization")


# ============================
# CHECK EXISTING RESULTS (WITH NULL CHECK)
# ============================
def check_result_exists(primary_use, option_number, model_name, category, variant_name, sample_idx):
    """
    Check if result already exists in database AND has no NULL values in critical columns.
    
    Critical columns (must be non-NULL):
    - Core metrics: fidelity, sparsity, complexity, completeness_error, computation_time
    - Training metadata: best_validation_loss, final_training_loss, training_time
    
    Optional columns (can be NULL):
    - Reliability metrics: reliability_ped, reliability_correlation, reliability_topk_overlap, reliability_kendall_tau
    - n_parameters: Can be NULL for variants loaded from xai_results
    
    Returns:
        bool: True if exists with all critical columns non-NULL, False otherwise
    """
    option_number = int(option_number)
    sample_idx = int(sample_idx)
    
    def _check():
        conn = sqlite3.connect(ABLATION_DB)
        cursor = conn.cursor()
        
        # Get all columns
        cursor.execute('''
            SELECT 
                fidelity, sparsity, complexity,
                reliability_ped, reliability_correlation,
                reliability_topk_overlap, reliability_kendall_tau,
                completeness_error, computation_time,
                best_validation_loss, final_training_loss,
                training_time, n_parameters
            FROM ablation_results 
            WHERE primary_use=? AND option_number=? AND model_name=? 
              AND ablation_category=? AND variant_name=? AND sample_idx=?
        ''', (primary_use, option_number, model_name, category, variant_name, sample_idx))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return False  # Doesn't exist
        
        # Unpack all columns
        fidelity = row[0]
        sparsity = row[1]
        complexity = row[2]
        reliability_ped = row[3]
        reliability_correlation = row[4]
        reliability_topk_overlap = row[5]
        reliability_kendall_tau = row[6]
        completeness_error = row[7]
        computation_time = row[8]
        best_validation_loss = row[9]
        final_training_loss = row[10]
        training_time = row[11]
        n_parameters = row[12]
        
        # CRITICAL COLUMNS - Must be non-NULL
        critical_metrics = [
            fidelity,
            sparsity,
            complexity,
            completeness_error,
            computation_time
        ]
        
        critical_training = [
            best_validation_loss,
            final_training_loss,
            training_time
        ]
        
        # Check all critical columns
        all_critical_valid = all(val is not None for val in critical_metrics + critical_training)
        
        # OPTIONAL COLUMNS - Can be NULL
        # - reliability_ped, reliability_correlation, reliability_topk_overlap, reliability_kendall_tau
        #   (can be NULL if noisy sample evaluation fails)
        # - n_parameters 
        #   (can be NULL for variants loaded from xai_results where we don't have this info)
        
        return all_critical_valid
    
    try:
        return db_execute_with_retry(_check, f"Check exists {variant_name}/{sample_idx}")
    except:
        return False


def get_variant_progress(primary_use, option_number, model_name, category, variant_name, total_samples):
    """
    Get progress for a specific variant.
    
    IMPORTANT: Uses same NULL validation as check_result_exists()
    Only counts rows as "completed" if ALL critical columns are non-NULL.
    
    Returns:
        dict: {
            'total': int,
            'completed': int,
            'remaining': int,
            'completed_indices': list,
            'remaining_indices': list
        }
    """
    option_number = int(option_number)
    
    def _get_progress():
        conn = sqlite3.connect(ABLATION_DB)
        cursor = conn.cursor()
        
        # Get ALL rows for this variant with critical columns
        cursor.execute('''
            SELECT 
                sample_idx,
                fidelity, sparsity, complexity,
                completeness_error, computation_time,
                best_validation_loss, final_training_loss, training_time
            FROM ablation_results 
            WHERE primary_use=? AND option_number=? AND model_name=? 
              AND ablation_category=? AND variant_name=?
            ORDER BY sample_idx
        ''', (primary_use, option_number, model_name, category, variant_name))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Validate each row for NULL values
        completed_indices = []
        
        for row in rows:
            sample_idx = row[0]
            
            # Check critical metrics
            fidelity = row[1]
            sparsity = row[2]
            complexity = row[3]
            completeness_error = row[4]
            computation_time = row[5]
            
            # Check training metadata
            best_validation_loss = row[6]
            final_training_loss = row[7]
            training_time = row[8]
            
            # Validate all critical columns
            critical_valid = all([
                fidelity is not None,
                sparsity is not None,
                complexity is not None,
                completeness_error is not None,
                computation_time is not None,
                best_validation_loss is not None,
                final_training_loss is not None,
                training_time is not None
            ])
            
            # Only count as completed if all critical columns are non-NULL
            if critical_valid:
                completed_indices.append(sample_idx)
        
        all_indices = list(range(total_samples))
        remaining_indices = [i for i in all_indices if i not in completed_indices]
        
        return {
            'total': total_samples,
            'completed': len(completed_indices),
            'remaining': len(remaining_indices),
            'completed_indices': completed_indices,
            'remaining_indices': remaining_indices
        }
    
    try:
        return db_execute_with_retry(_get_progress, f"Get progress {variant_name}")
    except:
        return {
            'total': total_samples,
            'completed': 0,
            'remaining': total_samples,
            'completed_indices': [],
            'remaining_indices': list(range(total_samples))
        }


def get_all_variants_progress(primary_use, option_number, model_name, all_variants, total_samples):
    """
    Get progress summary for all variants.
    
    Returns:
        dict: {variant_name: progress_dict, ...}
    """
    progress = {}
    
    for variant_name, variant_config in all_variants.items():
        category = variant_config['category']
        progress[variant_name] = get_variant_progress(
            primary_use, option_number, model_name, category, variant_name, total_samples
        )
    
    return progress


# ============================
# ABLATION VARIANTS
# ============================
def get_architectural_ablations():
    """Define architectural component ablations"""
    return {
        'full_tde': {
            'category': 'architecture',
            'description': 'Full TDE with all components',
            'load_from_xai': True,
            'params': {
                'use_attention_gate': True,
            }
        },
        'no_attention': {
            'category': 'architecture',
            'description': 'TDE without attention gating',
            'load_from_xai': False,
            'params': {
                'use_attention_gate': False,
            }
        },
        'tcn_baseline': {
            'category': 'architecture',
            'description': 'Baseline with only temporal convolution',
            'load_from_xai': False,
            'params': {
                'use_attention_gate': False,
            }
        }
    }


def get_loss_term_ablations():
    """Define loss term ablations"""
    base = {
        'l1_lambda': 0.01,
        'l2_lambda': 0.01,
        'smoothness_lambda': 0.1,
        'efficiency_lambda': 0.1,
        'sparsity_lambda': 0.1,
    }
    
    return {
        'full_loss': {
            'category': 'loss_terms',
            'description': 'Full loss with all regularization terms',
            'load_from_xai': True,
            'params': base.copy()
        },
        'no_l1': {
            'category': 'loss_terms',
            'description': 'Without L1 regularization',
            'load_from_xai': False,
            'params': {**base, 'l1_lambda': 0.0}
        },
        'no_smoothness': {
            'category': 'loss_terms',
            'description': 'Without temporal smoothness',
            'load_from_xai': False,
            'params': {**base, 'smoothness_lambda': 0.0}
        },
        'only_coalition': {
            'category': 'loss_terms',
            'description': 'Only coalition loss (no regularization)',
            'load_from_xai': False,
            'params': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.0,
                'smoothness_lambda': 0.0,
                'efficiency_lambda': 0.0,
                'sparsity_lambda': 0.0,
            }
        }
    }


# ============================
# LOAD FROM XAI_RESULTS.DB
# ============================
def load_best_hyperparameters(primary_use, option_number, model_name):
    """Load best hyperparameters from explainer_metadata table"""
    
    def _load():
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT best_hyperparameters
            FROM explainer_metadata
            WHERE primary_use = ? 
              AND option_number = ? 
              AND model_name = ? 
              AND explainer_type = 'TDE'
        ''', (primary_use, option_number, model_name))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            params = json.loads(row[0])
            return params
        return None
    
    try:
        return db_execute_with_retry(_load, "Load hyperparameters")
    except:
        return None


def get_simple_default_hyperparameters():
    """Get simple default hyperparameters (feature sampling)"""
    return {
        'n_epochs': Config.epochs,
        'batch_size': 256,
        'patience': 5,
        'verbose': False,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'n_conv_layers': 2,
        'kernel_size': 3,
        'dropout_rate': 0.2,
        'n_attention_heads': 4,
        'masking_mode': 'feature',
        'paired_sampling': True,
        'samples_per_feature': 2,
    }


def load_full_model_results(primary_use, option_number, model_name):
    """Load full TDE results from xai_results.db"""
    
    def _load():
        conn = sqlite3.connect(XAI_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                sample_idx,
                fidelity, sparsity, complexity,
                reliability_ped, reliability_correlation,
                reliability_topk_overlap, reliability_kendall_tau,
                efficiency_error, computation_time,
                shap_values_original_json
            FROM xai_results
            WHERE primary_use = ? 
              AND option_number = ? 
              AND model_name = ? 
              AND xai_method = 'tde'
            ORDER BY sample_idx
        ''', (primary_use, option_number, model_name))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        results = {}
        for row in rows:
            sample_idx = row[0]
            results[sample_idx] = {
                'fidelity': row[1],
                'sparsity': row[2],
                'complexity': row[3],
                'reliability_ped': row[4],
                'reliability_correlation': row[5],
                'reliability_topk_overlap': row[6],
                'reliability_kendall_tau': row[7],
                'completeness_error': row[8],
                'computation_time': row[9],
                'shap_values_json': row[10]
            }
        
        return results
    
    try:
        return db_execute_with_retry(_load, "Load full model results")
    except:
        return None


def load_training_info_from_metadata(primary_use, option_number, model_name):
    """
    Load training info from explainer_metadata table for full_tde variant.
    
    Returns:
        dict with best_validation_loss, final_training_loss, training_time, n_parameters
        or None if not found
    """
    
    def _load():
        conn = sqlite3.connect(EXPLAINER_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                best_validation_loss,
                final_training_loss,
                training_time,
                n_training_samples,
                time_steps,
                n_features
            FROM explainer_metadata
            WHERE primary_use = ? 
              AND option_number = ? 
              AND model_name = ? 
              AND explainer_type = 'TDE'
        ''', (primary_use, option_number, model_name))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'best_validation_loss': row[0],
                'final_training_loss': row[1],
                'training_time': row[2],
                'n_parameters': None,  # Not stored in explainer_metadata
                'n_training_samples': row[3],
                'time_steps': row[4],
                'n_features': row[5]
            }
        return None
    
    try:
        return db_execute_with_retry(_load, "Load training info from metadata")
    except:
        return None


def get_test_samples_from_xai(primary_use, option_number):
    """Load test samples from xai_results.db"""
    
    def _load():
        conn = sqlite3.connect(XAI_DB)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sample_idx, original_sample_json, noisy_sample_json
            FROM test_samples
            WHERE primary_use = ? AND option_number = ?
            ORDER BY sample_idx
        ''', (primary_use, option_number))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return None
        
        samples = {}
        for row in rows:
            sample_idx = row[0]
            samples[sample_idx] = {
                'original': np.array(json.loads(row[1]), dtype=np.float32),
                'noisy': np.array(json.loads(row[2]), dtype=np.float32)
            }
        
        return samples
    
    try:
        return db_execute_with_retry(_load, "Load test samples")
    except:
        return None


# ============================
# METRICS (SAME AS BEFORE)
# ============================
class AblationMetrics:
    """Compute metrics for ablation study"""
    
    def __init__(self, model, baseline, base_pred, time_steps, n_features, dev):
        self.device = dev
        self.time_steps = time_steps
        self.n_features = n_features
        self.baseline = baseline
        self.base_pred = base_pred
        
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
        
        self.wrapped_model = SingleHorizonWrapper(model, horizon_idx=0).to(dev)
        self.wrapped_model.eval()
    
    def _get_prediction(self, x):
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return self.wrapped_model(x.to(self.device)).cpu().numpy().flatten()[0]
    
    def compute_all_metrics(self, shap_vals, sample, shap_func_noisy=None, sample_noisy=None):
        """Compute all metrics for a single sample"""
        metrics = {}
        
        metrics['fidelity'] = self.fidelity(sample, shap_vals)
        metrics['sparsity'] = self.sparsity(shap_vals)
        metrics['complexity'] = self.complexity(shap_vals)
        metrics['completeness_error'] = self.completeness_error(sample, shap_vals)
        
        if shap_func_noisy is not None and sample_noisy is not None:
            shap_vals_noisy = shap_func_noisy(sample_noisy)
            if shap_vals_noisy is not None:
                rel_metrics = self.reliability_all(shap_vals, shap_vals_noisy)
                metrics['reliability_ped'] = rel_metrics['ped']
                metrics['reliability_correlation'] = rel_metrics['correlation']
                metrics['reliability_topk_overlap'] = rel_metrics['topk_overlap']
                metrics['reliability_kendall_tau'] = rel_metrics['kendall_tau']
            else:
                metrics['reliability_ped'] = None
                metrics['reliability_correlation'] = None
                metrics['reliability_topk_overlap'] = None
                metrics['reliability_kendall_tau'] = None
        else:
            metrics['reliability_ped'] = None
            metrics['reliability_correlation'] = None
            metrics['reliability_topk_overlap'] = None
            metrics['reliability_kendall_tau'] = None
        
        return metrics
    
    def fidelity(self, sample, shap_vals, top_k_pct=10):
        if shap_vals is None:
            return None
        try:
            sample_2d = sample[0] if sample.ndim == 3 else sample
            baseline_2d = self.baseline.cpu().numpy() if isinstance(self.baseline, torch.Tensor) else self.baseline
            if baseline_2d.ndim == 3:
                baseline_2d = baseline_2d[0]
            
            orig_pred = self._get_prediction(sample_2d)
            k = max(1, int(shap_vals.size * top_k_pct / 100))
            top_k_idx = np.argsort(np.abs(shap_vals).flatten())[-k:]
            
            masked = sample_2d.copy()
            for idx in top_k_idx:
                masked[idx // self.n_features, idx % self.n_features] = baseline_2d[idx // self.n_features, idx % self.n_features]
            
            masked_pred = self._get_prediction(masked)
            return float(abs(orig_pred - masked_pred))
        except:
            return None
    
    def sparsity(self, shap_vals, threshold_pct=1):
        if shap_vals is None:
            return None
        try:
            abs_shap = np.abs(shap_vals)
            max_val = np.max(abs_shap)
            if max_val == 0:
                return 100.0
            threshold = max_val * threshold_pct / 100
            return float(np.sum(abs_shap < threshold) / abs_shap.size * 100)
        except:
            return None
    
    def complexity(self, shap_vals):
        if shap_vals is None:
            return None
        try:
            abs_shap = np.abs(shap_vals).flatten() + 1e-10
            probs = abs_shap / np.sum(abs_shap)
            return float(-np.sum(probs * np.log(probs)))
        except:
            return None
    
    def reliability_all(self, shap_orig, shap_noisy):
        """Compute all reliability metrics"""
        if shap_orig is None or shap_noisy is None:
            return {'ped': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}
        
        try:
            orig_flat = shap_orig.flatten()
            noisy_flat = shap_noisy.flatten()
            
            mask = np.isfinite(orig_flat) & np.isfinite(noisy_flat)
            if np.sum(mask) < 10:
                return {'ped': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}
            
            orig_valid = orig_flat[mask]
            noisy_valid = noisy_flat[mask]
            
            max_mag = max(np.max(np.abs(orig_valid)), np.max(np.abs(noisy_valid)), 1e-10)
            ped = np.mean(np.abs(orig_valid - noisy_valid)) / max_mag * 100
            
            correlation, _ = pearsonr(orig_valid, noisy_valid)
            
            k = max(1, int(len(orig_flat) * 10 / 100))
            top_k_orig = set(np.argsort(np.abs(orig_flat))[-k:])
            top_k_noisy = set(np.argsort(np.abs(noisy_flat))[-k:])
            topk_overlap = len(top_k_orig & top_k_noisy) / k * 100
            
            kendall_tau_val, _ = kendalltau(orig_valid, noisy_valid)
            
            return {
                'ped': float(ped) if np.isfinite(ped) else None,
                'correlation': float(correlation) if np.isfinite(correlation) else None,
                'topk_overlap': float(topk_overlap) if np.isfinite(topk_overlap) else None,
                'kendall_tau': float(kendall_tau_val) if np.isfinite(kendall_tau_val) else None
            }
        except:
            return {'ped': None, 'correlation': None, 'topk_overlap': None, 'kendall_tau': None}
    
    def completeness_error(self, sample, shap_vals):
        """Measures SHAP completeness"""
        if shap_vals is None:
            return None
        try:
            sample_2d = sample[0] if sample.ndim == 3 else sample
            sample_pred = self._get_prediction(sample_2d)
            
            baseline_2d = self.baseline.cpu().numpy() if isinstance(self.baseline, torch.Tensor) else self.baseline
            if baseline_2d.ndim == 3:
                baseline_2d = baseline_2d[0]
            
            baseline_pred = self._get_prediction(baseline_2d)
            expected = sample_pred - baseline_pred
            actual = np.sum(shap_vals)
            
            return float(abs(actual - expected) / (abs(expected) + 1e-10))
        except:
            return None


# ============================
# DATABASE OPERATIONS (ATOMIC)
# ============================
def save_sample_result(primary_use, option_number, model_name, category, variant_name,
                       sample_idx, metrics, training_info, config):
    """
    Save single sample result with atomic transaction.
    Uses INSERT or REPLACE for idempotency.
    """
    
    option_number = int(option_number)
    sample_idx = int(sample_idx)
    
    def _save():
        conn = sqlite3.connect(ABLATION_DB)
        conn.isolation_level = None  # Auto-commit mode
        cursor = conn.cursor()
        
        try:
            cursor.execute('BEGIN IMMEDIATE')  # Exclusive lock
            
            cursor.execute('''
                INSERT OR REPLACE INTO ablation_results (
                    primary_use, option_number, model_name, ablation_category, variant_name, sample_idx,
                    fidelity, sparsity, complexity,
                    reliability_ped, reliability_correlation,
                    reliability_topk_overlap, reliability_kendall_tau,
                    completeness_error, computation_time,
                    best_validation_loss, final_training_loss,
                    training_time, n_parameters,
                    config_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                primary_use, option_number, model_name, category, variant_name, sample_idx,
                metrics.get('fidelity'), metrics.get('sparsity'), metrics.get('complexity'),
                metrics.get('reliability_ped'), metrics.get('reliability_correlation'),
                metrics.get('reliability_topk_overlap'), metrics.get('reliability_kendall_tau'),
                metrics.get('completeness_error'), metrics.get('computation_time'),
                training_info.get('best_validation_loss'), training_info.get('final_training_loss'),
                training_info.get('training_time'), training_info.get('n_parameters'),
                json.dumps(config), datetime.now().isoformat()
            ))
            
            cursor.execute('COMMIT')
            conn.close()
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            raise e
    
    try:
        return db_execute_with_retry(_save, f"Save result {variant_name}/{sample_idx}")
    except Exception as e:
        print(f"     ❌ Failed to save {variant_name}/{sample_idx}: {e}")
        return False


def save_shap_values(primary_use, option_number, model_name, variant_name, sample_idx, shap_values):
    """Save SHAP values with atomic transaction"""
    
    if shap_values is None:
        return False
    
    option_number = int(option_number)
    sample_idx = int(sample_idx)
    
    try:
        if isinstance(shap_values, np.ndarray):
            shap_json = json.dumps(shap_values.tolist())
        else:
            shap_json = json.dumps(shap_values)
    except:
        return False
    
    def _save():
        conn = sqlite3.connect(ABLATION_DB)
        conn.isolation_level = None
        cursor = conn.cursor()
        
        try:
            cursor.execute('BEGIN IMMEDIATE')
            
            cursor.execute('''
                INSERT OR REPLACE INTO ablation_shap_values 
                (primary_use, option_number, model_name, variant_name, sample_idx, shap_values_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (primary_use, option_number, model_name, variant_name, sample_idx, shap_json))
            
            cursor.execute('COMMIT')
            conn.close()
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            raise e
    
    try:
        return db_execute_with_retry(_save, f"Save SHAP {variant_name}/{sample_idx}")
    except:
        return False


# ============================
# ABLATION RUNNER (WITH SKIP LOGIC)
# ============================
def run_single_ablation(primary_use, option_number, model_name, variant_name, variant_config,
                       test_samples, model, predict_func, feature_names, 
                       save_shap=True, pbar=None):
    """
    Run single ablation experiment.
    
    SKIP LOGIC:
    - If load_from_xai=True: Copy from xai_results.db (skip if exists)
    - Otherwise: Train and evaluate (skip completed samples)
    """
    
    option_number = int(option_number)
    category = variant_config['category']
    
    if pbar:
        pbar.set_description(f"🔬 {variant_name[:20]:<20}")
    
    # Check existing progress
    total_samples = len(test_samples)
    progress = get_variant_progress(primary_use, option_number, model_name, category, variant_name, total_samples)
    
    if progress['completed'] == progress['total'] and not Config.force_rerun:
        if pbar:
            pbar.set_postfix_str(f"✅ Already complete ({progress['completed']}/{progress['total']})")
        return progress['completed']
    
    # CASE 1: Load from xai_results.db
    if variant_config.get('load_from_xai', False):
        print(f"\n   📥 Loading {variant_name} from xai_results.db...")
        
        full_results = load_full_model_results(primary_use, option_number, model_name)
        
        if full_results is None:
            if pbar:
                pbar.set_postfix_str("⚠️  No data in xai_results.db")
            return 0
        
        # Load training info from explainer_metadata
        print(f"   📥 Loading training info from explainer_metadata...")
        training_info = load_training_info_from_metadata(primary_use, option_number, model_name)
        
        if training_info is None:
            # CRITICAL: Cannot proceed without training metadata
            print(f"   ❌ ERROR: No training info found in explainer_metadata!")
            print(f"      This variant requires training metadata from explainer_metadata table.")
            print(f"      Please ensure TDE has been trained and metadata saved for:")
            print(f"      - primary_use: {primary_use}")
            print(f"      - option_number: {option_number}")
            print(f"      - model_name: {model_name}")
            print(f"      - explainer_type: TDE")
            if pbar:
                pbar.set_postfix_str("❌ No training metadata")
            return 0  # Skip this variant entirely
        else:
            print(f"   ✅ Loaded: val_loss={training_info['best_validation_loss']:.6f}, "
                  f"train_time={training_info['training_time']:.1f}s")
        
        saved_count = 0
        skipped_count = 0
        
        for sample_idx, metrics in full_results.items():
            # Skip if exists and not forcing re-run
            if not Config.force_rerun and check_result_exists(
                primary_use, option_number, model_name, category, variant_name, sample_idx
            ):
                skipped_count += 1
                continue
            
            # Save result (now guaranteed to have training_info)
            if save_sample_result(primary_use, option_number, model_name,
                                category, variant_name,
                                sample_idx, metrics, training_info,
                                variant_config['params']):
                saved_count += 1
            
            # Save SHAP values if available
            if save_shap and metrics.get('shap_values_json'):
                save_shap_values(primary_use, option_number, model_name,
                               variant_name, sample_idx, metrics['shap_values_json'])
        
        if pbar:
            pbar.set_postfix_str(f"✅ Saved={saved_count}, Skipped={skipped_count}")
        
        return saved_count
    
    # CASE 2: Train from scratch
    # Get hyperparameters
    cache_key = f"{primary_use}_{option_number}_{model_name}"
    
    if Config.use_best_hyperparams:
        if cache_key not in Config.best_hyperparams_cache:
            Config.best_hyperparams_cache[cache_key] = load_best_hyperparameters(
                primary_use, option_number, model_name
            )
        
        base_params = Config.best_hyperparams_cache[cache_key]
        
        if base_params is None:
            print(f"   ⚠️  Using simple defaults as fallback")
            base_params = get_simple_default_hyperparameters()
        else:
            base_params['n_epochs'] = Config.epochs
            base_params['verbose'] = False
    else:
        base_params = get_simple_default_hyperparameters()
    
    full_params = {**base_params, **variant_config['params']}
    
    # Prepare training data
    sample_arrays = [s['original'] for s in test_samples.values()]
    X_data = np.array(sample_arrays)
    
    split_idx = int(len(X_data) * 0.8)
    X_train = X_data[:split_idx]
    X_val = X_data[split_idx:]
    
    if len(X_train) < 1 or len(X_val) < 1:
        if pbar:
            pbar.set_postfix_str("❌ Not enough samples")
        return 0
    
    try:
        # Training (only if needed)
        start_time = time.time()
        explainer = TemporalDeepExplainerAblation(**full_params)
        val_loss = explainer.train(X_train, X_val, predict_func, feature_names, gpu_model=model)
        training_time = time.time() - start_time
        
        training_info = {
            'best_validation_loss': val_loss,
            'final_training_loss': explainer.history['train_loss'][-1] if explainer.history['train_loss'] else None,
            'training_time': training_time,
            'n_parameters': sum(p.numel() for p in explainer.explainer.parameters())
        }
        
        # Evaluation (skip completed samples)
        time_steps = X_train.shape[1]
        n_features = X_train.shape[2]
        
        metrics_calc = AblationMetrics(
            model, explainer.baseline, explainer.base_pred,
            time_steps, n_features, device
        )
        
        saved_count = 0
        skipped_count = 0
        failed_count = 0
        shap_saved = 0
        
        for sample_idx, sample_data in test_samples.items():
            # Skip if exists and not forcing re-run
            if not Config.force_rerun and check_result_exists(
                primary_use, option_number, model_name, category, variant_name, sample_idx
            ):
                skipped_count += 1
                continue
            
            try:
                sample = sample_data['original']
                sample_noisy = sample_data['noisy']
                
                comp_start = time.time()
                shap_vals = explainer.explain(sample)
                comp_time = time.time() - comp_start
                
                metrics = metrics_calc.compute_all_metrics(
                    shap_vals, sample,
                    lambda x: explainer.explain(x), sample_noisy
                )
                metrics['computation_time'] = comp_time
                
                # Save result
                if save_sample_result(primary_use, option_number, model_name,
                                    category, variant_name,
                                    sample_idx, metrics, training_info,
                                    full_params):
                    saved_count += 1
                else:
                    failed_count += 1
                
                # Save SHAP values
                if save_shap and shap_vals is not None:
                    if save_shap_values(primary_use, option_number, model_name,
                                      variant_name, sample_idx, shap_vals):
                        shap_saved += 1
                        
            except Exception as e:
                print(f"\n      ❌ Sample {sample_idx} failed: {e}")
                failed_count += 1
                continue  # Continue with next sample
        
        del explainer
        torch.cuda.empty_cache()
        
        if pbar:
            pbar.set_postfix_str(
                f"✅ Saved={saved_count}, Skip={skipped_count}, Fail={failed_count}, SHAP={shap_saved}"
            )
        
        return saved_count
        
    except Exception as e:
        if pbar:
            pbar.set_postfix_str(f"❌ ERROR: {str(e)[:30]}")
        print(f"\n❌ Error in {variant_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def run_ablation_study(primary_use, option_number, model_name,
                       ablation_types=['architecture', 'loss_terms']):
    """Run complete ablation study with skip logic"""
    
    option_number = int(option_number)
    
    print(f"\n{'='*80}")
    print(f"📊 {primary_use} - Option {option_number} - {model_name}")
    print(f"{'='*80}")
    
    # STEP 1: Load test samples (lightweight - just metadata)
    print("📦 Loading test samples from xai_results.db...")
    test_samples = get_test_samples_from_xai(primary_use, option_number)
    
    if test_samples is None:
        print("❌ Cannot proceed without test samples")
        return []
    
    total_samples = len(test_samples)
    print(f"   ✅ Loaded {total_samples} test samples")
    
    # STEP 2: Collect all variants
    all_variants = {}
    
    if 'architecture' in ablation_types:
        all_variants.update(get_architectural_ablations())
    
    if 'loss_terms' in ablation_types:
        all_variants.update(get_loss_term_ablations())
    
    print(f"\n🔬 Total variants to process: {len(all_variants)}")
    
    # STEP 3: Check progress for ALL variants BEFORE loading model
    print(f"\n📊 Checking existing results (with NULL validation)...")
    progress_all = get_all_variants_progress(primary_use, option_number, model_name, all_variants, total_samples)
    
    variants_needing_work = []
    
    for variant_name, prog in progress_all.items():
        status = "✅" if prog['completed'] == prog['total'] else "⏳"
        print(f"   {status} {variant_name:<20}: {prog['completed']}/{prog['total']} samples")
        
        # Track which variants need work
        if prog['remaining'] > 0 or Config.force_rerun:
            variants_needing_work.append(variant_name)
    
    total_remaining = sum(p['remaining'] for p in progress_all.values())
    total_possible = len(all_variants) * total_samples
    
    print(f"\n🔬 Overall: {total_possible - total_remaining}/{total_possible} samples completed")
    
    # STEP 4: Early exit if everything is complete
    if total_remaining == 0 and not Config.force_rerun:
        print(f"\n✅ ALL VARIANTS COMPLETE! Skipping model loading.")
        print(f"   Use force_rerun=True to re-run.")
        return []
    
    if not variants_needing_work:
        print(f"\n✅ No work needed. All variants complete.")
        return []
    
    print(f"\n⚙️  Variants needing work: {len(variants_needing_work)}/{len(all_variants)}")
    print(f"   {', '.join(variants_needing_work)}")
    
    # STEP 5: NOW load model (only if needed)
    print(f"\n🤖 Loading model (needed for {len(variants_needing_work)} variants)...")
    model_dir = Path(RESULTS_BASE_DIR) / primary_use / f"option_{option_number}" / model_name.lower()
    model_path = model_dir / "trained_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_complete_model(str(model_path), device=device)
    model.eval()
    print(f"   ✅ Model loaded")
    
    # Get shape info from first sample
    first_sample = next(iter(test_samples.values()))['original']
    time_steps = first_sample.shape[0]
    n_features = first_sample.shape[1]
    
    # Load feature names
    print(f"📦 Loading feature names...")
    container = preprocess.load_and_preprocess_data_with_sequences(
        db_path=ENERGY_DB,
        primary_use=primary_use,
        option_number=option_number,
        scaled=True,
        scale_type="both"
    )
    feature_names = container.feature_names
    print(f"   ✅ {len(feature_names)} features loaded")
    
    def predict_first_horizon(X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if X.ndim == 2:
            X = X.reshape(-1, time_steps, n_features)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        return pred[:, 0] if pred.ndim > 1 and pred.shape[1] > 0 else pred.flatten()
    
    # STEP 6: Run ablations (only for variants needing work)
    print(f"\n🔬 Starting ablation processing...")
    results_summary = []
    
    with tqdm(total=len(all_variants), desc=f"🔬 Ablating {model_name}",
              unit="variant", position=0, leave=True) as pbar:
        
        for variant_name, variant_config in all_variants.items():
            saved_count = run_single_ablation(
                primary_use, option_number, model_name,
                variant_name, variant_config,
                test_samples, model, predict_first_horizon, feature_names,
                save_shap=SAVE_SHAP_VALUES,
                pbar=pbar
            )
            
            results_summary.append({
                'variant_name': variant_name,
                'category': variant_config['category'],
                'samples_saved': saved_count
            })
            
            pbar.update(1)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results_summary


# ============================
# FIX INCOMPLETE RESULTS
# ============================
def fix_incomplete_results(primary_use, option_number, model_name, variants_to_fix=None):
    """
    Fix existing results with NULL training metadata by loading from explainer_metadata.
    
    This is useful when results were saved before training metadata was available.
    
    Args:
        primary_use: Dataset primary use
        option_number: Dataset option number
        model_name: Model name
        variants_to_fix: List of variant names to fix, or None for all
    
    Returns:
        int: Number of rows updated
    """
    
    print(f"\n{'='*80}")
    print(f"🔧 FIXING INCOMPLETE RESULTS")
    print(f"{'='*80}")
    print(f"Dataset: {primary_use}/{option_number}/{model_name}")
    
    # Load training info from explainer_metadata
    print(f"\n📥 Loading training info from explainer_metadata...")
    training_info = load_training_info_from_metadata(primary_use, option_number, model_name)
    
    if training_info is None:
        print(f"❌ ERROR: No training metadata found in explainer_metadata!")
        print(f"   Cannot fix results without training metadata.")
        print(f"   Please ensure TDE has been trained for this model first.")
        return 0
    
    print(f"✅ Found training metadata:")
    print(f"   best_validation_loss: {training_info['best_validation_loss']:.6f}")
    print(f"   final_training_loss: {training_info['final_training_loss']:.6f}")
    print(f"   training_time: {training_info['training_time']:.1f}s")
    
    # Find incomplete results
    print(f"\n🔍 Finding incomplete results...")
    incomplete = find_incomplete_results(primary_use, option_number, model_name)
    
    if not incomplete:
        print(f"✅ No incomplete results found!")
        return 0
    
    # Filter by variant if specified
    if variants_to_fix:
        incomplete = [r for r in incomplete if r['variant_name'] in variants_to_fix]
        if not incomplete:
            print(f"⚠️  No incomplete results for specified variants: {variants_to_fix}")
            return 0
    
    print(f"📊 Found {len(incomplete)} incomplete rows")
    
    # Group by variant
    by_variant = {}
    for item in incomplete:
        variant = item['variant_name']
        if variant not in by_variant:
            by_variant[variant] = []
        by_variant[variant].append(item)
    
    for variant, items in by_variant.items():
        print(f"\n  {variant}: {len(items)} rows with NULL training metadata")
    
    # Confirm
    confirm = input(f"\n👉 Update these {len(incomplete)} rows? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled.")
        return 0
    
    # Update rows
    print(f"\n🔧 Updating rows...")
    updated_count = 0
    
    def _update_row(item):
        conn = sqlite3.connect(ABLATION_DB)
        conn.isolation_level = None
        cursor = conn.cursor()
        
        try:
            cursor.execute('BEGIN IMMEDIATE')
            
            cursor.execute('''
                UPDATE ablation_results
                SET best_validation_loss = ?,
                    final_training_loss = ?,
                    training_time = ?,
                    timestamp = ?
                WHERE primary_use = ? AND option_number = ? AND model_name = ?
                  AND ablation_category = ? AND variant_name = ? AND sample_idx = ?
            ''', (
                training_info['best_validation_loss'],
                training_info['final_training_loss'],
                training_info['training_time'],
                datetime.now().isoformat(),
                item['primary_use'],
                item['option_number'],
                item['model_name'],
                item['ablation_category'],
                item['variant_name'],
                item['sample_idx']
            ))
            
            cursor.execute('COMMIT')
            conn.close()
            return True
            
        except Exception as e:
            cursor.execute('ROLLBACK')
            conn.close()
            print(f"   ❌ Failed to update {item['variant_name']}/sample_{item['sample_idx']}: {e}")
            return False
    
    for item in tqdm(incomplete, desc="Updating", unit="row"):
        try:
            if db_execute_with_retry(lambda: _update_row(item), "Update row"):
                updated_count += 1
        except Exception as e:
            print(f"\n❌ Error updating row: {e}")
            continue
    
    print(f"\n{'='*80}")
    print(f"✅ Updated {updated_count}/{len(incomplete)} rows")
    print(f"{'='*80}")
    
    # Verify
    print(f"\n🔍 Verifying...")
    incomplete_after = find_incomplete_results(primary_use, option_number, model_name)
    
    if variants_to_fix:
        incomplete_after = [r for r in incomplete_after if r['variant_name'] in variants_to_fix]
    
    if not incomplete_after:
        print(f"✅ All rows now complete!")
    else:
        print(f"⚠️  {len(incomplete_after)} rows still incomplete")
        show_incomplete_summary(primary_use, option_number, model_name)
    
    return updated_count
def verify_database():
    """Verify database contents"""
    
    def _verify():
        conn = sqlite3.connect(ABLATION_DB)
        cursor = conn.cursor()
        
        print("\n" + "="*80)
        print("🔍 DATABASE VERIFICATION")
        print("="*80)
        
        cursor.execute("SELECT COUNT(*) FROM ablation_results")
        results_count = cursor.fetchone()[0]
        print(f"  ablation_results: {results_count} rows")
        
        cursor.execute("SELECT COUNT(*) FROM ablation_shap_values")
        shap_count = cursor.fetchone()[0]
        print(f"  ablation_shap_values: {shap_count} rows")
        
        if results_count > 0:
            cursor.execute('''
                SELECT primary_use, option_number, model_name, variant_name, COUNT(*) as n_samples
                FROM ablation_results
                GROUP BY primary_use, option_number, model_name, variant_name
            ''')
            configs = cursor.fetchall()
            print(f"\n  Configurations saved: {len(configs)}")
            for use, opt, model, variant, n in configs[:10]:
                print(f"    • {use}/{opt}/{model}/{variant}: {n} samples")
            if len(configs) > 10:
                print(f"    ... and {len(configs) - 10} more")
        
        conn.close()
        print("="*80)
    
    try:
        db_execute_with_retry(_verify, "Verify database")
    except Exception as e:
        print(f"❌ Verification error: {e}")


def find_incomplete_results(primary_use=None, option_number=None, model_name=None):
    """
    Find all results with NULL values in critical columns.
    
    Returns list of incomplete results with details about which columns are NULL.
    """
    
    def _find():
        conn = sqlite3.connect(ABLATION_DB)
        cursor = conn.cursor()
        
        # Build WHERE clause
        where_parts = []
        params = []
        
        if primary_use is not None:
            where_parts.append("primary_use = ?")
            params.append(primary_use)
        
        if option_number is not None:
            where_parts.append("option_number = ?")
            params.append(int(option_number))
        
        if model_name is not None:
            where_parts.append("model_name = ?")
            params.append(model_name)
        
        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        
        # Get all results
        query = f'''
            SELECT 
                primary_use, option_number, model_name, 
                ablation_category, variant_name, sample_idx,
                fidelity, sparsity, complexity,
                completeness_error, computation_time,
                best_validation_loss, final_training_loss, training_time,
                n_parameters
            FROM ablation_results
            WHERE {where_clause}
        '''
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        incomplete = []
        
        for row in rows:
            null_columns = []
            
            # Check critical metrics
            if row[6] is None: null_columns.append('fidelity')
            if row[7] is None: null_columns.append('sparsity')
            if row[8] is None: null_columns.append('complexity')
            if row[9] is None: null_columns.append('completeness_error')
            if row[10] is None: null_columns.append('computation_time')
            
            # Check training metadata
            if row[11] is None: null_columns.append('best_validation_loss')
            if row[12] is None: null_columns.append('final_training_loss')
            if row[13] is None: null_columns.append('training_time')
            
            # If any critical column is NULL, add to incomplete list
            if null_columns:
                incomplete.append({
                    'primary_use': row[0],
                    'option_number': row[1],
                    'model_name': row[2],
                    'ablation_category': row[3],
                    'variant_name': row[4],
                    'sample_idx': row[5],
                    'null_columns': null_columns
                })
        
        return incomplete
    
    try:
        return db_execute_with_retry(_find, "Find incomplete results")
    except:
        return []


def show_incomplete_summary(primary_use=None, option_number=None, model_name=None):
    """Show summary of incomplete results"""
    
    incomplete = find_incomplete_results(primary_use, option_number, model_name)
    
    if not incomplete:
        print("\n✅ No incomplete results found! All rows have complete data.")
        return
    
    print(f"\n{'='*80}")
    print(f"⚠️  INCOMPLETE RESULTS: {len(incomplete)} rows with NULL values")
    print(f"{'='*80}")
    
    # Group by configuration
    by_config = {}
    for item in incomplete:
        key = f"{item['primary_use']}/{item['option_number']}/{item['model_name']}/{item['variant_name']}"
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(item)
    
    for config, items in sorted(by_config.items()):
        print(f"\n📊 {config}: {len(items)} incomplete samples")
        
        # Count NULL columns
        null_counts = {}
        for item in items:
            for col in item['null_columns']:
                null_counts[col] = null_counts.get(col, 0) + 1
        
        print(f"   NULL columns:")
        for col, count in sorted(null_counts.items(), key=lambda x: -x[1]):
            print(f"     • {col}: {count} samples")
        
        # Show first few sample indices
        sample_indices = [item['sample_idx'] for item in items[:5]]
        if len(items) > 5:
            print(f"   Sample indices: {sample_indices} ... and {len(items)-5} more")
        else:
            print(f"   Sample indices: {sample_indices}")
    
    print(f"\n{'='*80}")
    print(f"💡 TIP: Run with force_rerun=False to re-compute only these incomplete rows")
    print(f"{'='*80}")


# ============================
# USER INPUT & MAIN
# ============================
def get_user_inputs():
    """Get user configuration"""
    print("\n" + "="*80)
    print("🔬 ABLATION STUDY SYSTEM (Skip Existing + Error Handling)")
    print("="*80)
    print(f"🔧 Device: {device}")
    print(f"💾 SHAP values: {'SAVED' if SAVE_SHAP_VALUES else 'NOT SAVED'}")
    print(f"🔄 Database retry: {MAX_DB_RETRIES} attempts with backoff")
    print("="*80)
    
    # Training configuration
    print(f"\n⏱️  Training Configuration:")
    epochs_input = input(f"   Training epochs for ablation variants [default: 5]: ").strip()
    if epochs_input:
        try:
            Config.epochs = int(epochs_input)
            print(f"   ✅ Using {Config.epochs} epochs")
        except:
            print(f"   ⚠️  Invalid, using default 5")
    else:
        print(f"   ✅ Using default 5 epochs")
    
    # Hyperparameter choice
    print(f"\n🎛️  Hyperparameter Options:")
    print(f"   0: Simple defaults (feature sampling, standard values)")
    print(f"   1: Best hyperparameters from explainer_metadata (if available)")
    
    hyperparam_input = input(f"\n👉 Choose hyperparameters [0/1, default: 0]: ").strip()
    
    if hyperparam_input == '1':
        Config.use_best_hyperparams = True
        print(f"   ✅ Will use best hyperparameters (with fallback)")
    else:
        Config.use_best_hyperparams = False
        print(f"   ✅ Will use simple defaults")
    
    # Force re-run option
    print(f"\n🔄 Existing Results:")
    print(f"   0: Skip existing results (resume from where left off) - RECOMMENDED")
    print(f"   1: Force re-run ALL (overwrite existing results)")
    
    force_input = input(f"\n👉 Choose [0/1, default: 0]: ").strip()
    
    if force_input == '1':
        Config.force_rerun = True
        print(f"   ⚠️  Will OVERWRITE existing results")
    else:
        Config.force_rerun = False
        print(f"   ✅ Will SKIP existing results (resume capability)")
    
    # Select datasets
    conn = sqlite3.connect(BENCHMARK_DB)
    df = pd.read_sql_query('SELECT DISTINCT primary_use, option_number, model_name FROM prediction_performance', conn)
    conn.close()
    
    uses = sorted(df['primary_use'].unique())
    print(f"\n📊 Available primary uses ({len(uses)} total):")
    for i, use in enumerate(uses):
        count = len(df[df['primary_use'] == use])
        print(f"  {i}: {use} ({count} configurations)")
    
    use_input = input(f"\n👉 Select primary uses [Enter=ALL]: ").strip()
    
    if not use_input:
        selected_uses = uses
        print(f"   ✅ Selected: ALL ({len(uses)} primary uses)")
    else:
        try:
            indices = [int(x.strip()) for x in use_input.split(',')]
            selected_uses = [uses[i] for i in indices if 0 <= i < len(uses)]
            print(f"   ✅ Selected: {', '.join(selected_uses)}")
        except:
            selected_uses = uses
            print(f"   ⚠️  Invalid, using ALL")
    
    print(f"\n🔬 Categories: architecture, loss_terms")
    ablation_types = ['architecture', 'loss_terms']
    
    all_configs = []
    
    for primary_use in selected_uses:
        df_use = df[df['primary_use'] == primary_use]
        
        for option_number in sorted(df_use['option_number'].unique()):
            option_number = int(option_number)
            df_option = df_use[df_use['option_number'] == option_number]
            
            for model_name in sorted(df_option['model_name'].unique()):
                all_configs.append({
                    'primary_use': primary_use,
                    'option_number': option_number,
                    'model_name': model_name,
                    'ablation_types': ablation_types
                })
    
    return all_configs


def main():
    """Main entry point"""
    init_ablation_database()
    
    configs = get_user_inputs()
    
    if not configs:
        print("\n❌ No configurations selected.")
        return
    
    print("\n" + "="*80)
    print("📋 CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Total configurations: {len(configs)}")
    print(f"Epochs per variant: {Config.epochs}")
    print(f"Hyperparameters: {'Best (from explainer_metadata)' if Config.use_best_hyperparams else 'Simple defaults'}")
    print(f"Mode: {'FORCE RE-RUN (overwrite)' if Config.force_rerun else 'SKIP EXISTING (resume)'}")
    print(f"Database retry: {MAX_DB_RETRIES} attempts")
    print(f"NULL validation: ENABLED (checks all critical columns)")
    print("="*80)
    
    confirm = input("\n👉 Proceed? (y/n) [y]: ").strip().lower()
    if confirm == 'n':
        print("❌ Cancelled.")
        return
    
    with tqdm(total=len(configs), desc="📊 Overall Progress", unit="config", position=1) as overall_pbar:
        
        for i, config in enumerate(configs, 1):
            overall_pbar.set_description(f"📊 Config {i}/{len(configs)}")
            
            try:
                run_ablation_study(
                    config['primary_use'],
                    config['option_number'],
                    config['model_name'],
                    config['ablation_types']
                )
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue  # Continue with next config
            
            overall_pbar.update(1)
    
    print("\n" + "="*80)
    print("✅ ALL COMPLETE!")
    print("="*80)
    
    verify_database()
    
    # Check for any incomplete results
    print("\n🔍 Checking for incomplete results (NULL values)...")
    
    all_incomplete = []
    configs_with_incomplete = []
    
    # Check across all processed configurations
    for config in configs:
        incomplete = find_incomplete_results(
            config['primary_use'],
            config['option_number'],
            config['model_name']
        )
        
        if incomplete:
            all_incomplete.extend(incomplete)
            configs_with_incomplete.append(config)
            print(f"\n⚠️  Found {len(incomplete)} incomplete rows in "
                  f"{config['primary_use']}/{config['option_number']}/{config['model_name']}")
            show_incomplete_summary(
                config['primary_use'],
                config['option_number'],
                config['model_name']
            )
    
    # Offer to fix incomplete results
    if all_incomplete:
        print("\n" + "="*80)
        print(f"⚠️  TOTAL: {len(all_incomplete)} incomplete rows across {len(configs_with_incomplete)} configurations")
        print("="*80)
        
        print("\n💡 These incomplete rows are likely missing training metadata.")
        print("   This can be fixed by loading from explainer_metadata table.")
        
        fix_option = input("\n👉 Auto-fix incomplete results? (y/n) [y]: ").strip().lower()
        
        if fix_option != 'n':
            print("\n🔧 Starting auto-fix...")
            
            total_fixed = 0
            for config in configs_with_incomplete:
                print(f"\n{'='*80}")
                print(f"Fixing: {config['primary_use']}/{config['option_number']}/{config['model_name']}")
                print(f"{'='*80}")
                
                # Only fix full_tde and full_loss variants (which load from xai_results)
                variants_to_fix = ['full_tde', 'full_loss']
                
                fixed = fix_incomplete_results(
                    config['primary_use'],
                    config['option_number'],
                    config['model_name'],
                    variants_to_fix
                )
                
                total_fixed += fixed
            
            print("\n" + "="*80)
            print(f"✅ AUTO-FIX COMPLETE: {total_fixed} rows updated")
            print("="*80)
            
            # Re-check
            print("\n🔍 Final verification...")
            remaining_incomplete = []
            for config in configs:
                incomplete = find_incomplete_results(
                    config['primary_use'],
                    config['option_number'],
                    config['model_name']
                )
                remaining_incomplete.extend(incomplete)
            
            if remaining_incomplete:
                print(f"⚠️  {len(remaining_incomplete)} rows still incomplete after fix")
            else:
                print(f"✅ All results now complete!")
    else:
        print("\n✅ No incomplete results found! All data is complete.")
    
    print("\n" + "="*80)
    print("💡 TIP: To manually fix incomplete results anytime, run:")
    print("   from tde_ablation import fix_incomplete_results")
    print("   fix_incomplete_results('health', 0, 'BGRU')")
    print("="*80)


if __name__ == "__main__":
    main()