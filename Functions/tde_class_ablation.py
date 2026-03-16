# ============================
# LIBRARY IMPORTS
# ============================

import os
import numpy as np
import warnings
from scipy.special import comb

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

import shap

# ============================
# DEVICE CONFIGURATION
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# TDE NETWORK ARCHITECTURE (WITH ABLATION SUPPORT)
# ============================
class TemporalExplainerNetworkAblation(nn.Module):
    """
    Temporal Deep Explainer (TDE) Network - ABLATION VERSION
    
    Additional ablation parameters:
        use_attention_gate: Enable/disable attention mechanism (default: True)
        apply_direct_input: Enable/disable direct input connection (default: True)
        apply_soft_threshold: Enable/disable soft thresholding (default: True)
    """
    
    def __init__(self, time_steps, n_features, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, use_attention_gate=True,
                 apply_direct_input=True, apply_soft_threshold=True):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.sparsity_threshold = sparsity_threshold
        self.use_attention_gate = use_attention_gate
        self.apply_direct_input = apply_direct_input
        self.apply_soft_threshold = apply_soft_threshold
        
        # ========================================
        # MAIN PATH: Dilated Temporal Convolutions
        # ========================================
        conv_layers = []
        in_channels = n_features
        
        for i in range(n_conv_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation // 2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_channels = hidden_dim
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Output projection: hidden_dim -> n_features
        self.output_proj = nn.Conv1d(hidden_dim, n_features, 1)
        
        # ========================================
        # SIDE PATH: Attention Gate (ABLATION CONTROLLED)
        # ========================================
        if use_attention_gate:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            self.gate_proj = nn.Sequential(
                nn.Conv1d(hidden_dim, n_features, 1),
                nn.Sigmoid()
            )
        else:
            self.attention = None
            self.gate_proj = None
        
        # ========================================
        # DIRECT INPUT CONNECTION (ABLATION CONTROLLED)
        # ========================================
        if apply_direct_input:
            self.input_weight = nn.Parameter(torch.zeros(time_steps, n_features))
        else:
            self.input_weight = None
        
        # Initialize output projection with small weights
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x, baseline=None):
        """
        Forward pass to generate SHAP values.
        
        Args:
            x: Input tensor (batch, time_steps, n_features)
            baseline: Baseline tensor (time_steps, n_features) or None
        
        Returns:
            torch.Tensor: SHAP values (batch, time_steps, n_features)
        """
        batch_size = x.size(0)
        
        # ========================================
        # Main convolution path
        # ========================================
        h = x.permute(0, 2, 1)
        h = self.conv(h)
        conv_out = self.output_proj(h)
        
        # ========================================
        # Attention gating (ABLATION CONTROLLED)
        # ========================================
        if self.use_attention_gate and self.attention is not None:
            h_att = h.permute(0, 2, 1)
            attn_out, _ = self.attention(h_att, h_att, h_att)
            attn_out = attn_out.permute(0, 2, 1)
            
            gate = self.gate_proj(attn_out)
            conv_out = conv_out * gate
        
        conv_out = conv_out.permute(0, 2, 1)
        
        # ========================================
        # Direct input contribution (ABLATION CONTROLLED)
        # ========================================
        if self.apply_direct_input and self.input_weight is not None:
            if baseline is not None:
                if baseline.dim() == 2:
                    baseline = baseline.unsqueeze(0)
                diff = x - baseline
            else:
                diff = x
            
            input_contrib = diff * torch.tanh(self.input_weight).unsqueeze(0)
            output = conv_out + input_contrib
        else:
            output = conv_out
        
        # ========================================
        # Soft thresholding (ABLATION CONTROLLED)
        # ========================================
        if self.apply_soft_threshold:
            output = torch.sign(output) * torch.relu(torch.abs(output) - self.sparsity_threshold)
        
        return output


# ============================
# TDE TRAINER CLASS (ABLATION VERSION - COMPLETENESS RENAMED)
# ============================
class TemporalDeepExplainerAblation:
    """
    Temporal Deep Explainer Trainer - ABLATION VERSION
    
    RENAMED: efficiency_lambda → completeness_lambda
    This measures how well SHAP values sum to the total prediction difference (completeness property).
    """
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, 
                 min_lr=1e-6, l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1, 
                 completeness_lambda=0.1, sparsity_lambda=0.1, target_sparsity=0.70,  # RENAMED
                 weight_decay=1e-4, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, optimizer_type='adam', learning_rate=1e-3,
                 window_size=6, paired_sampling=True, samples_per_feature=2,
                 masking_mode='window',
                 use_attention_gate=True, apply_direct_input=True, apply_soft_threshold=True,
                 **kwargs):
        
        self.device = device
        
        # Training hyperparameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        # Loss weights (RENAMED: efficiency → completeness)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.smoothness_lambda = smoothness_lambda
        self.completeness_lambda = completeness_lambda  # RENAMED
        self.sparsity_lambda = sparsity_lambda
        self.target_sparsity = target_sparsity
        
        # Optimizer settings
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        # Network architecture
        self.hidden_dim = hidden_dim
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.sparsity_threshold = sparsity_threshold
        self.n_attention_heads = n_attention_heads
        
        # Ablation flags
        self.use_attention_gate = use_attention_gate
        self.apply_direct_input = apply_direct_input
        self.apply_soft_threshold = apply_soft_threshold
        
        # Masking settings
        self.window_size = window_size
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        self.masking_mode = masking_mode
        
        # Model components
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.n_windows = None
        self.model_predict_func = None
        
        # Training state
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # GPU optimization
        self._gpu_model = None
        self._model_on_gpu = False
        self._baseline_cache = None
        self._shapley_probs_features = None
        
        # Mixed precision
        self.scaler = GradScaler() if torch.cuda.is_available() else None
    
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        """Initialize explainer network and compute baseline."""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        if len(feature_names) != self.n_features:
            raise ValueError(f"feature_names length ({len(feature_names)}) must match n_features ({self.n_features})")
        
        # Cache GPU model
        if gpu_model is not None:
            self._gpu_model = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        else:
            self._gpu_model = None
            self._model_on_gpu = False
        
        # Compute baseline
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Compute base prediction
        if self._model_on_gpu:
            with torch.no_grad():
                base_input = self.baseline.unsqueeze(0)
                base_raw = self._gpu_model(base_input)
                if base_raw.ndim > 1 and base_raw.shape[1] > 0:
                    self.base_pred = base_raw[:, 0].flatten()[0]
                else:
                    self.base_pred = base_raw.flatten()[0]
                
                if not isinstance(self.base_pred, torch.Tensor):
                    self.base_pred = torch.tensor(self.base_pred, dtype=torch.float32, device=self.device)
        else:
            base_np = self.baseline.unsqueeze(0).cpu().numpy()
            base_raw = model_predict_func(base_np)
            
            if isinstance(base_raw, torch.Tensor):
                base_raw = base_raw.cpu().numpy()
            base_raw = np.atleast_1d(base_raw).flatten()[0]
            self.base_pred = torch.tensor(float(base_raw), dtype=torch.float32, device=self.device)
        
        # Initialize explainer network (ABLATION VERSION - NO COMPILE)
        self.explainer = TemporalExplainerNetworkAblation(
            self.time_steps, self.n_features, 
            self.hidden_dim, self.n_conv_layers,
            self.kernel_size, self.dropout_rate, 
            self.sparsity_threshold, self.n_attention_heads,
            self.use_attention_gate, self.apply_direct_input, self.apply_soft_threshold
        ).to(self.device)
        
        # Pre-compute Shapley kernel
        _, self._shapley_probs_features = self._compute_shapley_kernel(self.n_features)
        
        # Reset baseline cache
        self._baseline_cache = None

    
    def _compute_shapley_kernel(self, d):
        """Compute Shapley kernel weights for coalition sampling."""
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float64)
        
        log_binom = (
            torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64)) 
            - torch.lgamma(k_values + 1) 
            - torch.lgamma(d - k_values + 1)
        )
        binom_coeffs = torch.exp(log_binom)
        
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        weights = weights.float()
        probs = weights / weights.sum()
        
        return weights, probs
    
    def _generate_window_masks(self, batch_size):
        """Generate window-based masks for coalition sampling."""
        total = batch_size * self.samples_per_feature
        
        masks = torch.ones(total, self.time_steps, self.n_features, device=self.device)
        
        max_windows = max(2, self.n_windows)
        n_windows_to_mask = torch.randint(1, max_windows, (total,), device=self.device)
        n_features_to_mask = torch.randint(1, self.n_features + 1, (total,), device=self.device)
        
        window_rand = torch.rand(total, self.n_windows, device=self.device)
        feature_rand = torch.rand(total, self.n_features, device=self.device)
        
        for i in range(total):
            _, top_windows = torch.topk(window_rand[i], n_windows_to_mask[i].item())
            _, top_features = torch.topk(feature_rand[i], n_features_to_mask[i].item())
            
            for w_idx in top_windows:
                start = w_idx.item() * self.window_size
                end = min(start + self.window_size, self.time_steps)
                masks[i, start:end, top_features] = 0.0
        
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks

    def _generate_feature_masks(self, batch_size):
        """Generate feature-based masks for coalition sampling."""
        probs_f = self._shapley_probs_features
        total = batch_size * self.samples_per_feature
        
        k_idx = torch.multinomial(probs_f, total, replacement=True)
        k_samples = torch.arange(1, self.n_features, device=self.device)[k_idx]
        
        rand = torch.rand(total, self.n_features, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        
        masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks
    
    def _generate_masks(self, batch_size):
        """Generate masks based on selected masking mode."""
        if self.masking_mode == 'window':
            return self._generate_window_masks(batch_size)
        return self._generate_feature_masks(batch_size)
        
    def _get_predictions(self, inputs):
        """Get predictions from black-box model."""
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                elif inputs.device != self.device:
                    inputs = inputs.to(self.device)
                
                pred = self._gpu_model(inputs)
                if pred.ndim > 1 and pred.shape[1] > 0:
                    return pred[:, 0]
                return pred.flatten()
            else:
                if isinstance(inputs, torch.Tensor):
                    inputs_np = inputs.cpu().numpy()
                else:
                    inputs_np = inputs
                
                preds = self.model_predict_func(inputs_np)
                return torch.tensor(
                    np.atleast_1d(preds).flatten(),
                    dtype=torch.float32, device=self.device
                )
    
    def _process_batch(self, X_batch, optimizer):
        """Process single training batch (RENAMED: efficiency → completeness)."""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device, non_blocking=True)
        
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_masks(batch_size)
        
        total = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat_factor, 1, 1)[:total]
        
        if self._baseline_cache is None or self._baseline_cache.size(0) < total:
            max_cache_size = max(total, self.batch_size * self.samples_per_feature * 4)
            self._baseline_cache = self.baseline.unsqueeze(0).expand(
                max_cache_size, -1, -1
            ).contiguous().clone()
        baseline_paired = self._baseline_cache[:total]
        
        masked = torch.addcmul(baseline_paired, X_paired - baseline_paired, masks)
        
        preds_masked = self._get_predictions(masked)
        
        if self.paired_sampling:
            n_unique = total // 2
            preds_unique = self._get_predictions(X_paired[:n_unique])
            preds_orig = preds_unique.repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)
        
        if not isinstance(preds_masked, torch.Tensor):
            preds_masked = torch.tensor(preds_masked, dtype=torch.float32, device=self.device)
        if not isinstance(preds_orig, torch.Tensor):
            preds_orig = torch.tensor(preds_orig, dtype=torch.float32, device=self.device)
        
        use_amp = self.scaler is not None and self.device.type == 'cuda'
        
        with autocast(enabled=use_amp):
            phi = self.explainer(X_paired, self.baseline)
            
            # Coalition loss
            masked_sum = (masks * phi).sum(dim=(1, 2))
            pred_diff = preds_masked - self.base_pred
            coalition_loss = ((masked_sum - pred_diff) ** 2).mean()
            
            # Completeness loss (RENAMED from efficiency)
            phi_sum = phi.sum(dim=(1, 2))
            orig_diff = preds_orig - self.base_pred
            completeness_loss = self.completeness_lambda * ((phi_sum - orig_diff) ** 2).mean()
            
            # Temporal smoothness loss
            if phi.size(1) > 1:
                smooth_loss = self.smoothness_lambda * (
                    phi[:, 1:, :] - phi[:, :-1, :]
                ).pow(2).mean()
            else:
                smooth_loss = torch.tensor(0.0, device=self.device)
            
            # Regularization losses
            phi_abs = torch.abs(phi)
            l1_loss = self.l1_lambda * phi_abs.mean()
            l2_loss = self.l2_lambda * (phi ** 2).mean()
            
            # Sparsity loss
            with torch.no_grad():
                max_val = phi_abs.max()
                threshold = max_val * 0.01 if max_val > 1e-10 else 1e-10
                current_sparsity = (phi_abs < threshold).float().mean()
            sparsity_loss = self.sparsity_lambda * (current_sparsity - self.target_sparsity) ** 2
            
            # Total loss (RENAMED: eff_loss → completeness_loss)
            loss = coalition_loss + completeness_loss + smooth_loss + l1_loss + l2_loss + sparsity_loss
        
        if not torch.isfinite(loss):
            return float('inf')
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
            optimizer.step()
        
        return loss.item()
    
    def _validate(self, X_val):
        """Compute validation loss (measures completeness)."""
        self.explainer.eval()
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val)),
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        total_loss, n_batches = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                phi = self.explainer(X_batch, self.baseline)
                preds = self._get_predictions(X_batch)
                
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
                
                # Validation loss = completeness error
                completeness_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                
                if torch.isfinite(completeness_err):
                    total_loss += completeness_err.item()
                    n_batches += 1
        
        self.explainer.train()
        
        return total_loss / max(n_batches, 1) if n_batches > 0 else float('inf')
    
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        """Train the TDE explainer."""
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model=gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')
        
        use_cuda = self.device.type == 'cuda'
        num_workers = 4 if use_cuda else 0
        
        effective_batch_size = min(self.batch_size, len(X_train) - 1)
        if effective_batch_size < 1:
            if self.verbose:
                print(f"    [ERROR] Not enough training samples")
            return float('inf')
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train)),
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        opt_cls = {
            'adam': torch.optim.Adam, 
            'adamw': torch.optim.AdamW
        }.get(self.optimizer_type, torch.optim.Adam)
        
        optimizer = opt_cls(
            self.explainer.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=2, factor=0.5, min_lr=self.min_lr
        )
        
        best_val, best_weights, no_improve = float('inf'), None, 0
        
        for epoch in range(self.n_epochs):
            self.explainer.train()
            epoch_loss, n_batches = 0.0, 0
            
            for (X_batch,) in loader:
                batch_loss = self._process_batch(X_batch, optimizer)
                if batch_loss != float('inf'):
                    epoch_loss += batch_loss
                    n_batches += 1
            
            if n_batches == 0:
                if self.verbose:
                    print(f"    [ERROR] All batches failed at epoch {epoch + 1}")
                return float('inf')
            
            epoch_loss /= n_batches
            val_loss = self._validate(X_val)
            
            if val_loss == float('inf'):
                if self.verbose:
                    print(f"    [ERROR] Validation failed at epoch {epoch + 1}")
                continue
            
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break

        if best_weights:
            self.explainer.load_state_dict(best_weights)
        
        self.best_loss = best_val
        self._baseline_cache = None
        
        return best_val

    
    def explain(self, instance):
        """Generate SHAP values for a single instance."""
        if self.explainer is None:
            raise ValueError("Explainer not trained. Call train() first.")
        
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        
        instance = instance.to(self.device)
        
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instance, self.baseline).cpu().numpy()[0]
        
        return phi