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
# DEVICE CONFIGURATION  <-- ADD THIS SECTION
# ============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================
# TDE NETWORK ARCHITECTURE
# ============================
class TemporalExplainerNetwork(nn.Module):
    """
    Temporal Deep Explainer (TDE) Network v5.2
    
    Architecture:
    1. Dilated Conv (main path) - captures local temporal patterns
    2. Attention Gate (side path) - learns global importance weights
    3. Gated combination - conv_out * sigmoid(attention_weights)
    4. Direct Input Connection - ensures input-dependence (critical for SHAP)
    5. Soft Thresholding - promotes sparsity
    
    Args:
        time_steps: Number of time steps in input sequence
        n_features: Number of input features
        hidden_dim: Hidden dimension for convolutions (default: 128)
        n_conv_layers: Number of dilated conv layers (default: 2)
        kernel_size: Convolution kernel size (default: 3)
        dropout_rate: Dropout probability (default: 0.2)
        sparsity_threshold: Soft threshold for sparsity (default: 0.01)
        n_attention_heads: Number of attention heads (default: 4)
        use_attention_gate: Whether to use attention gating (default: True)
    """
    
    def __init__(self, time_steps, n_features, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, use_attention_gate=True):
        super().__init__()
        
        self.time_steps = time_steps
        self.n_features = n_features
        self.sparsity_threshold = sparsity_threshold
        self.use_attention_gate = use_attention_gate
        
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
        # SIDE PATH: Attention Gate (Optional)
        # ========================================
        if use_attention_gate:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_attention_heads,
                dropout=dropout_rate,
                batch_first=True
            )
            # Gate projection: maps attention output to [0,1] weights
            self.gate_proj = nn.Sequential(
                nn.Conv1d(hidden_dim, n_features, 1),
                nn.Sigmoid()  # Bounded [0, 1] for gating
            )
        
        # ========================================
        # DIRECT INPUT CONNECTION (Critical for SHAP)
        # ========================================
        # Learnable weight for input contribution
        self.input_weight = nn.Parameter(torch.zeros(time_steps, n_features))
        
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
        # Reshape for Conv1d: (batch, features, time)
        h = x.permute(0, 2, 1)
        h = self.conv(h)  # (batch, hidden_dim, time)
        
        # Project to output features
        conv_out = self.output_proj(h)  # (batch, n_features, time)
        
        # ========================================
        # Attention gating (if enabled)
        # ========================================
        if self.use_attention_gate:
            # Attention expects (batch, time, hidden_dim)
            h_att = h.permute(0, 2, 1)
            attn_out, _ = self.attention(h_att, h_att, h_att)
            attn_out = attn_out.permute(0, 2, 1)  # (batch, hidden_dim, time)
            
            # Generate gate weights [0, 1]
            gate = self.gate_proj(attn_out)  # (batch, n_features, time)
            
            # Apply gating - modulates conv output by attention
            conv_out = conv_out * gate
        
        # Reshape back: (batch, time, features)
        conv_out = conv_out.permute(0, 2, 1)
        
        # ========================================
        # Direct input contribution
        # ========================================
        if baseline is not None:
            if baseline.dim() == 2:
                baseline = baseline.unsqueeze(0)
            diff = x - baseline
        else:
            diff = x
        
        # Learnable input weight with tanh to bound [-1, 1]
        input_contrib = diff * torch.tanh(self.input_weight).unsqueeze(0)
        
        # ========================================
        # Combine and apply soft thresholding
        # ========================================
        output = conv_out + input_contrib
        
        # Soft thresholding for sparsity: sign(x) * max(|x| - threshold, 0)
        output = torch.sign(output) * torch.relu(torch.abs(output) - self.sparsity_threshold)
        
        return output

# ============================
# TDE TRAINER CLASS
# ============================
class TemporalDeepExplainer:
    """
    Trainer for Temporal Deep Explainer (TDE) - GPU Optimized.
    
    Training approach:
    - Coalition-based loss (Shapley kernel weighting)
    - Efficiency constraint (SHAP values sum to prediction - baseline)
    - Temporal smoothness regularization
    - L1 + L2 regularization for sparsity
    - Target sparsity loss
    
    Masking strategies:
    - 'window': Masks contiguous time windows (preserves temporal structure)
    - 'feature': Masks entire features across all time steps
    
    Args:
        n_epochs: Maximum training epochs (default: 100)
        batch_size: Training batch size (default: 256)
        patience: Early stopping patience (default: 5)
        verbose: Print training progress (default: True)
        min_lr: Minimum learning rate (default: 1e-6)
        l1_lambda: L1 regularization weight (default: 0.01)
        l2_lambda: L2 regularization weight (default: 0.01)
        smoothness_lambda: Temporal smoothness weight (default: 0.1)
        efficiency_lambda: Efficiency constraint weight (default: 0.1)
        sparsity_lambda: Target sparsity loss weight (default: 0.1)
        target_sparsity: Target sparsity level (default: 0.70)
        weight_decay: Optimizer weight decay (default: 1e-4)
        hidden_dim: Network hidden dimension (default: 128)
        n_conv_layers: Number of conv layers (default: 2)
        kernel_size: Convolution kernel size (default: 3)
        dropout_rate: Dropout probability (default: 0.2)
        sparsity_threshold: Soft threshold value (default: 0.01)
        n_attention_heads: Number of attention heads (default: 4)
        optimizer_type: 'adam' or 'adamw' (default: 'adam')
        learning_rate: Initial learning rate (default: 1e-3)
        window_size: Size of masking windows (default: 6)
        paired_sampling: Use paired mask sampling (default: True)
        samples_per_feature: Samples per feature for masking (default: 2)
        masking_mode: 'window' or 'feature' (default: 'window')
    """
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, 
                 min_lr=1e-6, l1_lambda=0.01, l2_lambda=0.01, smoothness_lambda=0.1, 
                 efficiency_lambda=0.1, sparsity_lambda=0.1, target_sparsity=0.70,
                 weight_decay=1e-4, hidden_dim=128, n_conv_layers=2,
                 kernel_size=3, dropout_rate=0.2, sparsity_threshold=0.01,
                 n_attention_heads=4, optimizer_type='adam', learning_rate=1e-3,
                 window_size=6, paired_sampling=True, samples_per_feature=2,
                 masking_mode='window', **kwargs):
        
        self.device = device
        
        # Training hyperparameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        # Loss weights
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda
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
        
        # Masking settings
        self.window_size = window_size
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        self.masking_mode = masking_mode
        
        # Model components (initialized during setup)
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
        
        # GPU optimization cache
        self._gpu_model = None
        self._model_on_gpu = False
        self._baseline_cache = None
        self._shapley_probs_features = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Store init params for saving/loading
        self._init_params = {k: v for k, v in locals().items() 
                            if k not in ('self', 'kwargs')}
    
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        """
        Initialize explainer network and compute baseline.
        
        Args:
            X_train: Training data (n_samples, time_steps, n_features)
            model_predict_func: Function to get predictions from black-box model
            feature_names: List of feature names
            gpu_model: Optional PyTorch model on GPU for direct inference
        """
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.n_windows = (self.time_steps + self.window_size - 1) // self.window_size
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        # Validate feature_names length
        if len(feature_names) != self.n_features:
            raise ValueError(f"feature_names length ({len(feature_names)}) must match n_features ({self.n_features})")
        
        # Cache GPU model for direct inference (bypasses CPU transfer)
        if gpu_model is not None:
            self._gpu_model = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        else:
            self._gpu_model = None
            self._model_on_gpu = False
        
        # Compute baseline as median of training data
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Compute base prediction f(baseline) - FIXED: Better type handling
        if self._model_on_gpu:
            with torch.no_grad():
                base_input = self.baseline.unsqueeze(0)
                base_raw = self._gpu_model(base_input)
                if base_raw.ndim > 1 and base_raw.shape[1] > 0:
                    self.base_pred = base_raw[:, 0].flatten()[0]
                else:
                    self.base_pred = base_raw.flatten()[0]
                
                if not isinstance(self.base_pred, torch.Tensor):
                    self.base_pred = torch.tensor(self.base_pred, dtype=torch.float32, 
                                                device=self.device)
        else:
            base_np = self.baseline.unsqueeze(0).cpu().numpy()
            base_raw = model_predict_func(base_np)
            
            if isinstance(base_raw, torch.Tensor):
                base_raw = base_raw.cpu().numpy()
            base_raw = np.atleast_1d(base_raw).flatten()[0]
            self.base_pred = torch.tensor(
                float(base_raw),
                dtype=torch.float32, device=self.device
            )
        
        # Initialize explainer network
        self.explainer = TemporalExplainerNetwork(
            self.time_steps, self.n_features, self.hidden_dim, self.n_conv_layers,
            self.kernel_size, self.dropout_rate, self.sparsity_threshold,
            self.n_attention_heads
        ).to(self.device)
        
        # Pre-compute Shapley kernel for feature masking
        _, self._shapley_probs_features = self._compute_shapley_kernel(self.n_features)
        
        # Reset baseline cache
        self._baseline_cache = None
        
        # Compile model for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.explainer = torch.compile(self.explainer, mode='reduce-overhead')
            except Exception:
                pass

    
    def _compute_shapley_kernel(self, d):
        """
        Compute Shapley kernel weights for coalition sampling.
        
        Args:
            d: Number of features/elements
        
        Returns:
            tuple: (weights, probabilities) as tensors on device
        """
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float64)
        
        # Use log-gamma for numerical stability
        log_binom = (
            torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64)) 
            - torch.lgamma(k_values + 1) 
            - torch.lgamma(d - k_values + 1)
        )
        binom_coeffs = torch.exp(log_binom)
        
        # Shapley kernel formula
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        weights = weights.float()
        probs = weights / weights.sum()
        
        return weights, probs
    
    def _generate_window_masks(self, batch_size):
        """
        Generate window-based masks for coalition sampling.
        
        Masks contiguous time windows while preserving temporal structure.
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            torch.Tensor: Binary masks (total_samples, time_steps, n_features)
        """
        total = batch_size * self.samples_per_feature
        
        # Pre-allocate masks on GPU (all ones = all features present)
        masks = torch.ones(total, self.time_steps, self.n_features, device=self.device)
        
        # Sample number of windows and features to mask
        max_windows = max(2, self.n_windows)
        n_windows_to_mask = torch.randint(1, max_windows, (total,), device=self.device)
        n_features_to_mask = torch.randint(1, self.n_features + 1, (total,), device=self.device)
        
        # Random selection matrices
        window_rand = torch.rand(total, self.n_windows, device=self.device)
        feature_rand = torch.rand(total, self.n_features, device=self.device)
        
        # Create masks
        for i in range(total):
            _, top_windows = torch.topk(window_rand[i], n_windows_to_mask[i].item())
            _, top_features = torch.topk(feature_rand[i], n_features_to_mask[i].item())
            
            for w_idx in top_windows:
                start = w_idx.item() * self.window_size
                end = min(start + self.window_size, self.time_steps)
                masks[i, start:end, top_features] = 0.0
        
        # Add complementary masks for paired sampling
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks

    def _generate_feature_masks(self, batch_size):
        """
        Generate feature-based masks for coalition sampling.
        
        Masks entire features across all time steps.
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            torch.Tensor: Binary masks (total_samples, time_steps, n_features)
        """
        probs_f = self._shapley_probs_features
        total = batch_size * self.samples_per_feature
        
        # Sample coalition sizes from Shapley kernel
        k_idx = torch.multinomial(probs_f, total, replacement=True)
        k_samples = torch.arange(1, self.n_features, device=self.device)[k_idx]
        
        # Generate random masks
        rand = torch.rand(total, self.n_features, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        
        # Expand to time dimension
        masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
        
        # Add complementary masks for paired sampling
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks
    
    def _generate_masks(self, batch_size):
        """
        Generate masks based on selected masking mode.
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            torch.Tensor: Binary masks
        """
        if self.masking_mode == 'window':
            return self._generate_window_masks(batch_size)
        return self._generate_feature_masks(batch_size)
        
    def _get_predictions(self, inputs):
        """
        Get predictions from black-box model.
        
        Args:
            inputs: Input tensor or array (n_samples, time_steps, n_features)
        
        Returns:
            torch.Tensor: Predictions (n_samples,)
        """
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                # Direct GPU inference
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
                elif inputs.device != self.device:
                    inputs = inputs.to(self.device)
                
                pred = self._gpu_model(inputs)
                if pred.ndim > 1 and pred.shape[1] > 0:
                    return pred[:, 0]
                return pred.flatten()
            else:
                # CPU-based prediction function
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
        """
        Process single training batch.
        
        Args:
            X_batch: Batch tensor (batch_size, time_steps, n_features)
            optimizer: PyTorch optimizer
        
        Returns:
            float: Batch loss value
        """
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device, non_blocking=True)
        
        # Expand batch for multiple samples per feature
        expanded = X_batch.repeat(self.samples_per_feature, 1, 1)
        masks = self._generate_masks(batch_size)
        
        total = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat_factor, 1, 1)[:total]
        
        # Cache baseline expansion - FIXED: Added .clone() to prevent view issues
        if self._baseline_cache is None or self._baseline_cache.size(0) < total:
            max_cache_size = max(total, self.batch_size * self.samples_per_feature * 4)
            self._baseline_cache = self.baseline.unsqueeze(0).expand(
                max_cache_size, -1, -1
            ).contiguous().clone()
        baseline_paired = self._baseline_cache[:total]
        
        # Apply masking: masked = baseline + (x - baseline) * mask
        masked = torch.addcmul(baseline_paired, X_paired - baseline_paired, masks)
        
        # Get predictions
        preds_masked = self._get_predictions(masked)
        
        if self.paired_sampling:
            n_unique = total // 2
            preds_unique = self._get_predictions(X_paired[:n_unique])
            preds_orig = preds_unique.repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)
        
        # Ensure predictions are tensors
        if not isinstance(preds_masked, torch.Tensor):
            preds_masked = torch.tensor(preds_masked, dtype=torch.float32, device=self.device)
        if not isinstance(preds_orig, torch.Tensor):
            preds_orig = torch.tensor(preds_orig, dtype=torch.float32, device=self.device)
        
        # Mixed precision training
        use_amp = self.scaler is not None and self.device.type == 'cuda'
        
        with autocast(enabled=use_amp):
            # Forward pass
            phi = self.explainer(X_paired, self.baseline)
            
            # Coalition loss
            masked_sum = (masks * phi).sum(dim=(1, 2))
            pred_diff = preds_masked - self.base_pred
            coalition_loss = ((masked_sum - pred_diff) ** 2).mean()
            
            # Efficiency loss
            phi_sum = phi.sum(dim=(1, 2))
            orig_diff = preds_orig - self.base_pred
            eff_loss = self.efficiency_lambda * ((phi_sum - orig_diff) ** 2).mean()
            
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
            
            # Sparsity loss (current_sparsity = fraction of near-zero values)
            with torch.no_grad():
                max_val = phi_abs.max()
                threshold = max_val * 0.01 if max_val > 1e-10 else 1e-10
                current_sparsity = (phi_abs < threshold).float().mean()
            sparsity_loss = self.sparsity_lambda * (current_sparsity - self.target_sparsity) ** 2
            
            # Total loss
            loss = coalition_loss + eff_loss + smooth_loss + l1_loss + l2_loss + sparsity_loss
        
        # Check for invalid loss
        if not torch.isfinite(loss):
            return float('inf')
        
        # Backpropagation
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
        """
        Compute validation loss.
        
        Args:
            X_val: Validation data (n_samples, time_steps, n_features)
        
        Returns:
            float: Average validation loss
        """
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
                
                # Validation loss = efficiency error
                eff_err = ((phi.sum(dim=(1, 2)) - (preds - self.base_pred)) ** 2).mean()
                
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n_batches += 1
        
        self.explainer.train()
        
        return total_loss / max(n_batches, 1) if n_batches > 0 else float('inf')
    
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        """
        Train the TDE explainer.
        
        Args:
            X_train: Training data (n_samples, time_steps, n_features)
            X_val: Validation data (n_samples, time_steps, n_features)
            model_predict_func: Function to get predictions from black-box model
            feature_names: List of feature names
            gpu_model: Optional PyTorch model on GPU for direct inference
        
        Returns:
            float: Best validation loss achieved
        """
        # Setup
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model=gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')
        
        # DataLoader configuration
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
        
        # Setup optimizer
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
        
        # Training loop
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
            
            # Record history
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            # Check for improvement - FIXED: Proper handling of compiled models
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                if hasattr(self.explainer, '_orig_mod'):
                    best_weights = {k: v.clone()
                                for k, v in self.explainer._orig_mod.state_dict().items()}
                else:
                    best_weights = {k: v.clone()
                                for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            # Early stopping
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break

        # Restore best weights
        if best_weights:
            if hasattr(self.explainer, '_orig_mod'):
                self.explainer._orig_mod.load_state_dict(best_weights)
            else:
                self.explainer.load_state_dict(best_weights)
        
        self.best_loss = best_val
        self._baseline_cache = None  # Cleanup
        
        return best_val

    
    def explain(self, instance):
        """
        Generate SHAP values for a single instance.
        
        Args:
            instance: Input sample - can be:
                - np.ndarray (time_steps, n_features)
                - np.ndarray (1, time_steps, n_features)
                - torch.Tensor with same shapes
        
        Returns:
            np.ndarray: SHAP values (time_steps, n_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not trained. Call train() first.")
        
        # Convert to tensor
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        
        # Ensure 3D
        if instance.ndim == 2:
            instance = instance.unsqueeze(0)
        
        instance = instance.to(self.device)
        
        # Generate SHAP values
        self.explainer.eval()
        with torch.no_grad():
            phi = self.explainer(instance, self.baseline).cpu().numpy()[0]
        
        return phi

    def save(self, path, filename="tde_explainer"):
        """
        Save trained explainer to disk.
        
        Args:
            path: Directory path
            filename: Filename without extension (default: 'tde_explainer')
        
        Returns:
            str: Full path to saved file
        """
        os.makedirs(path, exist_ok=True)
        
        # Get state dict (handle compiled model)
        if hasattr(self.explainer, '_orig_mod'):
            state_dict = self.explainer._orig_mod.state_dict()
        else:
            state_dict = self.explainer.state_dict()
        
        # Prepare base_pred for saving
        base_pred_save = self.base_pred
        if isinstance(base_pred_save, torch.Tensor):
            base_pred_save = base_pred_save.cpu()
        else:
            base_pred_save = torch.tensor(float(base_pred_save))
        
        # Create state
        state = {
            'explainer': state_dict,
            'baseline': self.baseline.cpu(),
            'base_pred': base_pred_save,
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'n_windows': self.n_windows,
            'feature_names': self.feature_names,
            'best_loss': self.best_loss,
            'history': self.history,
            'init_params': self._init_params
        }
        
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        
        return save_path

    @classmethod
    def load(cls, path, filename="tde_explainer", device_override=None):
        """
        Load trained explainer from disk.
        
        Args:
            path: Directory path
            filename: Filename without extension (default: 'tde_explainer')
            device_override: Optional device override
        
        Returns:
            TemporalDeepExplainer: Loaded explainer instance
        """
        dev = device_override or device
        load_path = os.path.join(path, f"{filename}.pt")
        state = torch.load(load_path, map_location=dev, weights_only=False)
        
        # Reconstruct explainer
        params = state.get('init_params', {})
        exp = cls(**params)
        
        # Restore attributes
        exp.device = dev
        exp.time_steps = state['time_steps']
        exp.n_features = state['n_features']
        exp.n_windows = state.get('n_windows', exp.time_steps // exp.window_size)
        exp.feature_names = state['feature_names']
        exp.baseline = state['baseline'].to(dev)
        
        # Handle base_pred
        base_pred = state['base_pred']
        if isinstance(base_pred, torch.Tensor):
            exp.base_pred = base_pred.to(dev)
        else:
            exp.base_pred = torch.tensor(float(base_pred), device=dev)
        
        exp.best_loss = state.get('best_loss', float('inf'))
        exp.history = state.get('history', {})
        
        # Initialize GPU flags
        exp._gpu_model = None
        exp._model_on_gpu = False
        exp._baseline_cache = None
        
        # Restore network
        exp.explainer = TemporalExplainerNetwork(
            exp.time_steps, exp.n_features, 
            params.get('hidden_dim', 128),
            params.get('n_conv_layers', 2), 
            params.get('kernel_size', 3),
            params.get('dropout_rate', 0.2), 
            params.get('sparsity_threshold', 0.01),
            params.get('n_attention_heads', 4)
        ).to(dev)
        
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        
        # Pre-compute Shapley kernel
        _, exp._shapley_probs_features = exp._compute_shapley_kernel(exp.n_features)
        
        return exp

# ============================
# FASTSHAP NETWORK ARCHITECTURE
# ============================
class FastSHAPNetwork(nn.Module):
    """
    FastSHAP Neural Network - Pure MLP Architecture.
    
    This is a baseline comparison method that uses element-wise masking
    on flattened input. Unlike TDE, it has NO temporal structure awareness
    and treats all features independently.
    
    Architecture:
    - Input layer: input_dim -> hidden_dim
    - Hidden layers: hidden_dim -> hidden_dim (with ReLU + Dropout)
    - Output layer: hidden_dim -> input_dim
    
    The network learns: flattened_input -> flattened_SHAP_values
    
    Args:
        input_dim: Flattened input dimension (time_steps * n_features)
        hidden_dim: Hidden layer dimension (default: 256)
        n_layers: Number of hidden layers (default: 2)
        dropout_rate: Dropout probability (default: 0.2)
    """
    
    def __init__(self, input_dim, hidden_dim=256, n_layers=2, dropout_rate=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass to generate SHAP values.
        
        Args:
            x: Flattened input tensor (batch, input_dim)
        
        Returns:
            torch.Tensor: Flattened SHAP values (batch, input_dim)
        """
        return self.network(x)

# ============================
# FASTSHAP TRAINER CLASS
# ============================
class FastSHAPExplainer:
    """
    FastSHAP Explainer - GPU Optimized Implementation.
    
    Baseline comparison method using element-wise masking on flattened input
    with pure MLP architecture (no temporal awareness).
    
    Key differences from TDE:
    - No temporal structure awareness
    - Element-wise (flattened) processing
    - No attention mechanism
    - No direct input connection
    - Simpler architecture
    
    Training approach:
    - Coalition-based loss (Shapley kernel weighting)
    - Efficiency constraint (SHAP values sum to prediction - baseline)
    - L1 regularization for sparsity
    
    Args:
        n_epochs: Maximum training epochs (default: 100)
        batch_size: Training batch size (default: 256)
        patience: Early stopping patience (default: 5)
        verbose: Print training progress (default: True)
        min_lr: Minimum learning rate (default: 1e-6)
        l1_lambda: L1 regularization weight (default: 0.01)
        efficiency_lambda: Efficiency constraint weight (default: 0.1)
        weight_decay: Optimizer weight decay (default: 1e-4)
        hidden_dim: Network hidden dimension (default: 256)
        n_layers: Number of hidden layers (default: 2)
        dropout_rate: Dropout probability (default: 0.2)
        optimizer_type: 'adam' or 'adamw' (default: 'adam')
        learning_rate: Initial learning rate (default: 1e-3)
        paired_sampling: Use paired mask sampling (default: True)
        samples_per_feature: Samples per feature for masking (default: 2)
    """
    
    def __init__(self, n_epochs=100, batch_size=256, patience=5, verbose=True, 
                 min_lr=1e-6, l1_lambda=0.01, efficiency_lambda=0.1, 
                 weight_decay=1e-4, hidden_dim=256, n_layers=2, dropout_rate=0.2,
                 optimizer_type='adam', learning_rate=1e-3, paired_sampling=True, 
                 samples_per_feature=2, **kwargs):
        
        self.device = device
        
        # Training hyperparameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose
        self.min_lr = min_lr
        
        # Loss weights
        self.l1_lambda = l1_lambda
        self.efficiency_lambda = efficiency_lambda
        self.weight_decay = weight_decay
        
        # Network architecture
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Optimizer settings
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        
        # Sampling strategy
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature
        
        # Model components (initialized during setup)
        self.explainer = None
        self.baseline = None
        self.base_pred = None
        self.feature_names = None
        self.input_dim = None
        self.time_steps = None
        self.n_features = None
        self.model_predict_func = None
        
        # Training state
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        # GPU optimization cache
        self._gpu_model = None
        self._model_on_gpu = False
        self._shapley_probs_elements = None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        
        # Store init params for saving/loading
        self._init_params = {k: v for k, v in locals().items() 
                            if k not in ('self', 'kwargs')}
    
    def _setup(self, X_train, model_predict_func, feature_names, gpu_model=None):
        """
        Initialize FastSHAP network and compute baseline.
        
        Args:
            X_train: Training data (n_samples, time_steps, n_features)
            model_predict_func: Function to get predictions from black-box model
            feature_names: List of feature names
            gpu_model: Optional PyTorch model on GPU for direct inference
        """
        # Extract dimensions
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.input_dim = self.time_steps * self.n_features
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func
        
        # Cache GPU model for direct inference
        if gpu_model is not None:
            self._gpu_model = gpu_model
            self._gpu_model.eval()
            self._model_on_gpu = True
        else:
            self._gpu_model = None
            self._model_on_gpu = False
        
        # Flatten and compute median baseline
        X_flat = X_train.reshape(len(X_train), -1)
        X_tensor = torch.FloatTensor(X_flat).to(self.device)
        self.baseline = torch.median(X_tensor, dim=0)[0]
        
        # Compute base prediction f(baseline)
        if self._model_on_gpu:
            with torch.no_grad():
                baseline_3d = self.baseline.view(1, self.time_steps, self.n_features)
                base_raw = self._gpu_model(baseline_3d)
                
                if base_raw.ndim > 1 and base_raw.shape[1] > 0:
                    self.base_pred = base_raw[:, 0].flatten()[0]
                else:
                    self.base_pred = base_raw.flatten()[0]
                
                if not isinstance(self.base_pred, torch.Tensor):
                    self.base_pred = torch.tensor(self.base_pred, dtype=torch.float32, 
                                                  device=self.device)
        else:
            baseline_np = self.baseline.unsqueeze(0).cpu().numpy().reshape(
                1, self.time_steps, self.n_features
            )
            base_raw = model_predict_func(baseline_np)
            self.base_pred = torch.tensor(
                float(np.atleast_1d(base_raw).flatten()[0]),
                dtype=torch.float32, 
                device=self.device
            )
        
        # Initialize explainer network
        self.explainer = FastSHAPNetwork(
            self.input_dim, 
            self.hidden_dim, 
            self.n_layers, 
            self.dropout_rate
        ).to(self.device)
        
        # Pre-compute Shapley kernel for element masking
        _, self._shapley_probs_elements = self._compute_shapley_kernel(self.input_dim)
        
        # Compile model for faster execution (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                self.explainer = torch.compile(self.explainer, mode='reduce-overhead')
            except Exception:
                pass
    
    def _compute_shapley_kernel(self, d):
        """
        Compute Shapley kernel weights for coalition sampling.
        
        Args:
            d: Number of features/elements
        
        Returns:
            tuple: (weights, probabilities) as tensors on device
        """
        if d <= 1:
            return torch.ones(1, device=self.device), torch.ones(1, device=self.device)
        
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float64)
        
        # Use log-gamma for numerical stability
        log_binom = (
            torch.lgamma(torch.tensor(d + 1.0, device=self.device, dtype=torch.float64)) 
            - torch.lgamma(k_values + 1) 
            - torch.lgamma(d - k_values + 1)
        )
        binom_coeffs = torch.exp(log_binom)
        
        # Shapley kernel formula
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs + 1e-10)
        weights = weights.float()
        probs = weights / weights.sum()
        
        return weights, probs
    
    def _generate_element_masks(self, batch_size):
        """
        Generate element-wise binary masks for coalition sampling.
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            torch.Tensor: Binary masks (total_samples, input_dim)
        """
        probs = self._shapley_probs_elements
        total = batch_size * self.samples_per_feature
        d = self.input_dim
        
        # Sample coalition sizes from Shapley kernel
        k_idx = torch.multinomial(probs, total, replacement=True)
        k_samples = torch.arange(1, d, device=self.device)[k_idx]
        
        # Generate random masks with k features selected
        rand = torch.rand(total, d, device=self.device)
        sorted_idx = torch.argsort(rand, dim=1)
        masks = (sorted_idx < k_samples.unsqueeze(1)).float()
        
        # Add complementary masks for paired sampling
        if self.paired_sampling:
            masks = torch.cat([masks, 1.0 - masks], dim=0)
        
        return masks
    
    def _get_predictions(self, inputs_flat):
        """
        Get predictions from black-box model.
        
        Args:
            inputs_flat: Flattened input tensor (n_samples, input_dim)
        
        Returns:
            torch.Tensor: Predictions (n_samples,)
        """
        with torch.no_grad():
            if self._model_on_gpu and self._gpu_model is not None:
                # Direct GPU inference
                if not isinstance(inputs_flat, torch.Tensor):
                    inputs_flat = torch.tensor(inputs_flat, dtype=torch.float32, 
                                               device=self.device)
                elif inputs_flat.device != self.device:
                    inputs_flat = inputs_flat.to(self.device)
                
                # Reshape to 3D for model
                inputs_3d = inputs_flat.view(-1, self.time_steps, self.n_features)
                pred = self._gpu_model(inputs_3d)
                
                if pred.ndim > 1 and pred.shape[1] > 0:
                    return pred[:, 0]
                return pred.flatten()
            else:
                # CPU-based prediction
                if isinstance(inputs_flat, torch.Tensor):
                    inputs_np = inputs_flat.cpu().numpy()
                else:
                    inputs_np = inputs_flat
                
                inputs_3d = inputs_np.reshape(-1, self.time_steps, self.n_features)
                preds = self.model_predict_func(inputs_3d)
                
                return torch.tensor(
                    np.atleast_1d(preds).flatten(),
                    dtype=torch.float32, 
                    device=self.device
                )
    
    def _process_batch(self, X_batch_flat, optimizer):
        """
        Process single training batch.
        
        Args:
            X_batch_flat: Flattened batch tensor (batch_size, input_dim)
            optimizer: PyTorch optimizer
        
        Returns:
            float: Batch loss value
        """
        batch_size = X_batch_flat.size(0)
        X_batch_flat = X_batch_flat.to(self.device, non_blocking=True)
        
        # Expand batch for multiple samples per feature
        expanded = X_batch_flat.repeat(self.samples_per_feature, 1)
        masks = self._generate_element_masks(batch_size)
        
        # Match dimensions
        total = masks.size(0)
        repeat_factor = max(1, total // (batch_size * self.samples_per_feature))
        X_paired = expanded.repeat(repeat_factor, 1)[:total]
        baseline_paired = self.baseline.unsqueeze(0).expand(total, -1)
        
        # Apply masks: masked = x * mask + baseline * (1 - mask)
        masked = X_paired * masks + baseline_paired * (1.0 - masks)
        preds_masked = self._get_predictions(masked)
        
        # Get original predictions
        if self.paired_sampling:
            n_unique = total // 2
            preds_unique = self._get_predictions(X_paired[:n_unique])
            preds_orig = preds_unique.repeat(2)
        else:
            preds_orig = self._get_predictions(X_paired)
        
        # Ensure predictions are tensors
        if not isinstance(preds_masked, torch.Tensor):
            preds_masked = torch.tensor(preds_masked, dtype=torch.float32, 
                                        device=self.device)
        if not isinstance(preds_orig, torch.Tensor):
            preds_orig = torch.tensor(preds_orig, dtype=torch.float32, 
                                      device=self.device)
        
        # Mixed precision training
        use_amp = self.scaler is not None and self.device.type == 'cuda'
        
        with autocast(enabled=use_amp):
            # Forward pass
            phi = self.explainer(X_paired)
            
            # Coalition loss
            masked_sum = (masks * phi).sum(dim=1)
            coalition_loss = ((masked_sum - (preds_masked - self.base_pred)) ** 2).mean()
            
            # Efficiency loss
            phi_sum = phi.sum(dim=1)
            eff_loss = self.efficiency_lambda * (
                (phi_sum - (preds_orig - self.base_pred)) ** 2
            ).mean()
            
            # L1 regularization
            l1_loss = self.l1_lambda * torch.abs(phi).mean()
            
            # Total loss
            loss = coalition_loss + eff_loss + l1_loss
        
        # Check for invalid loss
        if not torch.isfinite(loss):
            return float('inf')
        
        # Backpropagation
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
    
    def _validate(self, X_val_flat):
        """
        Compute validation loss.
        
        Args:
            X_val_flat: Flattened validation data (n_samples, input_dim)
        
        Returns:
            float: Average validation loss
        """
        self.explainer.eval()
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_val_flat)),
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        total_loss, n_batches = 0.0, 0
        
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                
                # Predict SHAP values
                phi = self.explainer(X_batch)
                
                # Get model predictions
                preds = self._get_predictions(X_batch)
                
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
                
                # Efficiency error
                eff_err = ((phi.sum(dim=1) - (preds - self.base_pred)) ** 2).mean()
                
                if torch.isfinite(eff_err):
                    total_loss += eff_err.item()
                    n_batches += 1
        
        self.explainer.train()
        
        return total_loss / max(n_batches, 1) if n_batches > 0 else float('inf')
    
    def train(self, X_train, X_val, model_predict_func, feature_names, gpu_model=None):
        """
        Train the FastSHAP explainer.
        
        Args:
            X_train: Training data (n_samples, time_steps, n_features)
            X_val: Validation data (n_samples, time_steps, n_features)
            model_predict_func: Function to get predictions from black-box model
            feature_names: List of feature names
            gpu_model: Optional PyTorch model on GPU for direct inference
        
        Returns:
            float: Best validation loss achieved
        """
        # Setup
        try:
            self._setup(X_train, model_predict_func, feature_names, gpu_model=gpu_model)
        except Exception as e:
            if self.verbose:
                print(f"    [ERROR] Setup failed: {e}")
            return float('inf')
        
        # Flatten data
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_val_flat = X_val.reshape(len(X_val), -1)
        
        # DataLoader configuration
        use_cuda = self.device.type == 'cuda'
        num_workers = 4 if use_cuda else 0
        
        effective_batch_size = min(self.batch_size, len(X_train_flat) - 1)
        if effective_batch_size < 1:
            if self.verbose:
                print(f"    [ERROR] Not enough training samples")
            return float('inf')
        
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_flat)),
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True
        )
        
        # Setup optimizer
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
        
        # Training loop
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
            val_loss = self._validate(X_val_flat)
            
            if val_loss == float('inf'):
                continue
            
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)
            
            # Check for improvement
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                
                if hasattr(self.explainer, '_orig_mod'):
                    best_weights = {k: v.clone() 
                                   for k, v in self.explainer._orig_mod.state_dict().items()}
                else:
                    best_weights = {k: v.clone() 
                                   for k, v in self.explainer.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            # Print progress
            if self.verbose and (epoch + 1) % 5 == 0:
                print(f"    E{epoch + 1:3d} | L:{epoch_loss:.4f} V:{val_loss:.4f} LR:{lr:.6f}")
            
            # Early stopping
            if no_improve >= self.patience:
                if self.verbose:
                    print(f"    [STOP] epoch {epoch + 1}")
                break
        
        # Restore best weights
        if best_weights:
            if hasattr(self.explainer, '_orig_mod'):
                self.explainer._orig_mod.load_state_dict(best_weights)
            else:
                self.explainer.load_state_dict(best_weights)
        
        self.best_loss = best_val
        
        return best_val
    
    def explain(self, instance):
        """
        Generate SHAP values for a single instance.
        
        Args:
            instance: Input sample - can be:
                - np.ndarray (time_steps, n_features)
                - np.ndarray (1, time_steps, n_features)
                - np.ndarray flattened (input_dim,)
                - torch.Tensor with same shapes
        
        Returns:
            np.ndarray: SHAP values (time_steps, n_features)
        """
        if self.explainer is None:
            raise ValueError("Explainer not trained. Call train() first.")
        
        # Convert to tensor
        if isinstance(instance, np.ndarray):
            instance = torch.FloatTensor(instance)
        
        # Handle different shapes
        if instance.ndim == 3:
            instance = instance.reshape(instance.size(0), -1)
        elif instance.ndim == 2:
            # Could be (time_steps, n_features) or (1, input_dim)
            if instance.size(0) == self.time_steps and instance.size(1) == self.n_features:
                instance = instance.reshape(1, -1)
            elif instance.size(1) != self.input_dim:
                instance = instance.reshape(1, -1)
        elif instance.ndim == 1:
            instance = instance.unsqueeze(0)
        
        instance = instance.to(self.device)
        
        # Generate SHAP values
        self.explainer.eval()
        with torch.no_grad():
            phi_flat = self.explainer(instance).cpu().numpy()[0]
        
        # Reshape to original dimensions
        return phi_flat.reshape(self.time_steps, self.n_features)
    
    def save(self, path, filename="fastshap_explainer"):
        """
        Save trained explainer to disk.
        
        Args:
            path: Directory path
            filename: Filename without extension (default: 'fastshap_explainer')
        
        Returns:
            str: Full path to saved file
        """
        os.makedirs(path, exist_ok=True)
        
        # Get state dict (handle compiled model)
        if hasattr(self.explainer, '_orig_mod'):
            state_dict = self.explainer._orig_mod.state_dict()
        else:
            state_dict = self.explainer.state_dict()
        
        # Prepare base_pred for saving
        base_pred_save = self.base_pred
        if isinstance(base_pred_save, torch.Tensor):
            base_pred_save = base_pred_save.cpu()
        else:
            base_pred_save = torch.tensor(float(base_pred_save))
        
        # Create state
        state = {
            'explainer': state_dict,
            'baseline': self.baseline.cpu(),
            'base_pred': base_pred_save,
            'input_dim': self.input_dim,
            'time_steps': self.time_steps,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'best_loss': self.best_loss,
            'history': self.history,
            'init_params': self._init_params
        }
        
        save_path = os.path.join(path, f"{filename}.pt")
        torch.save(state, save_path)
        
        return save_path
    
    @classmethod
    def load(cls, path, filename="fastshap_explainer", device_override=None):
        """
        Load trained explainer from disk.
        
        Args:
            path: Directory path
            filename: Filename without extension (default: 'fastshap_explainer')
            device_override: Optional device override
        
        Returns:
            FastSHAPExplainer: Loaded explainer instance
        """
        dev = device_override or device
        load_path = os.path.join(path, f"{filename}.pt")
        state = torch.load(load_path, map_location=dev, weights_only=False)
        
        # Reconstruct explainer
        params = state.get('init_params', {})
        exp = cls(**params)
        
        # Restore attributes
        exp.device = dev
        exp.input_dim = state['input_dim']
        exp.time_steps = state['time_steps']
        exp.n_features = state['n_features']
        exp.feature_names = state['feature_names']
        exp.baseline = state['baseline'].to(dev)
        
        # Handle base_pred
        base_pred = state['base_pred']
        if isinstance(base_pred, torch.Tensor):
            exp.base_pred = base_pred.to(dev)
        else:
            exp.base_pred = torch.tensor(float(base_pred), device=dev)
        
        exp.best_loss = state.get('best_loss', float('inf'))
        exp.history = state.get('history', {})
        
        # Initialize GPU flags
        exp._gpu_model = None
        exp._model_on_gpu = False
        
        # Restore network
        exp.explainer = FastSHAPNetwork(
            exp.input_dim, 
            params.get('hidden_dim', 256),
            params.get('n_layers', 2), 
            params.get('dropout_rate', 0.2)
        ).to(dev)
        
        exp.explainer.load_state_dict(state['explainer'])
        exp.explainer.eval()
        
        # Recompute Shapley kernel
        _, exp._shapley_probs_elements = exp._compute_shapley_kernel(exp.input_dim)
        
        return exp
