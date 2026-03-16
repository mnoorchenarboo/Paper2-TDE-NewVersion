# Standard Library Imports
import os
import time
import warnings
import json
import sqlite3
from io import StringIO
from typing import Callable, Union, Tuple

# Third-party Imports
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import comb
import shap
import optuna
from optuna.samplers import TPESampler
from lime.lime_tabular import LimeTabularExplainer

# Machine Learning & Metrics
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import DataConversionWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.cluster import KMeans, MiniBatchKMeans

# TensorFlow/Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout, Input, LayerNormalization,
    MultiHeadAttention, Flatten, Concatenate, TimeDistributed, Add,
    BatchNormalization, Bidirectional, SpatialDropout1D, Activation
)
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau as KerasReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# PyTorch Imports
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLROnPlateau

# Local Imports
from Functions import BDG2, preprocess

from tensorflow.keras.callbacks import Callback
import numpy as np
import joblib
import os
import re
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*The default value of non_blocking.*")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import itables.options as opt
from itables import init_notebook_mode, show
from itables.downsample import as_nbytes, nbytes
from itables.sample_dfs import get_indicators

init_notebook_mode(all_interactive=True)
# Configure global options
opt.maxBytes = "0"  # Display all rows without downsampling
opt.buttons = ["copyHtml5", "csvHtml5", "excelHtml5"]  # Add export buttons

def myshow(df):
    # Display the dataframe with SearchBuilder
    show(df, layout={"top1": "searchBuilder"}, searchBuilder={"preDefined": {}})


class TemporalDeepExplainer:
    def __init__(self,
                 n_epochs=200,
                 batch_size=512,
                 patience=10,
                 delta=1e-4,
                 verbose=True,
                 min_lr=1e-6,
                 # Regularization parameters
                 l1_lambda=0.1,
                 weight_decay=1e-4,
                 activation_shrink=0.1,
                 smoothness_lambda=0.1,  # Temporal smoothness
                 efficiency_lambda=0.05,  # Value magnitude control
                 # Sampling parameters (new)
                 regularization_mode='element',  # 'element' or 'feature'
                 paired_sampling=False,  # Paired mask generation
                 samples_per_feature=1,  # Samples per feature
                 # Architecture parameters (new)
                 num_attention_heads=4,  # Number of attention heads in multi-head attention
                 num_conv_layers=2,  # Number of convolutional layers
                 num_filters=64,  # Number of filters in each convolutional layer
                 kernel_size=3,  # Kernel size for convolutions
                 dropout_rate=0.2,  # Dropout rate for regularization
                 optimizer_type="adam",  # Optimizer type (adam, adamw, sgd, rmsprop)
                 learning_rate=1e-3,  # Learning rate for training
                 activation_fn='relu'  # New parameter: 'gelu' (default) or 'relu'
                 ):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.min_lr = min_lr

        # Regularization parameters
        self.l1_lambda = l1_lambda
        self.weight_decay = weight_decay
        self.activation_shrink = activation_shrink
        self.smoothness_lambda = smoothness_lambda
        self.efficiency_lambda = efficiency_lambda

        # Sampling parameters
        self.paired_sampling = paired_sampling
        self.samples_per_feature = samples_per_feature

        # Architecture parameters
        self.num_attention_heads = num_attention_heads
        self.num_conv_layers = num_conv_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        # Optimizer settings
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate

        self.regularization_mode = regularization_mode
        if self.regularization_mode not in ['element', 'feature']:
            raise ValueError("regularization_mode must be 'element' or 'feature'")

        # Activation function option
        self.activation_fn = activation_fn  # 'gelu' or 'relu'

        # Track all initialization parameters for saving/loading
        self._init_params = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "patience": patience,
            "delta": delta,
            "verbose": verbose,
            "min_lr": min_lr,
            "l1_lambda": l1_lambda,
            "weight_decay": weight_decay,
            "activation_shrink": activation_shrink,
            "smoothness_lambda": smoothness_lambda,
            "efficiency_lambda": efficiency_lambda,
            "regularization_mode": regularization_mode,
            "paired_sampling": paired_sampling,
            "samples_per_feature": samples_per_feature,
            "num_attention_heads": num_attention_heads,
            "num_conv_layers": num_conv_layers,
            "num_filters": num_filters,
            "kernel_size": kernel_size,
            "dropout_rate": dropout_rate,
            "optimizer_type": optimizer_type,
            "learning_rate": learning_rate,
            "activation_fn": activation_fn
        }

        self.explainer = None
        self.baseline = None
        self.feature_names = None
        self.time_steps = None
        self.n_features = None
        self.model_predict_func = None
        self.base_pred = None
        self.best_loss = float('inf')

    def initialize(self, X_train, model_predict_func, feature_names):
        """Initialize the explainer with training data and model"""
        self._validate_input(X_train)
        self._setup_core_components(X_train, model_predict_func, feature_names)
        self._train_fastshap(X_train)
        return self

    def explain(self, instance):
        """Generate SHAP values for an input instance"""
        instance = self._preprocess_input(instance)  # Ensure correct format
        if self.explainer is None:
            raise ValueError("Explainer model is not initialized. Call `initialize()` first.")

        with torch.no_grad():
            shap_values = self.explainer(instance).cpu().numpy()

        return self._create_shap_dataframe(shap_values[0])

    class TemporalExplainer(nn.Module):
        """Enhanced neural network architecture for temporal SHAP value estimation"""

        def __init__(self, time_steps, n_features, num_attention_heads=4, num_conv_layers=2,
                     num_filters=64, kernel_size=3, dropout_rate=0.2, activation_shrink=0.1, activation_fn='gelu'):
            super().__init__()
            self.num_conv_layers = num_conv_layers
            self.dropout = nn.Dropout(dropout_rate)

            # Dynamic convolution layers
            conv_layers = []
            in_channels = n_features
            for _ in range(num_conv_layers):
                conv_layers.append(nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2))
                # Select activation based on activation_fn
                if activation_fn.lower() == 'relu':
                    conv_layers.append(nn.ReLU())
                elif activation_fn.lower() == 'gelu':
                    conv_layers.append(nn.GELU())
                else:
                    raise ValueError("Unsupported activation function. Use 'relu' or 'gelu'.")
                conv_layers.append(nn.LayerNorm([num_filters, time_steps]))
                conv_layers.append(nn.Dropout(dropout_rate))
                in_channels = num_filters  # Set new in_channels for the next layer

            self.time_conv = nn.Sequential(*conv_layers)

            # Multi-Head Attention
            self.attention = nn.MultiheadAttention(num_filters, num_attention_heads, batch_first=True)

            # Feature-wise convolution with sparsity
            self.feature_conv = nn.Sequential(
                nn.Conv1d(num_filters, n_features, kernel_size, padding=kernel_size // 2),
                nn.Softshrink(activation_shrink)  # Sparsity-inducing activation
            )

        def forward(self, x):
            x = x.permute(0, 2, 1)  # Change shape to (batch, features, time)
            temporal_features = self.time_conv(x)  # Pass through convolutional layers
            attn_out, _ = self.attention(
                temporal_features.permute(0, 2, 1),  # (batch, time, features)
                temporal_features.permute(0, 2, 1),
                temporal_features.permute(0, 2, 1)
            )
            combined = temporal_features + attn_out.permute(0, 2, 1)  # Add residual connection
            return self.feature_conv(combined).permute(0, 2, 1)  # Convert back to (batch, time, features)

    def _train_fastshap(self, X_train):
        """Core training procedure with hyperparameter-optimized architecture and optimizer selection"""
        # Choose d based on the mask type: feature-level vs. element-level
        if self.regularization_mode == 'feature':
            d = self.n_features
        else:
            d = self.time_steps * self.n_features

        weights, probs = self._compute_shapley_kernel(d)
        if d <= 1:
            raise ValueError("Feature dimension too small for SHAP computation")

        # Precompute Shapley kernel weights (weights & probs are computed above)
        # Note: In the loss, these kernel weights are currently not applied further.
        # They could be integrated into the loss if needed for improved consistency.

        # Initialize training components
        loader = self._create_dataloader(X_train)

        # Dynamically select optimizer
        optimizer_dict = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adagrad": torch.optim.Adagrad,  # NEW: Adaptive Gradient Descent
            "adadelta": torch.optim.Adadelta,  # NEW: Variant of AdaGrad
            "nadam": torch.optim.NAdam  # NEW: Adam with Nesterov momentum
        }
        if self.optimizer_type not in optimizer_dict:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        optimizer = optimizer_dict[self.optimizer_type](self.explainer.parameters(),
                                                        lr=self.learning_rate,
                                                        weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.patience // 2,
                                      min_lr=self.min_lr, verbose=self.verbose)

        # Early stopping initialization
        best_weights = None
        no_improve = 0
        smoothed_loss = None
        alpha = 0.1  # Smoothing factor for Exponential Moving Average (EMA)
        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            self.explainer.train()

            # Batch processing
            for X_batch in loader:
                X_batch = X_batch[0].to(self.device)  # Ensure batch is on the correct device
                loss = self._process_batch(X_batch, d, weights, probs, optimizer)
                epoch_loss += loss.item()

            # Calculate metrics
            current_loss = epoch_loss / len(loader)
            smoothed_loss = self._update_smoothed_loss(smoothed_loss, current_loss, alpha)
            lr = optimizer.param_groups[0]['lr']

            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(smoothed_loss)
            else:
                scheduler.step()

            # Early stopping logic
            improvement, best_weights, no_improve = self._check_early_stop(
                smoothed_loss, best_loss, best_weights,
                no_improve, epoch, lr
            )

            # Print training progress
            if self.verbose:
                print(f"Epoch {epoch + 1:03d} | Loss: {current_loss:.4f} "
                      f"(Smoothed: {smoothed_loss:.4f}) | LR: {lr:.2e}")

            # Break condition
            if improvement == 'stop':
                if best_weights:
                    self.explainer.load_state_dict(best_weights)
                break

            # Update best loss
            if improvement:
                best_loss = smoothed_loss

        return self

    def _update_smoothed_loss(self, current_smooth, new_loss, alpha):
        """Update the exponential moving average (EMA) of loss to stabilize training loss tracking."""
        if current_smooth is None:
            return new_loss  # Initialize EMA with the first loss value
        return alpha * new_loss + (1 - alpha) * current_smooth

    # def _check_early_stop(self, smoothed_loss, best_loss, best_weights,
    #                       no_improve, epoch, lr):
    #     """Evaluate early stopping conditions with adaptive thresholds."""
    #     improvement = False
    #     # Adaptive improvement threshold (increase threshold over epochs)
    #     progress = epoch / self.n_epochs
    #     adaptive_delta = self.delta * (1 + progress)
    #
    #     # Check for improvement in smoothed loss
    #     if smoothed_loss < (best_loss - adaptive_delta):
    #         best_loss = smoothed_loss
    #         best_weights = self.explainer.state_dict().copy()
    #         no_improve = 0  # Reset counter since improvement was found
    #         improvement = True
    #     else:
    #         no_improve += 1  # Increase counter when no improvement is observed
    #
    #     # Define stopping conditions
    #     stop_conditions = [
    #         no_improve >= self.patience,  # No improvement for patience epochs
    #         epoch >= self.n_epochs // 2,  # Minimum 50% of total epochs must be completed
    #         lr <= self.min_lr * 10,  # Learning rate is too small to continue useful updates
    #         abs(best_loss - smoothed_loss) < self.delta  # Change in loss is below the threshold
    #     ]
    #
    #     # Trigger early stopping if all conditions are met
    #     if all(stop_conditions):
    #         if self.verbose:
    #             print(f"\n[EARLY STOPPING] Triggered at Epoch {epoch + 1}")
    #             print(f" - Best Smoothed Loss: {best_loss:.6f}")
    #             print(f" - Current Smoothed Loss: {smoothed_loss:.6f}")
    #             print(f" - No Improvement for {no_improve} Epochs")
    #             print(f" - Learning Rate: {lr:.2e}")
    #         return 'stop', best_weights, no_improve
    #
    #     return improvement, best_weights, no_improve

    def _check_early_stop(self, smoothed_loss, best_loss, best_weights, no_improve, epoch, lr):
        """Evaluate early stopping conditions with adaptive thresholds (without absolute loss-change check)."""
        improvement = False
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
            best_weights = self.explainer.state_dict().copy()
            no_improve = 0
            improvement = True
        else:
            no_improve += 1

        stop_conditions = [
            no_improve >= self.patience,  # No improvement for patience epochs
            epoch >= self.n_epochs // 2,  # At least half of the epochs have passed
            lr <= self.min_lr * 10,  # Learning rate is low
        ]

        if all(stop_conditions):
            if self.verbose:
                print(f"\n[EARLY STOPPING] Triggered at Epoch {epoch + 1}")
                print(f" - Best Smoothed Loss: {best_loss:.6f}")
                print(f" - Current Smoothed Loss: {smoothed_loss:.6f}")
                print(f" - No Improvement for {no_improve} Epochs")
                print(f" - Learning Rate: {lr:.2e}")
            return 'stop', best_weights, no_improve

        return improvement, best_weights, no_improve

    def _create_dataloader(self, X_train):
        """Create a configured DataLoader for training with optimized settings."""
        # Convert to Tensor and move to device
        dataset = TensorDataset(torch.FloatTensor(X_train))
        # Determine number of workers dynamically for efficiency
        num_workers = min(4, max(1, torch.get_num_threads() // 2))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            prefetch_factor=2 if num_workers > 0 else None
        )

    def _compute_shapley_kernel(self, d):
        """Calculate Shapley kernel weights for coalition sampling with improved numerical stability."""
        if d <= 1:
            raise ValueError("Feature dimension must be greater than 1 for Shapley computation.")
        k_values = torch.arange(1, d, device=self.device, dtype=torch.float32)
        # Compute binomial coefficients safely
        binom_coeffs = torch.tensor([comb(d, int(k.item()), exact=True) for k in k_values],
                                    device=self.device, dtype=torch.float32)
        weights = (d - 1) / (k_values * (d - k_values) * binom_coeffs)
        probs = weights / weights.sum()
        return weights, probs

    def _process_batch(self, X_batch, d, weights, probs, optimizer):
        """Process a batch of data for SHAP value computation with regularization and gradient clipping."""
        batch_size = X_batch.size(0)
        X_batch = X_batch.to(self.device)

        # Generate expanded batch with samples_per_feature
        expanded_X = X_batch.repeat(self.samples_per_feature, 1, 1)  # Shape: (B*K, T, F)

        # Generate coalition masks
        masks = self._generate_shapley_masks(batch_size, d, probs).to(self.device)

        # Prepare inputs
        total_samples = masks.size(0)
        X_paired = expanded_X.repeat(total_samples // (batch_size * self.samples_per_feature), 1, 1)
        baseline_paired = self.baseline.repeat(total_samples, 1, 1)

        # Apply masking
        masked_inputs = X_paired * masks + baseline_paired * (1 - masks)

        # Get predictions from the model
        preds = self._get_model_predictions(masked_inputs).to(self.device)

        # Compute SHAP values from the explainer
        phi = self.explainer(X_paired)  # Shape: (total_samples, T, F)

        # Mean Squared Error loss for SHAP consistency
        mse_loss = ((preds - self.base_pred - (masks * phi).sum((1, 2))) ** 2).mean()

        # Temporal smoothness regularization (only if time dimension > 1)
        if phi.size(1) > 1:
            smooth_loss = self.smoothness_lambda * (phi[:, 1:, :] - phi[:, :-1, :]).pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=self.device)

        # Regularization calculations
        if self.regularization_mode == 'feature':
            # Sum over time steps for each feature
            phi_sum = phi.sum(dim=1)
            l1_reg = self.l1_lambda * torch.abs(phi_sum).mean()
            eff_loss = self.efficiency_lambda * torch.pow(phi_sum, 2).mean()
        else:  # 'element'
            l1_reg = self.l1_lambda * torch.abs(phi).mean()
            eff_loss = self.efficiency_lambda * torch.pow(phi, 2).mean()

        # Compute final loss
        loss = mse_loss + l1_reg + smooth_loss + eff_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.explainer.parameters(), max_norm=1.0)
        optimizer.step()

        return loss

    def _generate_shapley_masks(self, batch_size, d, probs):
        """
        Generate coalition masks for Shapley sampling.

        If regularization_mode is 'feature', generate masks at the feature level
        (i.e. the same mask is applied across all time steps). Otherwise, generate
        element-wise masks (for every time-feature pair).
        """
        if self.regularization_mode == 'feature':
            # Here, d should equal the number of features.
            d_features = self.n_features
            # Compute kernel weights for features only.
            _, probs_feature = self._compute_shapley_kernel(d_features)
            # Sample coalition sizes for each sample.
            k_indices = torch.multinomial(probs_feature, batch_size * self.samples_per_feature, replacement=True)
            k_samples = torch.arange(1, d_features, device=self.device, dtype=torch.int64)[k_indices]
            # Generate random values per feature.
            rand = torch.rand(batch_size * self.samples_per_feature, d_features, device=self.device)
            sorted_indices = torch.argsort(rand, dim=1)
            # Create masks: each row has k_features ones (selected features).
            masks = (sorted_indices < k_samples.unsqueeze(1)).float()
            # Replicate the feature mask along the time axis.
            masks = masks.unsqueeze(1).repeat(1, self.time_steps, 1)
            if self.paired_sampling:
                paired_masks = 1 - masks
                masks = torch.cat([masks, paired_masks], dim=0)
            return masks
        else:
            # Element-level masking (existing approach)
            k_indices = torch.multinomial(probs, batch_size * self.samples_per_feature, replacement=True)
            k_samples = torch.arange(1, d, device=self.device, dtype=torch.int64)[k_indices]
            rand = torch.rand(batch_size * self.samples_per_feature, d, device=self.device)
            sorted_indices = torch.argsort(rand, dim=1)
            masks = (sorted_indices < k_samples.unsqueeze(1)).float()
            masks = masks.view(-1, self.time_steps, self.n_features)
            if self.paired_sampling:
                paired_masks = 1 - masks
                masks = torch.cat([masks, paired_masks], dim=0)
            return masks

    def _get_model_predictions(self, inputs):
        """Get predictions from the black-box model with improved device handling."""
        with torch.no_grad():
            masked_np = inputs.cpu().numpy()
            preds = torch.tensor(self.model_predict_func(masked_np), dtype=torch.float32, device=self.device)
        return preds.flatten()

    def _check_improvement(self, current_loss, best_weights, no_improve):
        """Check for loss improvement and update best weights with better handling."""
        if current_loss < (self.best_loss - self.delta):
            self.best_loss = current_loss
            best_weights = {k: v.clone() for k, v in self.explainer.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        return no_improve, best_weights

    def _update_scheduler(self, scheduler, current_loss):
        """Update learning rate scheduler with improved robustness."""
        if scheduler is None:
            return
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(current_loss)
        else:
            try:
                scheduler.step()
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Scheduler step failed: {e}")

    def _validate_input(self, X_train):
        """Validate input dimensions and ensure it is compatible with time-series SHAP computation."""
        if not isinstance(X_train, np.ndarray) and not isinstance(X_train, torch.Tensor):
            raise TypeError("Input must be a NumPy array or PyTorch tensor.")
        if X_train.ndim != 3:
            raise ValueError("Input must be 3D: (samples, time_steps, features)")
        if X_train.shape[0] < 10:
            raise ValueError("Need at least 10 samples for stable training")

    def _setup_core_components(self, X_train, model_predict_func, feature_names):
        """Initialize core components and baseline values with updated architecture hyperparameters."""
        self.time_steps = X_train.shape[1]
        self.n_features = X_train.shape[2]
        self.feature_names = feature_names
        self.model_predict_func = model_predict_func

        # Convert X_train to a tensor and move to the correct device.
        X_train = torch.FloatTensor(X_train).to(self.device)

        # Initialize baseline using the median of training data.
        self.baseline = torch.median(X_train, dim=0)[0].to(self.device)
        baseline_np = self.baseline.unsqueeze(0).cpu().numpy()
        self.base_pred = torch.tensor(model_predict_func(baseline_np), dtype=torch.float32, device=self.device)

        # Initialize the explainer model.
        self.explainer = self.TemporalExplainer(
            time_steps=self.time_steps,
            n_features=self.n_features,
            num_attention_heads=self.num_attention_heads,
            num_conv_layers=self.num_conv_layers,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            activation_shrink=self.activation_shrink
        ).to(self.device)

    def _preprocess_input(self, instance):
        """Preprocess input instance for explanation, ensuring correct dimensions and device compatibility."""
        if not isinstance(instance, (np.ndarray, torch.Tensor)):
            raise TypeError("Input instance must be a NumPy array or a PyTorch tensor.")
        if instance.ndim == 2:
            instance = np.expand_dims(instance, axis=0)  # Add batch dimension
        return torch.tensor(instance, dtype=torch.float32, device=self.device)

    def _create_shap_dataframe(self, shap_values):
        """Create a formatted DataFrame for SHAP values with proper indexing and error handling."""
        if not isinstance(shap_values, np.ndarray):
            raise TypeError("SHAP values must be a NumPy array.")
        if shap_values.shape[0] != self.time_steps:
            raise ValueError(f"SHAP values must have {self.time_steps} time steps, but got {shap_values.shape[0]}.")
        return pd.DataFrame(
            shap_values,
            columns=self.feature_names,
            index=[f"t-{self.time_steps - t - 1}" for t in range(self.time_steps)]
        )

    def save(self, path, filename="explainer"):
        """Save model state with custom filename and all initial parameters."""
        import os
        os.makedirs(path, exist_ok=True)
        self._init_params = {
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "delta": self.delta,
            "verbose": self.verbose,
            "min_lr": self.min_lr,
            "l1_lambda": self.l1_lambda,
            "weight_decay": self.weight_decay,
            "activation_shrink": self.activation_shrink,
            "smoothness_lambda": self.smoothness_lambda,
            "efficiency_lambda": self.efficiency_lambda,
            "regularization_mode": self.regularization_mode,
            "paired_sampling": self.paired_sampling,
            "samples_per_feature": self.samples_per_feature,
            "num_attention_heads": self.num_attention_heads,
            "num_conv_layers": self.num_conv_layers,
            "num_filters": self.num_filters,
            "kernel_size": self.kernel_size,
            "dropout_rate": self.dropout_rate,
            "optimizer_type": self.optimizer_type,
            "learning_rate": self.learning_rate
        }
        try:
            state = {
                "explainer": self.explainer.state_dict(),
                "baseline": self.baseline.cpu(),
                "base_pred": self.base_pred.cpu(),
                "time_steps": self.time_steps,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
                "best_loss": self.best_loss,
                "init_params": self._init_params
            }
            torch.save(state, os.path.join(path, f"{filename}.pt"))
            if self.verbose:
                print(f"Model saved successfully at: {os.path.join(path, f'{filename}.pt')}")
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, path, filename="explainer", device=None):
        """Robust model loading with parameter fallbacks."""
        import os
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            state = torch.load(os.path.join(path, f"{filename}.pt"), map_location=device)
            init_params = state.get("init_params", {})
            if not isinstance(init_params, dict):
                init_params = {}
            explainer = cls(**init_params)
            explainer.time_steps = state["time_steps"]
            explainer.n_features = state["n_features"]
            explainer.feature_names = state["feature_names"]
            explainer.baseline = state["baseline"].to(device)
            explainer.best_loss = state.get("best_loss", float("inf"))
            explainer.base_pred = state["base_pred"].to(device)
            explainer.explainer = explainer.TemporalExplainer(
                time_steps=explainer.time_steps,
                n_features=explainer.n_features,
                num_attention_heads=init_params.get("num_attention_heads", 4),
                num_conv_layers=init_params.get("num_conv_layers", 2),
                num_filters=init_params.get("num_filters", 64),
                kernel_size=init_params.get("kernel_size", 3),
                dropout_rate=init_params.get("dropout_rate", 0.2),
                activation_shrink=init_params.get("activation_shrink", 0.1)
            ).to(device)
            explainer.explainer.load_state_dict(state["explainer"])
            if explainer.verbose:
                print(f"Model successfully loaded from {os.path.join(path, f'{filename}.pt')} on {device}")
            return explainer
        except Exception as e:
            print(f"Error loading model: {e}")
            return None


def load_and_plot_models(mydata, model_type):

    # Unpack the data
    X_train = mydata.X_train 
    y_train = mydata.y_train 
    X_val = mydata.X_val 
    y_val = mydata.y_val 
    X_test = mydata.X_test 
    y_test = mydata.y_test 

    df_name = mydata.data_type
    # Directories for loading models
    model_dir = f"./Results/Models/{df_name}"

    # Load the deep learning model
    dl_model_path = f"{model_dir}/{model_type}.keras"
    if not os.path.exists(dl_model_path):
        raise FileNotFoundError(f"Deep Learning model file not found at {dl_model_path}")
    dl_model = load_model(dl_model_path)

    # Load the corresponding random forest model
    rf_model_path = f"{model_dir}/{model_type}_RandomForest.pkl"
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Random Forest model file not found at {rf_model_path}")
    rf_model = joblib.load(rf_model_path)

    # Predictions
    y_test_dl_pred = dl_model.predict(X_test)[:, 0].flatten()
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    y_test_rf_pred = rf_model.predict(X_test_reshaped).flatten()
    y_test_actual = y_test[:, 0].flatten()

    # Calculate metrics for each model
    def calculate_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mae, mape, r2

    dl_metrics = calculate_metrics(y_test_actual, y_test_dl_pred)
    rf_metrics = calculate_metrics(y_test_dl_pred, y_test_rf_pred)

    # Plot the results
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot actual, DL predictions, and RF predictions
    plt.plot(
        y_test_actual[:len(y_test_actual)//10], 
        label='True', color='#ed8787', alpha=0.99
    )
    plt.plot(
        y_test_dl_pred[:len(y_test_dl_pred)//10], 
        label=f'Predicted ({model_type})', color='#1f77b4', alpha=0.99
    )
    plt.plot(
        y_test_rf_pred[:len(y_test_rf_pred)//10], 
        label=f'Predicted ({model_type} RF)', color='#2ca02c', alpha=0.99
    )

    # Set title and labels
    plt.title(f'Dataset={mydata.data_type}, Model={model_type}, MSE(RF)={rf_metrics[0]:.2f}, MAE(RF)={rf_metrics[2]:.2f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')

    # Add border lines (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Set legend
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_facecolor('lightgray')

    # Save and show the plot
    plot_dir = f"./Results/Plots/{df_name}"
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = f"{plot_dir}/{model_type}_ComparisonPlot.pdf"
    plt.savefig(plot_path, format='pdf', facecolor='white', bbox_inches='tight')
    plt.show()

    # Explicitly close the plot
    plt.close()

    print(f"Plot saved at {plot_path}")

class MAPECallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_val_pred = self.model.predict(X_val)
        mape_score = mean_absolute_percentage_error(y_val[:, 0].flatten(), y_val_pred[:, 0].flatten()) * 100
        print(f"Epoch {epoch + 1} - MAPE: {mape_score:.2f}%")

def create_lstm_model(trial, input_shape, output_shape):
    model = Sequential()

    lstm_units = trial.suggest_int('lstm_units', 64, 128)  # Narrowed and consistent with successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(lstm_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_gru_model(trial, input_shape, output_shape):
    model = Sequential()

    gru_units = trial.suggest_int('gru_units', 64, 128)  # Consistent with other successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_gru_layers):
        return_sequences = (i < num_gru_layers - 1)
        model.add(GRU(gru_units, return_sequences=return_sequences))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(gru_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_blstm_model(trial, input_shape, output_shape):
    model = Sequential()

    lstm_units = trial.suggest_int('lstm_units', 64, 128)  # Narrowed and consistent with successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)
        model.add(Bidirectional(LSTM(lstm_units, return_sequences=return_sequences)))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(lstm_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_bgru_model(trial, input_shape, output_shape):
    model = Sequential()

    gru_units = trial.suggest_int('gru_units', 64, 128)  # Consistent with other successful models
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)  # Fixed to the range used by successful models
    num_gru_layers = trial.suggest_int('num_gru_layers', 1, 2)  # Keep it simple

    model.add(Input(shape=input_shape))

    for i in range(num_gru_layers):
        return_sequences = (i < num_gru_layers - 1)
        model.add(Bidirectional(GRU(gru_units, return_sequences=return_sequences)))
        model.add(LayerNormalization())  # Consistency with successful architectures
        model.add(Dropout(dropout_rate))

    # Add a Dense layer as in CNN/TFT/TST
    model.add(Dense(gru_units // 2, activation='relu'))

    model.add(Dense(output_shape, activation='linear'))

    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3)  # Narrowed range
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_cnn_model(trial, input_shape, output_shape):
    model = Sequential()
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)

    model.add(Input(shape=input_shape))
    for i in range(num_conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg)))
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def create_tcn_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_tcn_blocks = trial.suggest_int('num_tcn_blocks', 2, 5)
    dilation_base = trial.suggest_int('dilation_base', 2, 4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Define the TCN block
    def tcn_block(x, filters, kernel_size, dilation_rate, l2_reg):
        x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SpatialDropout1D(dropout_rate)(x)
        return x

    inputs = Input(shape=input_shape)
    x = inputs

    for i in range(num_tcn_blocks):
        dilation_rate = dilation_base ** i
        prev_x = x  # Save the previous layer's output
        x = tcn_block(x, filters, kernel_size, dilation_rate, l2_reg)

        # Adjust the shape of prev_x to match x using a 1x1 convolution
        if prev_x.shape[-1] != x.shape[-1]:
            prev_x = Conv1D(filters, kernel_size=1, padding='same')(prev_x)

        # Skip connection (Residual)
        x = Add()([x, prev_x])

    x = Flatten()(x)
    x = Dense(output_shape, activation='linear')(x)

    model = Model(inputs, x)

    # Compile the model
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def create_dilated_cnn_model(trial, input_shape, output_shape):
    model = Sequential()
    
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dilation_rate = trial.suggest_int('dilation_rate', 1, 4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_conv_layers = trial.suggest_int('num_conv_layers', 1, 3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    
    model.add(Input(shape=input_shape))
    
    for i in range(num_conv_layers):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, 
                         dilation_rate=dilation_rate, activation='relu', 
                         kernel_regularizer=l2(l2_reg)))
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    model.add(Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg)))
    
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def create_wavenet_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    filters = trial.suggest_int('filters', 32, 128)
    kernel_size = trial.suggest_int('kernel_size', 2, 5)
    dilation_rate = trial.suggest_int('dilation_rate', 1, 4)
    num_wavenet_blocks = trial.suggest_int('num_wavenet_blocks', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
    
    # Input layer
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Define a single WaveNet block
    def wavenet_block(x, filters, kernel_size, dilation_rate, l2_reg):
        res_x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding='causal',
                       kernel_regularizer=l2(l2_reg))(x)
        res_x = Activation('relu')(res_x)
        res_x = Dropout(dropout_rate)(res_x)
        
        # Adjust the shape of x to match res_x
        if x.shape[-1] != filters:
            x = Conv1D(filters, 1, padding='same', kernel_regularizer=l2(l2_reg))(x)
        
        res_x = Conv1D(filters, 1, padding='same', kernel_regularizer=l2(l2_reg))(res_x)
        return Add()([x, res_x])  # Residual connection
    
    # Add multiple WaveNet blocks
    for i in range(num_wavenet_blocks):
        x = wavenet_block(x, filters, kernel_size, dilation_rate, l2_reg)
    
    # Flatten and add final Dense layer to match output shape
    x = Flatten()(x)
    outputs = Dense(output_shape, activation='linear', kernel_regularizer=l2(l2_reg))(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001) if optimizer_name == 'adam' else RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def create_tft_model(trial, input_shape, output_shape):

    # Hyperparameter suggestions
    hidden_units = trial.suggest_int('hidden_units', 32, 256)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
    num_lstm_layers = trial.suggest_int('num_lstm_layers', 1, 4)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_attention_blocks = trial.suggest_int('num_attention_blocks', 1, 4)
    l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Input layer
    inputs = Input(shape=input_shape)

    # LSTM layers with Layer Normalization and Residual Connections
    lstm_output = inputs
    for i in range(num_lstm_layers):
        lstm_output_residual = lstm_output
        lstm_output = LSTM(hidden_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(lstm_output)
        lstm_output = LayerNormalization()(lstm_output)  # Layer Normalization after LSTM
        
        # Adjust the dimension of residual to match lstm_output
        if lstm_output_residual.shape[-1] != lstm_output.shape[-1]:
            lstm_output_residual = Dense(hidden_units)(lstm_output_residual)
        
        lstm_output = Add()([lstm_output, lstm_output_residual])  # Residual Connection
        lstm_output = Dropout(dropout_rate)(lstm_output)

    # Attention mechanism (Temporal fusion)
    attention_heads = []
    for _ in range(num_attention_blocks):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units, dropout=dropout_rate)(lstm_output, lstm_output)
        attention_output = LayerNormalization()(attention_output)
        attention_heads.append(attention_output)

    if len(attention_heads) > 1:
        lstm_output = Concatenate()(attention_heads)
    else:
        lstm_output = attention_heads[0]

    # Final dense layers to match the output shape
    lstm_output = TimeDistributed(Dense(hidden_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))(lstm_output)
    lstm_output = Dropout(dropout_rate)(lstm_output)

    # Automatically adapt to the output sequence length
    lstm_output = lstm_output[:, -output_shape:, :]
    lstm_output = TimeDistributed(Dense(1, activation='linear'))(lstm_output)

    # Model creation
    model = Model(inputs, lstm_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def positional_encoding(sequence_length, d_model):
    angle_rads = get_angles(
        tf.range(sequence_length)[:, tf.newaxis],
        tf.range(d_model)[tf.newaxis, :],
        d_model
    )

    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)

    return tf.cast(pos_encoding[tf.newaxis, ...], tf.float32)

def get_angles(pos, i, d_model):
    pos = tf.cast(pos, tf.float32)  # Cast pos to float32
    i = tf.cast(i, tf.float32)      # Cast i to float32
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / d_model)
    return pos * angle_rates

def create_tst_model(trial, input_shape, output_shape):
    # Hyperparameter suggestions
    d_model = trial.suggest_int('d_model', 32, 128)
    num_heads = trial.suggest_int('num_heads', 2, 8)
    num_transformer_layers = trial.suggest_int('num_transformer_layers', 2, 6)
    ff_dim = trial.suggest_int('ff_dim', 32, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])

    # Input layer
    inputs = Input(shape=input_shape)

    # Linear projection to match d_model
    x = Dense(d_model)(inputs)

    # Positional encoding
    pos_encoding = positional_encoding(input_shape[0], d_model)
    x += pos_encoding[:, :input_shape[0], :]

    # Transformer layers
    for _ in range(num_transformer_layers):
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        attn_output = Add()([x, attn_output])  # Residual connection
        attn_output = LayerNormalization(epsilon=1e-6)(attn_output)

        # Feed-forward network
        ffn_output = Dense(ff_dim, activation='relu')(attn_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = Add()([attn_output, ffn_output])  # Residual connection
        x = LayerNormalization(epsilon=1e-6)(x)

    # Reduce the sequence length according to output_shape
    x = TimeDistributed(Dense(1, activation='linear'))(x)
    x = x[:, -output_shape:, :]  # Select the last 'output_shape' time steps

    # Model creation
    model = Model(inputs, x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model

def optimize_and_save_model(data, df_name, n_trials=50, epochs=100, verbosity=1, model_type='LSTM', more_info='More information about dataset'):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Create the subfolder in Models and Plots directories if it doesn't exist
    result_dir = "./Results"
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f"./Results/Models/{df_name}"
    plot_dir = f"./Results/Plots/{df_name}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Choose the model type
    if model_type == 'LSTM':
        model_creator = create_lstm_model
    elif model_type == 'GRU':
        model_creator = create_gru_model
    elif model_type == 'BLSTM':
        model_creator = create_blstm_model   
    elif model_type == 'BGRU':
        model_creator = create_bgru_model   
    elif model_type == 'CNN':
        model_creator = create_cnn_model
    elif model_type == 'TCN':
        model_creator = create_tcn_model
    elif model_type == 'DCNN':
        model_creator = create_dilated_cnn_model
    elif model_type == 'WaveNet':
        model_creator = create_wavenet_model
    elif model_type == 'TFT':
        model_creator = create_tft_model
    elif model_type == 'TST':
        model_creator = create_tst_model 
    else:
        raise ValueError("Invalid model type")

    def objective(trial):
        model = model_creator(trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbosity, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])
        val_loss = model.evaluate(X_val, y_val, verbose=verbosity)
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_model = model_creator(best_trial, (X_train.shape[1], X_train.shape[2]), y_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbosity, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=verbosity)
    # Fit the best model
    best_model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=verbosity, callbacks=[early_stopping, reduce_lr])

    # Save the best model in TensorFlow SavedModel format
    best_model.save(f"{model_dir}/{model_type}.keras")

    y_pred = best_model.predict(X_test)

    # Select first horizon
    y_test = y_test[:, 0].flatten()
    y_pred = y_pred[:, 0].flatten()

    # Calculate and round metrics
    mse_score = round(mean_squared_error(y_test, y_pred), 2)
    rmse_score = round(np.sqrt(mse_score), 2)
    mae_score = round(mean_absolute_error(y_test, y_pred), 2)
    smape_score = round(100 * np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))), 2)
    mape_score = round(mean_absolute_percentage_error(y_test, y_pred) * 100, 2)
    r2_score_value = round(r2_score(y_test, y_pred), 4)
    
    # Plot the performance with white background and border
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set the Y-axis limit based on the range of the data
    plt.ylim([min(y_test[:len(y_test)//10]) - 0.5, max(y_test[:len(y_test)//10]) + 0.5])
    
    # Set the X-axis limit based on the number of time steps plotted
    plt.xlim([0, len(y_test[:len(y_test)//10])])
    
    # Plot with custom colors
    plt.plot(y_test[:len(y_test)//10], label='True', color='#ed8787', alpha=0.99)  # Custom red color with transparency
    plt.plot(y_pred[:len(y_pred)//10], label='Predicted', color='#1f77b4', alpha=0.99)  # Custom blue color with transparency

    # Add the metrics to the title
    plt.title(f'Model={model_type}, MSE={mse_score:.2f}, RMSE={rmse_score:.2f}, MAE={mae_score:.2f}, MAPE={mape_score:.2f}%, SMAPE={smape_score:.2f}%, R²={r2_score_value:.2f}')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    
    # Add border lines (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Optional: Adjust the position of the vertical and horizontal lines
    plt.axvline(x=0, color='black', linewidth=1)  # At x=0
    
    # Set the legend with default light gray background
    legend = plt.legend(loc='upper right')
    legend.get_frame().set_facecolor('lightgray')
    
    # Save the plot as a PDF file with a white background
    plt.savefig(f"{plot_dir}/{model_type}_PerformancePlot.pdf", format='pdf', facecolor='white', bbox_inches='tight')
    
    plt.show()

    # Explicitly close the plot to avoid issues with subsequent plots
    plt.close()
    
    # # Plot the optimization history with a larger figure size and white background
    # fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure

    # # Adjust the figure settings
    # fig.set_size_inches(10, 6)
    # fig.patch.set_facecolor('white')

    # # Access the axes to adjust spines
    # ax = fig.gca()
    # ax.set_facecolor('white')

    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(1)

    # # Adjust legend position inside the plot
    # legend = ax.legend(loc='upper right')
    # legend.get_frame().set_facecolor('lightgray')
    
    # # Add a title to the plot
    # plt.title('Optimization History')
    
    # # Save the plot as a PDF file with a white background
    # fig.savefig(f"{plot_dir}/{model_type}_OptimizationHistory.pdf", format='pdf', facecolor='white', bbox_inches='tight')
    
    # # Show the plot
    # plt.show()

    # # Explicitly close the plot to avoid issues with subsequent plots
    # plt.close(fig)


    # Database connection and table creation
    conn = sqlite3.connect('Results/result.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Hyperparameters_DeepLearning (
            dataset_name TEXT,
            model_type TEXT,
            best_params TEXT,
            duration REAL,
            mse_score REAL,
            rmse_score REAL,
            mae_score REAL,
            smape_score REAL,
            mape_score REAL,
            r2_score REAL,
            more_info TEXT,
            UNIQUE(dataset_name, model_type)
        )
    ''')
    
    # Define the value for more_info (you can customize this as needed)
    # more_info = "Additional information or notes about the model"  # Example content
    
    # Insert or update record in SQLite with values rounded to two decimal places
    cursor.execute('''
        INSERT INTO Hyperparameters_DeepLearning (dataset_name, model_type, best_params, duration, mse_score, rmse_score, mae_score, smape_score, mape_score, r2_score, more_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_name, model_type) DO UPDATE SET
            best_params=excluded.best_params,
            duration=excluded.duration,
            mse_score=excluded.mse_score,
            rmse_score=excluded.rmse_score,
            mae_score=excluded.mae_score,
            smape_score=excluded.smape_score,
            mape_score=excluded.mape_score,
            r2_score=excluded.r2_score,
            more_info=excluded.more_info
    ''', (
        df_name,
        f'{model_type}',
        str(best_trial.params),
        round(best_trial.duration.total_seconds(), 2),
        round(mse_score, 2),
        round(rmse_score, 2),
        round(mae_score, 2),
        round(smape_score, 2),
        round(mape_score, 2),
        round(r2_score_value, 2),
        more_info
    ))
    
    conn.commit()
    conn.close()

    return best_model

def optimize_and_save_rf_model(data, df_name, n_trials=10, verbosity=1, model_type='LSTM', more_info='More information about dataset'):
    # Suppress the DataConversionWarning
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    # Define the database-specific model type
    db_model_type = f"{model_type} RandomForest"

    X_train, y_train, X_val, y_val, X_test, y_test = data

    # Directories setup
    result_dir = "./Results"
    os.makedirs(result_dir, exist_ok=True)
    model_dir = f"./{result_dir}/Models/{df_name}"
    plot_dir = f"./{result_dir}/Plots/{df_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load the previously saved model
    best_model = load_model(f"{model_dir}/{model_type}.keras")
    y_train = best_model.predict(X_train)[:, 0]

    # Reshape data
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        scores = cross_val_score(rf, X_train_reshaped, y_train, cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()

    # Optimization
    study = optuna.create_study(direction='minimize', sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)
    best_trial = study.best_trial

    # Train final model
    best_rf = RandomForestRegressor(**best_trial.params, random_state=42)
    best_rf.fit(X_train_reshaped, y_train)

    # Save the best Random Forest model
    joblib.dump(best_rf, f"{model_dir}/{model_type}_RandomForest.pkl")

    # Predictions and metrics
    y_test_pred_flat = best_rf.predict(X_test_reshaped).flatten()
    y_test_flat = y_test[:, 0].flatten()
    mse_score = mean_squared_error(y_test_flat, y_test_pred_flat)
    rmse_score = np.sqrt(mse_score)
    mae_score = mean_absolute_error(y_test_flat, y_test_pred_flat)
    mape_score = np.mean(np.abs((y_test_flat - y_test_pred_flat) / y_test_flat)) * 100
    smape_score = 100 * np.mean(2 * np.abs(y_test_pred_flat - y_test_flat) / (np.abs(y_test_flat) + np.abs(y_test_pred_flat)))
    r2_score_value = r2_score(y_test_flat, y_test_pred_flat)

    # # Plot optimization history
    # fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure
    # fig.set_size_inches(10, 6)
    # fig.patch.set_facecolor('white')
    # ax = fig.gca()
    # ax.set_facecolor('white')

    # for spine in ax.spines.values():
    #     spine.set_edgecolor('black')
    #     spine.set_linewidth(1)
    # legend = ax.legend(loc='upper right')
    # legend.get_frame().set_facecolor('lightgray')
    # plt.title('Optimization History')
    # fig.savefig(f"{plot_dir}/{model_type}_RandomForest_OptimizationHistory.pdf", format='pdf', facecolor='white', bbox_inches='tight')
    # plt.show()
    # plt.close()


    model_type = f'{model_type} RandomForest'
    # Database connection and insertion
    conn = sqlite3.connect('Results/result.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Hyperparameters_RandomForest (
            dataset_name TEXT,
            model_type TEXT,
            best_params TEXT,
            duration REAL,
            mse_score REAL,
            rmse_score REAL,
            mae_score REAL,
            smape_score REAL,
            mape_score REAL,
            r2_score REAL,
            more_info TEXT,
            UNIQUE(dataset_name, model_type)
        )
    ''')

    # Insert or update record in SQLite with values rounded to two decimal places
    cursor.execute('''
        INSERT INTO Hyperparameters_RandomForest (dataset_name, model_type, best_params, duration, mse_score, rmse_score, mae_score, smape_score, mape_score, r2_score, more_info)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset_name, model_type) DO UPDATE SET
            best_params=excluded.best_params,
            duration=excluded.duration,
            mse_score=excluded.mse_score,
            rmse_score=excluded.rmse_score,
            mae_score=excluded.mae_score,
            smape_score=excluded.smape_score,
            mape_score=excluded.mape_score,
            r2_score=excluded.r2_score,
            more_info=excluded.more_info
    ''', (
        df_name,
        db_model_type,  # Use db_model_type here
        str(best_trial.params),
        round(best_trial.duration.total_seconds(), 2),
        round(mse_score, 2),
        round(rmse_score, 2),
        round(mae_score, 2),
        round(smape_score, 2),
        round(mape_score, 2),
        round(r2_score_value, 2),
        more_info
    ))
    # Commit the transaction to save changes to the database
    conn.commit()
    conn.close()
    return best_rf

def save_importance_plots(df_name, X_train, features_name, model_type='CNN', target_column = 'energy_consumption', cmap='coolwarm'):
    # Create the subfolder in Models and Plots directories if it doesn't exist
    result_dir = "./Results"
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f"./Results/Models/{df_name}"
    plot_dir = f"./Results/Plots/{df_name}"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Load the saved Random Forest model
    model_path = f"./{model_dir}/{model_type}_RandomForest.pkl"
    best_rf = joblib.load(model_path)

    # Get the feature importances
    feature_importances = best_rf.feature_importances_

    # Reshape into a 48x10 dataframe
    importances_reshaped = feature_importances.reshape(X_train.shape[1], X_train.shape[2])
    importances_df = pd.DataFrame(importances_reshaped, columns=features_name)
    # importances_df.drop(columns=[target_column], inplace=True)
    importances_df = np.exp(importances_df)
    
    # Open a PDF file to save the plots
    pdf_path = f"{plot_dir}/{model_type}_GlobalFeatures_RandomForest.pdf"
    with PdfPages(pdf_path) as pdf:
        # Plot 1: Heatmap with Annotations
        plt.figure(figsize=(10, 8))
        sns.heatmap(importances_df, cmap=cmap, annot=True, fmt=".4f")
        plt.title(f'Global Feature Importance of Sequence Windows for {model_type} Model - Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Sequences')
        plt.tight_layout()  # Adjust layout to fit everything within the figure
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()

        # Calculate normalized mean absolute values
        mean_absolute_values = importances_df.apply(lambda x: np.mean(np.abs(x)))
        normalized_values = mean_absolute_values / mean_absolute_values.sum()

        # Convert the normalized values to lists
        features = list(normalized_values.index)
        mean_abs_values = list(normalized_values.values)

        # Plot 2: Normalized Mean Absolute Value by Feature
        plt.figure(figsize=(16, 6))  # Further increase figure width to 16 units
        plt.barh(features, mean_abs_values, color='skyblue', edgecolor='black')
        plt.title(f'Normalized Mean Absolute Feature Importance for {model_type} Model - Bar Plot', fontsize=16)
        plt.xlabel('Normalized Mean Absolute Value', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Annotate the bars with the values, keeping the horizontal position unchanged
        for index, value in enumerate(mean_abs_values):
            plt.text(value, index, f'{value:.4f}', va='center', ha='left', fontsize=12)

        plt.tight_layout()  # Adjust layout to fit everything within the figure
        pdf.savefig()  # Save the current figure to the PDF
        plt.close()

    print(f"Plots have been saved to {pdf_path}")
    return normalized_values.values.flatten()

def load_and_analyze_model(data, df_name, model_type, more_info='More information about dataset'):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    
    model_dir = f"./Results/Models/{df_name}"
    model_path = f"{model_dir}/{model_type}.keras"
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Select first horizon
    y_test_flat = y_test[:, 0].flatten()
    y_pred_flat = y_pred[:, 0].flatten()
    
    # Calculate the prediction error
    errors = y_pred_flat - y_test_flat
    
    # Calculate the IQR and determine outliers
    q1 = np.percentile(errors, 25)
    q3 = np.percentile(errors, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_indices = np.where((errors < lower_bound) | (errors > upper_bound))[0]
    
    # Sort the outliers by absolute error values in descending order
    sorted_indices = np.argsort(-np.abs(errors[outlier_indices]))
    outlier_indices = outlier_indices[sorted_indices]
    
    # Prepare the anomalies data for database insertion
    anomalies_data = {
        'dataset_name': df_name,
        'model_type': model_type,
        'index': outlier_indices,
        'error': errors[outlier_indices],
        'prediction': y_pred_flat[outlier_indices],
        'true_value': y_test_flat[outlier_indices],
        'more_info': more_info
    }
    
    anomalies_df = pd.DataFrame(anomalies_data)
    
    # Save to SQLite database
    conn = sqlite3.connect('./Results/result.db')
    cursor = conn.cursor()
    
    # Create table if it does not exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomalies (
            dataset_name TEXT,
            model_type TEXT,
            "index" INTEGER, -- Escaped column name
            error REAL,
            prediction REAL,
            true_value REAL,
            more_info TEXT,
            PRIMARY KEY (dataset_name, model_type, "index") -- Escaped column name here as well
        )
    ''')
    
    # Delete existing entries with the same dataset_name and model_type
    cursor.execute('''
        DELETE FROM anomalies 
        WHERE dataset_name = ? AND model_type = ?
    ''', (df_name, model_type))
    
    # Insert new data
    anomalies_df.to_sql('anomalies', conn, if_exists='append', index=False)
    
    # Commit and close the connection
    conn.commit()
    conn.close()

    # print("Data saved to SQLite database successfully.")
    return outlier_indices

def get_index_values(dataset_name, model_type, database_path = './Results/result.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Execute a query to retrieve the 'index' column from the 'Anomalies' table
    cursor.execute("""
        SELECT "index" FROM Anomalies
        WHERE dataset_name = ? AND model_type = ?
    """, (dataset_name, model_type))

    # Fetch all results and store them in a list
    index_values = [row[0] for row in cursor.fetchall()]

    # Close the connection
    conn.close()

    return index_values

def common_anomaly_ids(model_types, df_name):
    # Define model types and data
    model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN', 'TCN', 'DCNN', 'WaveNet', 'TFT', 'TST']
    
    # Dictionary to store indices for each model type
    indices_dict = {}
    
    # Populate indices_dict with indices from each model type
    for model_type in model_types:
        index_values = get_index_values(dataset_name = df_name, model_type = model_type)
        indices_dict[model_type] = set(index_values)  # Store as a set for easy intersection
    
    # Find common indices across all model types
    common_indices = list(set.intersection(*indices_dict.values()))
    # len(common_indices)
    return common_indices

# def generate_shap_heatmap(mydata, xai_method, model_type, row_id, background_type = 'random',
#                           database_path='./Results/result.db'):
#     # Connect to the SQLite database
#     conn = sqlite3.connect(database_path)
#     cursor = conn.cursor()
#
#     dataset_name = mydata.data_type
#     # Define the plot name and create directories for saving results
#     plot_name = f"{xai_method}_{background_type}_{model_type}"
#     result_dir = f"./Results/SHAP/{dataset_name}/"
#     os.makedirs(result_dir, exist_ok=True)
#
#     # Retrieve shap_df, baseline_real, features_name, and anomaly_data (new_data) from the XAI table
#     cursor.execute('''
#         SELECT shap_df, baseline_real, features_name, anomaly_data
#         FROM XAI
#         WHERE xai_method = ? AND background_type = ? AND dataset_name = ? AND model_type = ? AND row_id = ?
#     ''', (xai_method, background_type, dataset_name, model_type, row_id))
#
#     # Fetch the data
#     row = cursor.fetchone()
#     conn.close()
#
#     # Check if data is found
#     if row:
#         shap_df_json, baseline_prediction, features_name_json, anomaly_data_json = row
#
#         # Convert shap_df to DataFrame
#         shap_df = pd.read_json(StringIO(shap_df_json)) if shap_df_json else pd.DataFrame()
#
#         # Extract feature names from the Index string, excluding 'dtype' entries
#         features_name = re.findall(r"'([^']*)'", features_name_json)
#         if 'object' in features_name:
#             features_name.remove('object')
#
#         # Convert anomaly_data (new_data) to a NumPy array
#         new_data = pd.read_json(StringIO(anomaly_data_json)).values if anomaly_data_json else None
#     else:
#         print("No data found for the specified parameters.")
#         return
#
#     # Prepare SHAP explanation
#     explanation = shap.Explanation(
#         values=np.array(shap_df),
#         base_values=np.full((shap_df.shape[0],), baseline_prediction),
#         data=new_data,
#         feature_names=features_name
#     )
#
#     # Plot the heatmap
#     plt.figure(figsize=(10, 6))
#     shap.plots.heatmap(explanation, show=False)
#     plt.savefig(f"{result_dir}/{plot_name}.pdf", format='pdf', bbox_inches='tight')
#     plt.close()
#
#     print(f"SHAP heatmap saved to {result_dir}/{plot_name}.pdf")



def batch_random_sampling_list(X, batch_size=10, n_samples=10):
    """
    Perform random sampling in batches and save random sample records for each batch.

    Parameters:
        X (numpy.ndarray): Input data of shape (samples, timesteps, features).
        batch_size (int): Size of each batch.
        n_samples (int): Number of random samples to select from each batch.

    Returns:
        list: A list where each element contains randomly selected samples from a batch.
    """
    # Flatten the input data
    X_flattened = X.reshape(X.shape[0], -1)

    # List to store random samples for each batch
    random_samples_list = []

    # Process data in batches
    for batch_start in range(0, X_flattened.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X_flattened.shape[0])
        batch = X_flattened[batch_start:batch_end]

        # Select random samples from the batch
        if len(batch) < n_samples:
            random_samples = batch  # If the batch has fewer rows than n_samples, take all rows
        else:
            random_indices = np.random.choice(len(batch), n_samples, replace=False)
            random_samples = batch[random_indices]

        # Save the random samples for this batch
        random_samples_list.append(random_samples)

    return random_samples_list

def batch_random_sampling(X, batch_size=10, n_samples=10, current_row_id=0):
    """
    Perform random sampling for the current batch based on the current_row_id.

    Parameters:
        X (numpy.ndarray): Input data of shape (samples, timesteps, features).
        batch_size (int): Size of each batch.
        n_samples (int): Number of random samples to select from the current batch.
        current_row_id (int): The row ID to determine the current batch.

    Returns:
        numpy.ndarray: Randomly selected samples reshaped to (n_samples, timesteps * features).
    """
    # Identify the start and end of the current batch
    batch_start = (current_row_id // batch_size) * batch_size
    batch_end = min(batch_start + batch_size, X.shape[0])

    # Extract the current batch
    current_batch = X[batch_start:batch_end]

    # Flatten the last two dimensions (timesteps and features) for each sample
    current_batch_flattened = current_batch.reshape(current_batch.shape[0], -1)

    # Select random samples from the flattened batch
    if len(current_batch_flattened) < n_samples:
        random_samples = current_batch_flattened  # Take all rows if fewer than n_samples
    else:
        random_indices = np.random.choice(len(current_batch_flattened), n_samples, replace=False)
        random_samples = current_batch_flattened[random_indices]

    return random_samples

def batch_kmeans_clustering_list(X, batch_size=10, n_clusters=10):
    """
    Perform K-means clustering in batches and save cluster centers as a list.

    Parameters:
        X (numpy.ndarray): Input data of shape (samples, timesteps, features).
        batch_size (int): Size of each batch.
        n_clusters (int): Number of clusters for K-means.

    Returns:
        list: A list where each element contains the cluster centers for a batch.
    """
    # Flatten the input data
    X_flattened = X.reshape(X.shape[0], -1)

    # Initialize MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

    # List to store cluster centers for each batch
    cluster_centers_list = []

    # Process data in batches
    for batch_start in range(0, X_flattened.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X_flattened.shape[0])
        batch = X_flattened[batch_start:batch_end]
        
        # Update k-means clusters incrementally
        kmeans.partial_fit(batch)
        
        # Save cluster centers for this batch
        cluster_centers_list.append(kmeans.cluster_centers_.copy())

    return cluster_centers_list

def find_minimal_error_points_with_local_threshold(
        mydata,
        model_type,
        min_data_points=10,
        neighbors=2
):
    # Validate inputs
    if min_data_points <= 0:
        raise ValueError("min_data_points must be a positive integer.")
    if neighbors < 0:
        raise ValueError("neighbors must be a non-negative integer.")

    # Load data and model
    X, y, df_name = mydata.X, mydata.y, mydata.data_type
    model_dir = f"./Results/Models/{df_name}"
    rf_model_path = f"{model_dir}/{model_type}_RandomForest.pkl"

    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"Model not found at {rf_model_path}")

    rf_model = joblib.load(rf_model_path)
    X_reshaped = X.reshape(X.shape[0], -1)
    y_pred_rf = rf_model.predict(X_reshaped).flatten()
    y_actual = y[:, 0, 0].flatten()
    rf_errors = np.abs(y_actual - y_pred_rf)

    # Calculate local thresholds (max error in neighborhood window)
    valid_indices = []
    local_thresholds = []

    for i in range(len(rf_errors) - neighbors):
        # Define the neighborhood window (e.g., [i, i+1, ..., i+neighbors])
        window_errors = rf_errors[i:i + neighbors + 1]
        threshold = np.max(window_errors)
        valid_indices.append(i)
        local_thresholds.append(threshold)

    # Sort indices by their local thresholds (ascending) and then by their position
    sorted_indices = sorted(
        zip(valid_indices, local_thresholds),
        key=lambda x: (x[1], x[0])  # Prioritize lower threshold, then earlier index
    )

    # Extract the top N indices
    if len(sorted_indices) < min_data_points:
        raise ValueError(f"Only {len(sorted_indices)} valid points found. Need at least {min_data_points}.")

    selected_indices = [idx for idx, _ in sorted_indices[:min_data_points]]
    selected_thresholds = [thresh for _, thresh in sorted_indices[:min_data_points]]

    # Return the worst-case threshold (max of selected thresholds) and indices
    final_threshold = np.max(selected_thresholds)
    return final_threshold, selected_indices

class DataContainer:
    """
    A container class to store and access training, validation, testing splits, and full DataFrame.
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, X, y, full_data, data_type, more_info, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y
        self.full_data = full_data
        self.data_type = data_type
        self.more_info = more_info
        self.feature_names = feature_names  # New attribute for feature names

def load_and_preprocess_data(dataset_path='./Data/', dataset_type="Residential", 
                             option_number=1, numerical_columns=None, categorical_columns=None, 
                             binary_columns=None, datetime_column=None, target_column=None,
                             scaled=True, scale_type='features', val_ratio=0.1, test_ratio=0.1, 
                             input_seq_length=48, output_seq_length=24):
    """
    Load and preprocess dataset based on the specified type and criteria.

    Returns:
        DataContainer: An object containing training, validation, testing splits, the full DataFrame, 
                       and feature names.
    """
    feature_names = None  # Placeholder for feature names

    if dataset_type == "Residential":
        target_column = 'energy_consumption'
        dataset_folder = os.path.join(dataset_path, 'LondonHydro')
        csv_files = [file for file in os.listdir(dataset_folder) if file.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the specified dataset path.")
        df = pd.read_csv(os.path.join(dataset_folder, csv_files[option_number]))

        # Define columns and preprocess
        numerical_columns = ['energy_consumption', 'temperature', 'humidity', 'wind_speed']
        datetime_column = 'timestamp'
        df[target_column] = np.exp(df[target_column])  # Adjust the target column
        df = preprocess.pre_process(df, numerical_columns, categorical_columns, binary_columns, datetime_column, target_column)
        df = preprocess.get_datetime_features(df, datetime_column, cos_sin=False)

        feature_names = ["Hour", "DayOfWeek", "DayOfMonth", "Month", "DayOfYear", "IsWeekend",
                         "temperature", "humidity", "wind_speed", "energy_consumption"]
        df = df[feature_names]
        data_type = 'Residential'
        more_info = csv_files[option_number]
        
    elif dataset_type in ['Manufacturing facility', 'Office building', 'Retail store', 'Medical clinic']:
        # Set primary use mapping dynamically
        primary_use_map = {
            'Manufacturing facility': 'industrial',
            'Office building': 'office',
            'Retail store': 'retail',
            'Medical clinic': 'health'
        }
        if dataset_type not in primary_use_map:
            raise ValueError(f"Invalid dataset_type: {dataset_type}. Please choose a valid BDG2 dataset type.")

        # Load BDG2 dataset
        target_column = 'energy_consumption'
        dataset_file = os.path.join(dataset_path, 'building-data-genome-project-2/electricity_cleaned.txt')
        df = pd.read_csv(dataset_file)

        # Apply criteria to filter the dataset
        df, more_info = BDG2.get_column_by_criteria(df, primary_use=primary_use_map[dataset_type], option_number=option_number)

        # Extract feature names dynamically
        feature_names = df.columns.tolist()

        data_type = dataset_type
    else:
        raise ValueError("Invalid dataset_type. Choose either 'Residential' or one of the BDG2 options.")

    # Load and preprocess data with sequences
    X_train, y_train, X_val, y_val, X_test, y_test, X, y, full_data = preprocess.load_and_preprocess_data_with_sequences(
        df, target=target_column, scaled=scaled, scale_type=scale_type,
        val_ratio=val_ratio, test_ratio=test_ratio, 
        input_seq_length=input_seq_length, output_seq_length=output_seq_length)

    # Print shapes for verification
    print("\nShapes:")
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Wrap results in DataContainer
    return DataContainer(X_train, y_train, X_val, y_val, X_test, y_test, X, y, full_data, data_type, more_info, feature_names)

def find_nearest_points_to_centers(batch, cluster_centers):
    """
    Find the nearest data point in the batch to each cluster center.

    Parameters:
        batch (numpy.ndarray): Batch of data points.
        cluster_centers (numpy.ndarray): Cluster centers.

    Returns:
        numpy.ndarray: Nearest data points to each cluster center.
    """
    nearest_points = []
    for center in cluster_centers:
        distances = np.linalg.norm(batch - center, axis=1)  # Compute distance to each point in the batch
        nearest_idx = np.argmin(distances)  # Index of the closest point
        nearest_points.append(batch[nearest_idx])  # Add the closest point
    return np.array(nearest_points)

def batch_kmeans_clustering_list(X, batch_size=10, n_clusters=10):
    """
    Perform K-means clustering in batches and return the nearest data points to the cluster centers.

    Parameters:
        X (numpy.ndarray): Input data of shape (samples, timesteps, features).
        batch_size (int): Size of each batch.
        n_clusters (int): Number of clusters for K-means.

    Returns:
        list: A list where each element contains the nearest points to the cluster centers for a batch.
    """
    # Flatten the input data
    X_flattened = X.reshape(X.shape[0], -1)

    # Initialize MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)

    # List to store nearest points for each batch
    nearest_points_list = []

    # Process data in batches
    for batch_start in range(0, X_flattened.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, X_flattened.shape[0])
        batch = X_flattened[batch_start:batch_end]
        
        # Check if the batch contains enough data points
        if len(batch) < n_clusters:
            print(f"Batch size ({len(batch)}) is smaller than the number of clusters ({n_clusters}). Skipping this batch.")
            continue
        
        # Perform K-means clustering on the batch
        kmeans.partial_fit(batch)
        
        # Find the nearest data points to each cluster center
        nearest_points = find_nearest_points_to_centers(batch, kmeans.cluster_centers_)
        nearest_points_list.append(nearest_points)

    return nearest_points_list

def batch_random_sampling(X, batch_size=10, n_samples=10, current_row_id=0):
    """
    Perform random sampling for the current batch only.
    Parameters:
        X (numpy.ndarray): Input data of shape (samples, timesteps, features).
        batch_size (int): Size of each batch.
        n_samples (int): Number of random samples to select.
        current_row_id (int): Row ID to determine the current batch.
    Returns:
        numpy.ndarray: Random samples from the current batch.
    """
    X_flattened = X.reshape(X.shape[0], -1)
    batch_start = (current_row_id // batch_size) * batch_size
    batch_end = min(batch_start + batch_size, X_flattened.shape[0])
    batch = X_flattened[batch_start:batch_end]
    if len(batch) < n_samples:
        return batch # If the batch has fewer rows than n_samples, take all rows
    else:
        random_indices = np.random.choice(len(batch), n_samples, replace=False)
        return batch[random_indices]

#xai_method, model_type, row_id, data_type, noise_factor, background_training, background_size, background_type, horizon = 'TDE', 'CNN', 0, 'original', 0.05, None, 10, 'random', 0

def myshap(
        mydata,
        xai_method='Kernel',
        model_type='CNN',
        row_id=0,
        data_type='original',
        noise_factor=0.05,
        background_training=None,
        background_size=10,
        background_type='random',
        horizon=0,
        replace_sql=True
):
    import os
    import time
    import json
    import sqlite3
    import numpy as np
    import pandas as pd
    import joblib
    import shap
    from keras.models import load_model
    from lime.lime_tabular import LimeTabularExplainer
    # If you have your own TDE explainer, ensure it is imported:
    # from your_module import TimeSeriesSHAPExplainerUpdated

    # Initial data setup
    df_name = mydata.data_type
    x = mydata.X
    y = mydata.y
    features_name = mydata.feature_names
    more_info = mydata.more_info

    if background_training is None:
        random_indices = np.random.choice(mydata.X.shape[0], size=background_size, replace=False)
        background_training = mydata.X[random_indices]

    # Database setup
    conn = sqlite3.connect('./Results/result.db')
    try:
        cursor = conn.cursor()

        # Create table if it does not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS XAI (
                xai_method TEXT,
                background_type TEXT,
                background_size INTEGER,
                dataset_name TEXT,
                more_info TEXT,
                model_type TEXT,
                row_id INTEGER,
                shap_df TEXT,
                anomaly_data TEXT,
                background_training TEXT,
                baseline_real REAL,
                baseline_prediction REAL,
                baseline_error REAL,
                features_name TEXT,
                calculation_time REAL,
                data_type TEXT,
                PRIMARY KEY (xai_method, background_type, background_size, 
                             dataset_name, model_type, row_id, data_type)
            )
        ''')

        # If not replacing, check if an entry exists and skip computation if so.
        if not replace_sql:
            cursor.execute('''
                SELECT 1 FROM XAI
                WHERE xai_method = ? AND background_type = ? AND background_size = ? 
                      AND dataset_name = ? AND model_type = ? AND row_id = ? AND data_type = ?
            ''', (xai_method, background_type, background_size, df_name, model_type, row_id, data_type))
            if cursor.fetchone():
                print(f"Entry exists for {data_type} data. Skipping computation.")
                return

        # --- Computation Section ---
        new_data = x[row_id:row_id + 1, :, :]
        if data_type == 'random':
            noise = new_data * noise_factor * np.random.randn(*new_data.shape)
            new_data = new_data + noise
        real_outcome = y[row_id, horizon, 0]

        # Create directories if needed
        result_dir = f"./Results/SHAP/{df_name}/"
        os.makedirs(result_dir, exist_ok=True)
        model_dir = f"./Results/Models/{df_name}"
        os.makedirs(model_dir, exist_ok=True)

        # Model loading
        best_model = load_model(f"{model_dir}/{model_type}.keras")
        rf_model = joblib.load(f"{model_dir}/{model_type}_RandomForest.pkl")

        # Prediction function
        def model_predict(data):
            preds = best_model.predict(data.reshape((-1, *new_data.shape[1:])))
            return preds[:, horizon] if preds.ndim > 1 else preds

        # SHAP calculation
        start_time = time.time()
        bg_reshaped = background_training.reshape(background_size, -1)
        sample_reshaped = new_data.reshape(1, -1)

        explainer, shap_values, baseline_pred = None, None, None
        if xai_method == 'Kernel':
            explainer = shap.KernelExplainer(model_predict, bg_reshaped, algorithm='linear')
            shap_values = explainer.shap_values(sample_reshaped)
        elif xai_method == 'Tree':
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(sample_reshaped)
        elif xai_method == 'Permutation':
            explainer = shap.PermutationExplainer(model_predict, bg_reshaped)
            shap_values = explainer.shap_values(sample_reshaped)
        elif xai_method == 'Sampling':
            explainer = shap.SamplingExplainer(model_predict, bg_reshaped, silent=True)
            shap_values = explainer.shap_values(sample_reshaped)
        elif xai_method == 'Partition':
            explainer = shap.PartitionExplainer(model_predict, bg_reshaped)
            shap_values = explainer(sample_reshaped).values
        elif xai_method == 'Lime':
            explainer = LimeTabularExplainer(bg_reshaped, mode="regression")
            lime_exp = explainer.explain_instance(sample_reshaped.flatten(), model_predict)
            shap_values = np.zeros((1, np.prod(new_data.shape[1:])))
            for feat_idx, importance in lime_exp.local_exp[1]:
                shap_values[0, feat_idx] = importance
        elif xai_method == 'TDE':
            fastshap_dir = f"./Results/FastSHAP/{df_name}/"
            try:
                loaded_explainer = TimeSeriesSHAPExplainerUpdated.load(
                    fastshap_dir,
                    filename=f"{model_type}"
                )
            except FileNotFoundError:
                raise ValueError(f"No pre-trained TDE explainer found for {model_type}")
            shap_df = loaded_explainer.explain(new_data)
            baseline_pred = loaded_explainer.base_pred.item()
        else:
            raise ValueError("Unsupported XAI method.")

        calculation_time = time.time() - start_time

        if xai_method != 'TDE':
            # Prepare DataFrame for SHAP values
            shap_df = pd.DataFrame(
                shap_values.reshape(new_data.shape[1], -1),
                columns=features_name
            )
            baseline_pred = model_predict(background_training).mean()

        baseline_err = abs(real_outcome - baseline_pred)

        # Prepare data for insertion into the database
        db_data = {
            'xai_method': xai_method,
            'background_type': background_type,
            'background_size': background_size,
            'dataset_name': df_name,
            'more_info': more_info,
            'model_type': model_type,
            'row_id': row_id,
            'shap_df': shap_df.to_json(),
            'anomaly_data': pd.DataFrame(new_data[0]).to_json(),
            'background_training': pd.DataFrame(bg_reshaped).to_json(),
            'baseline_real': float(real_outcome),
            'baseline_prediction': float(baseline_pred),
            'baseline_error': float(baseline_err),
            'features_name': json.dumps(features_name),
            'calculation_time': calculation_time,
            'data_type': data_type
        }

        columns = ', '.join(db_data.keys())
        placeholders = ', '.join(['?'] * len(db_data))

        # Use INSERT OR REPLACE to update existing records if replace_sql is True.
        # Otherwise, the record would have already been skipped.
        if replace_sql:
            sql_query = f'INSERT OR REPLACE INTO XAI ({columns}) VALUES ({placeholders})'
        else:
            sql_query = f'INSERT INTO XAI ({columns}) VALUES ({placeholders})'

        cursor.execute(sql_query, tuple(db_data.values()))
        conn.commit()
        print("Data saved successfully.")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        conn.rollback()
    finally:
        conn.close()

def generate_shap_heatmap_all_rows(mydata, xai_method, model_type, data_type='original',
                                   database_path='./Results/result.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    dataset_name = mydata.data_type
    # Define the plot name and create directories for saving results
    plot_name = f"{xai_method}_{model_type}_{data_type}"
    result_dir = f"./Results/SHAP/{dataset_name}/"
    os.makedirs(result_dir, exist_ok=True)

    # Prepare the multi-page PDF file
    pdf_file_path = f"{result_dir}/{plot_name}.pdf"
    with PdfPages(pdf_file_path) as pdf:
        # Retrieve all rows for the specified parameters
        cursor.execute('''
            SELECT row_id, shap_df, baseline_real, features_name, anomaly_data
            FROM XAI
            WHERE xai_method = ? AND data_type = ? AND dataset_name = ? AND model_type = ?
        ''', (xai_method, data_type, dataset_name, model_type))

        rows = cursor.fetchall()

        if not rows:
            print("No data found for the specified parameters.")
            return

        # Iterate through each row and generate a SHAP heatmap
        for row in rows:
            row_id, shap_df_json, baseline_prediction, features_name_json, anomaly_data_json = row

            # Convert shap_df to DataFrame
            shap_df = pd.read_json(StringIO(shap_df_json)) if shap_df_json else pd.DataFrame()

            # Extract feature names from the Index string, excluding 'dtype' entries
            features_name = mydata.feature_names

            # Convert anomaly_data (new_data) to a NumPy array
            new_data = pd.read_json(StringIO(anomaly_data_json)).values if anomaly_data_json else None

            # Skip if shap_df is empty
            if shap_df.empty:
                print(f"shap_df is empty for row_id {row_id}. Skipping.")
                continue

            # Ensure feature_names matches the number of columns in shap_df
            if len(features_name) != shap_df.shape[1]:
                print(f"Mismatch in feature names and shap_df columns for row_id {row_id}. Adjusting.")
                features_name = features_name[:shap_df.shape[1]]

            # Ensure new_data shape matches shap_df
            if new_data is not None and new_data.shape[1] != shap_df.shape[1]:
                print(f"Mismatch between new_data and shap_df columns for row_id {row_id}. Adjusting.")
                new_data = new_data[:, :shap_df.shape[1]]

            # Prepare SHAP explanation
            explanation = shap.Explanation(
                values=np.array(shap_df),
                base_values=np.full((shap_df.shape[0],), baseline_prediction),
                data=new_data,
                feature_names=features_name
            )

            # Plot the heatmap for this row
            plt.figure(figsize=(10, 6))
            shap.plots.heatmap(explanation, show=False)
            plt.title(f"SHAP Heatmap\nDataset: {dataset_name}, XAI Method: {xai_method}, Model Type: {model_type}, Row ID: {row_id}")
            pdf.savefig(bbox_inches='tight')
            plt.close()

    print(f"SHAP heatmaps saved to {pdf_file_path}")
    conn.close()
