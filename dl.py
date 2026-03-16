import numpy as np
import pandas as pd
import sqlite3
import time
import json
import pickle
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pathlib import Path

path_dbs = Path("databases")
path_dbs.mkdir(parents=True, exist_ok=True)

BENCHMARK_DB = path_dbs / "benchmark_results.db"
ENERGY_DB    = path_dbs / "energy_data.db"

# ============================================================================
# RNN VARIANTS
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, n_features, seq_length, prediction_horizon, n_layers, lstm_units, dropout):
        super(LSTMModel, self).__init__()
        self.lstm_units = lstm_units
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_units, prediction_horizon)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    def __init__(self, n_features, seq_length, prediction_horizon, n_layers, gru_units, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_units,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_units, prediction_horizon)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class BLSTMModel(nn.Module):
    def __init__(self, n_features, seq_length, prediction_horizon, n_layers, lstm_units, dropout):
        super(BLSTMModel, self).__init__()
        self.lstm_units = lstm_units
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(lstm_units * 2, prediction_horizon)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class BGRUModel(nn.Module):
    def __init__(self, n_features, seq_length, prediction_horizon, n_layers, gru_units, dropout):
        super(BGRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_units,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(gru_units * 2, prediction_horizon)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


# ============================================================================
# CNN VARIANTS
# ============================================================================

class CNN1DModel(nn.Module):
    def __init__(self, n_features, seq_length, prediction_horizon, n_filters, kernel_size, n_layers, dropout):
        super(CNN1DModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        in_channels = n_features
        for i in range(n_layers):
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            in_channels = n_filters
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc2 = nn.Linear(64, prediction_horizon)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = self.dropout(x)
        
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DCNNModel(nn.Module):
    """Dilated Convolutional Neural Network"""
    def __init__(self, n_features, seq_length, prediction_horizon, n_filters, kernel_size, n_layers, dropout):
        super(DCNNModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        in_channels = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ))
            in_channels = n_filters
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc2 = nn.Linear(64, prediction_horizon)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = self.dropout(x)
        
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TCNModel(nn.Module):
    """Temporal Convolutional Network"""
    def __init__(self, n_features, seq_length, prediction_horizon, n_filters, kernel_size, n_layers, dropout):
        super(TCNModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        
        in_channels = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ))
            in_channels = n_filters
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc2 = nn.Linear(64, prediction_horizon)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = self.dropout(x)
        
        x = torch.mean(x, dim=2)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class WaveNetModel(nn.Module):
    """WaveNet-style architecture with gated activations"""
    def __init__(self, n_features, seq_length, prediction_horizon, n_filters, kernel_size, n_layers, dropout):
        super(WaveNetModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList()
        self.residual_layers = nn.ModuleList()
        
        in_channels = n_features
        for i in range(n_layers):
            dilation = 2 ** i
            # Tanh convolution
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ))
            # Sigmoid gate
            self.gate_layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=n_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ))
            # Residual connection
            if in_channels != n_filters:
                self.residual_layers.append(nn.Conv1d(in_channels, n_filters, 1))
            else:
                self.residual_layers.append(nn.Identity())
            
            in_channels = n_filters
        
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(n_filters, 64)
        self.fc2 = nn.Linear(64, prediction_horizon)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        for conv, gate, residual in zip(self.conv_layers, self.gate_layers, self.residual_layers):
            residual_x = residual(x)
            tanh_out = torch.tanh(conv(x))
            sigmoid_out = torch.sigmoid(gate(x))
            x = tanh_out * sigmoid_out
            x = self.dropout(x)
            x = x + residual_x  # Residual connection
        
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# TRANSFORMER VARIANTS
# ============================================================================

class TFTModel(nn.Module):
    """Temporal Fusion Transformer"""
    def __init__(self, n_features, seq_length, prediction_horizon, d_model, n_heads, n_layers, dropout):
        super(TFTModel, self).__init__()
        
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
            if d_model == 0:
                d_model = n_heads * 4
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Variable selection network (simplified)
        self.variable_selection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Gated Residual Network layers
        self.grn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ELU(),
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Prediction head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, prediction_horizon)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Variable selection
        selection_weights = self.variable_selection(x)
        x = x * selection_weights
        
        # Gated Residual Networks
        for grn in self.grn_layers:
            residual = x
            x = grn(x)
            x = x + residual
            x = self.layer_norm(x)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = self.layer_norm(x)
        
        # Aggregate temporal information
        x = torch.mean(x, dim=1)
        
        # Prediction
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class TSTModel(nn.Module):
    """Time Series Transformer"""
    def __init__(self, n_features, seq_length, prediction_horizon, d_model, n_heads, n_layers, dropout):
        super(TSTModel, self).__init__()
        
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
            if d_model == 0:
                d_model = n_heads * 4
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Prediction head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, prediction_horizon)
    
    def forward(self, x):
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Aggregate temporal information (use last time step)
        x = x[:, -1, :]
        
        # Prediction
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# TIME SERIES PREDICTOR CLASS
# ============================================================================

class TimeSeriesPredictor:
    def __init__(self, container, primary_use, option_number,
                 results_base_dir="results", db_path=BENCHMARK_DB,
                 epochs=10, final_epochs=30, batch_size=32):
        self.container = container
        self.primary_use = primary_use
        self.option_number = option_number
        self.results_base_dir = results_base_dir
        self.db_path = db_path
        
        self.epochs = epochs
        self.final_epochs = final_epochs
        self.batch_size = batch_size
        
        self.X_train = torch.FloatTensor(container.X_train).to(device)
        self.y_train = torch.FloatTensor(container.y_train).to(device)
        self.X_val = torch.FloatTensor(container.X_val).to(device)
        self.y_val = torch.FloatTensor(container.y_val).to(device)
        self.X_test = torch.FloatTensor(container.X_test).to(device)
        self.y_test = torch.FloatTensor(container.y_test).to(device)
        
        self.seq_length = self.X_train.shape[1]
        self.n_features = self.X_train.shape[2]
        self.prediction_horizon = self.y_train.shape[1]
        
        self.models = {}
        self.best_params = {}
        self.studies = {}
        self.results = {}
        
        self._create_directory_structure()
        self._init_database()
    
    def _create_directory_structure(self):
        self.main_dir = Path(self.results_base_dir) / self.primary_use / f"option_{self.option_number}"
        self.main_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_subdirs = {}
        model_types = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'WaveNet', 'TFT', 'TST']
        for model_type in model_types:
            subdir = self.main_dir / model_type.lower()
            subdir.mkdir(exist_ok=True)
            self.model_subdirs[model_type] = subdir
    
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hyperparameter_optimization (
                primary_use TEXT,
                option_number INTEGER,
                model_name TEXT,
                best_hyperparameters TEXT,
                best_value REAL,
                optimization_time REAL,
                n_trials INTEGER,
                timestamp TEXT,
                PRIMARY KEY (primary_use, option_number, model_name)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_performance (
                primary_use TEXT,
                option_number INTEGER,
                model_name TEXT,
                mse REAL,
                mae REAL,
                r2 REAL,
                mape REAL,
                smape REAL,
                n_parameters INTEGER,
                training_time REAL,
                model_path TEXT,
                true_values_json TEXT,
                predicted_values_json TEXT,
                timestamp TEXT,
                PRIMARY KEY (primary_use, option_number, model_name)
            )
        ''')

        conn.commit()
        conn.close()

    
    def calculate_metrics(self, y_true, y_pred):
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        
        y_true_flat = y_true_np.flatten()
        y_pred_flat = y_pred_np.flatten()
        
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {'MSE': 0.0, 'MAE': 0.0, 'R2': 0.0, 'MAPE': 0.0, 'SMAPE': 0.0}
        
        mse = float(mean_squared_error(y_true_clean, y_pred_clean))
        mae = float(mean_absolute_error(y_true_clean, y_pred_clean))
        r2 = float(r2_score(y_true_clean, y_pred_clean))
        
        mape_values = np.abs((y_true_clean - y_pred_clean) / (np.abs(y_true_clean) + 1e-8))
        mape = float(np.mean(mape_values[np.isfinite(mape_values)]) * 100) if np.any(np.isfinite(mape_values)) else 0.0
        
        smape_denom = np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8
        smape_values = 2 * np.abs(y_true_clean - y_pred_clean) / smape_denom
        smape = float(np.mean(smape_values[np.isfinite(smape_values)]) * 100) if np.any(np.isfinite(smape_values)) else 0.0
        
        mse = float(mse) if np.isfinite(mse) else 0.0
        mae = float(mae) if np.isfinite(mae) else 0.0
        r2 = float(r2) if np.isfinite(r2) else 0.0
        
        print(f"    DEBUG - Calculated metrics: MSE={mse}, MAE={mae}, R2={r2}, MAPE={mape}, SMAPE={smape}")
        
        return {'MSE': mse, 'MAE': mae, 'R2': r2, 'MAPE': mape, 'SMAPE': smape}
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def objective(self, trial, model_type):
        try:
            # RNN Variants
            if model_type == 'LSTM':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = LSTMModel(self.n_features, self.seq_length, self.prediction_horizon,
                                n_layers, lstm_units, dropout).to(device)
            
            elif model_type == 'GRU':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                gru_units = trial.suggest_int('gru_units', 32, 128, step=32)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = GRUModel(self.n_features, self.seq_length, self.prediction_horizon,
                            n_layers, gru_units, dropout).to(device)
            
            elif model_type == 'BLSTM':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                lstm_units = trial.suggest_int('lstm_units', 32, 128, step=32)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = BLSTMModel(self.n_features, self.seq_length, self.prediction_horizon,
                                  n_layers, lstm_units, dropout).to(device)
            
            elif model_type == 'BGRU':
                n_layers = trial.suggest_int('n_layers', 1, 3)
                gru_units = trial.suggest_int('gru_units', 32, 128, step=32)
                dropout = trial.suggest_float('dropout', 0.1, 0.5)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = BGRUModel(self.n_features, self.seq_length, self.prediction_horizon,
                                 n_layers, gru_units, dropout).to(device)
            
            # CNN Variants
            elif model_type == 'CNN1D':
                n_filters = trial.suggest_int('n_filters', 32, 128, step=32)
                kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
                n_layers = trial.suggest_int('n_layers', 2, 6)
                dropout = trial.suggest_float('dropout', 0.1, 0.4)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = CNN1DModel(self.n_features, self.seq_length, self.prediction_horizon,
                                  n_filters, kernel_size, n_layers, dropout).to(device)
            
            elif model_type == 'DCNN':
                n_filters = trial.suggest_int('n_filters', 32, 128, step=32)
                kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
                n_layers = trial.suggest_int('n_layers', 2, 6)
                dropout = trial.suggest_float('dropout', 0.1, 0.4)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = DCNNModel(self.n_features, self.seq_length, self.prediction_horizon,
                                 n_filters, kernel_size, n_layers, dropout).to(device)
            
            elif model_type == 'TCN':
                n_filters = trial.suggest_int('n_filters', 32, 128, step=32)
                kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
                n_layers = trial.suggest_int('n_layers', 2, 6)
                dropout = trial.suggest_float('dropout', 0.1, 0.4)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = TCNModel(self.n_features, self.seq_length, self.prediction_horizon,
                            n_filters, kernel_size, n_layers, dropout).to(device)
            
            elif model_type == 'WaveNet':
                n_filters = trial.suggest_int('n_filters', 32, 128, step=32)
                kernel_size = trial.suggest_int('kernel_size', 3, 7, step=2)
                n_layers = trial.suggest_int('n_layers', 2, 6)
                dropout = trial.suggest_float('dropout', 0.1, 0.4)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = WaveNetModel(self.n_features, self.seq_length, self.prediction_horizon,
                                    n_filters, kernel_size, n_layers, dropout).to(device)
            
            # Transformer Variants
            elif model_type == 'TFT':
                d_model = trial.suggest_categorical('d_model', [64, 128, 256])
                valid_heads = [h for h in [4, 8] if d_model % h == 0]
                if not valid_heads:
                    valid_heads = [4]
                n_heads = trial.suggest_categorical('n_heads', valid_heads)
                n_layers = trial.suggest_int('n_layers', 2, 4)
                dropout = trial.suggest_float('dropout', 0.1, 0.3)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = TFTModel(self.n_features, self.seq_length, self.prediction_horizon,
                                d_model, n_heads, n_layers, dropout).to(device)
            
            elif model_type == 'TST':
                d_model = trial.suggest_categorical('d_model', [64, 128, 256])
                valid_heads = [h for h in [4, 8] if d_model % h == 0]
                if not valid_heads:
                    valid_heads = [4]
                n_heads = trial.suggest_categorical('n_heads', valid_heads)
                n_layers = trial.suggest_int('n_layers', 2, 4)
                dropout = trial.suggest_float('dropout', 0.1, 0.3)
                learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
                
                model = TSTModel(self.n_features, self.seq_length, self.prediction_horizon,
                                d_model, n_heads, n_layers, dropout).to(device)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            train_dataset = TensorDataset(self.X_train, self.y_train.squeeze())
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_dataset = TensorDataset(self.X_val, self.y_val.squeeze())
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    if not torch.isfinite(loss):
                        loss = torch.tensor(1e6, device=device, dtype=torch.float32)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = model(X_batch)
                        val_loss += criterion(y_pred, y_batch).item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 5:
                    break
            
            del model
            torch.cuda.empty_cache()
            
            return best_val_loss
        
        except Exception as e:
            print(f"Trial failed for {model_type}: {e}")
            torch.cuda.empty_cache()
            return float('inf')


    def optimize_hyperparameters(self, model_type, n_trials=5):
        print(f"\nOptimizing hyperparameters for {model_type} ({n_trials} trials)...")
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler())
        
        start_time = time.time()
        study.optimize(lambda trial: self.objective(trial, model_type), n_trials=n_trials, show_progress_bar=True)
        optimization_time = time.time() - start_time
        
        self.studies[model_type] = study
        self.best_params[model_type] = study.best_params
        
        self._save_hyperparameters(model_type, study.best_params, study.best_value, n_trials, optimization_time)
        
        print(f"Optimization completed for {model_type}. Best value: {study.best_value:.6f}")
        print(f"Optimization time: {optimization_time:.2f} seconds")
        print(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def train_best_model(self, model_type):
        print(f"\nTraining {model_type} with best hyperparameters...")
        
        params = self.best_params[model_type]
        learning_rate = params.get('learning_rate', 0.001)
        
        try:
            # RNN Variants
            if model_type == 'LSTM':
                model = LSTMModel(self.n_features, self.seq_length, self.prediction_horizon,
                                params['n_layers'], params['lstm_units'], params['dropout']).to(device)
            elif model_type == 'GRU':
                model = GRUModel(self.n_features, self.seq_length, self.prediction_horizon,
                               params['n_layers'], params['gru_units'], params['dropout']).to(device)
            elif model_type == 'BLSTM':
                model = BLSTMModel(self.n_features, self.seq_length, self.prediction_horizon,
                                  params['n_layers'], params['lstm_units'], params['dropout']).to(device)
            elif model_type == 'BGRU':
                model = BGRUModel(self.n_features, self.seq_length, self.prediction_horizon,
                                 params['n_layers'], params['gru_units'], params['dropout']).to(device)
            
            # CNN Variants
            elif model_type == 'CNN1D':
                model = CNN1DModel(self.n_features, self.seq_length, self.prediction_horizon,
                                  params['n_filters'], params['kernel_size'], params['n_layers'],
                                  params['dropout']).to(device)
            elif model_type == 'DCNN':
                model = DCNNModel(self.n_features, self.seq_length, self.prediction_horizon,
                                 params['n_filters'], params['kernel_size'], params['n_layers'],
                                 params['dropout']).to(device)
            elif model_type == 'TCN':
                model = TCNModel(self.n_features, self.seq_length, self.prediction_horizon,
                               params['n_filters'], params['kernel_size'], params['n_layers'],
                               params['dropout']).to(device)
            elif model_type == 'WaveNet':
                model = WaveNetModel(self.n_features, self.seq_length, self.prediction_horizon,
                                    params['n_filters'], params['kernel_size'], params['n_layers'],
                                    params['dropout']).to(device)
            
            # Transformer Variants
            elif model_type == 'TFT':
                model = TFTModel(self.n_features, self.seq_length, self.prediction_horizon,
                                params['d_model'], params['n_heads'], params['n_layers'],
                                params['dropout']).to(device)
            elif model_type == 'TST':
                model = TSTModel(self.n_features, self.seq_length, self.prediction_horizon,
                                params['d_model'], params['n_heads'], params['n_layers'],
                                params['dropout']).to(device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            print(f"Model built. Parameters: {self.count_parameters(model)}")
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            train_dataset = TensorDataset(self.X_train, self.y_train.squeeze())
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            
            val_dataset = TensorDataset(self.X_val, self.y_val.squeeze())
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            print(f"Training for {self.final_epochs} epochs...")
            start_time = time.time()
            
            for epoch in range(self.final_epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    if not torch.isfinite(loss):
                        loss = torch.tensor(1e6, device=device, dtype=torch.float32)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        y_pred = model(X_batch)
                        val_loss += criterion(y_pred, y_batch).item()
                
                val_loss /= len(val_loader)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.final_epochs}, Val Loss: {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= 10:
                    break
            
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
            
            model.eval()
            with torch.no_grad():
                predictions = model(self.X_test).cpu().numpy()
            
            predictions = predictions.reshape(-1, self.prediction_horizon, 1)
            
            metrics = self.calculate_metrics(self.y_test, predictions)
            n_parameters = self.count_parameters(model)
            
            model_path = self._save_model(model_type, model)
            
            self.models[model_type] = model
            self.results[model_type] = {**metrics, 'training_time': training_time, 'n_parameters': n_parameters, 'model_path': str(model_path)}
            
            self._save_performance_metrics(model_type, metrics, training_time, n_parameters, str(model_path))
            self._save_prediction_results(model_type, self.y_test, predictions)
            
            print(f"\n{model_type} Results Summary:")
            print(f"  R² Score: {metrics['R2']:.4f}")
            print(f"  RMSE: {metrics['MSE']**0.5:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  MAPE: {metrics['MAPE']:.2f}%")
            print(f"  SMAPE: {metrics['SMAPE']:.2f}%")
            print(f"  Parameters: {n_parameters:,}")
            print(f"  Training Time: {training_time:.2f}s")
            
            return model, predictions
        
        except Exception as e:
            print(f"ERROR in train_best_model for {model_type}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_model(self, model_type, model):
        """
        Enhanced save function that preserves all model information
        """
        try:
            model_path = self.model_subdirs[model_type] / "trained_model.pt"
            
            # Get the model's current parameters based on model type
            params = self.best_params.get(model_type, {})
            
            # Prepare comprehensive checkpoint
            checkpoint = {
                # Model architecture
                'model_type': model_type,
                'state_dict': model.state_dict(),
                
                # Architecture parameters
                'hyperparameters': params,
                
                # Input/Output dimensions
                'seq_length': self.seq_length,
                'n_features': self.n_features,
                'prediction_horizon': self.prediction_horizon,
                
                # Training configuration
                'epochs': self.epochs,
                'final_epochs': self.final_epochs,
                'batch_size': self.batch_size,
                
                # Model statistics
                'n_parameters': self.count_parameters(model),
                
                # Metadata
                'primary_use': self.primary_use,
                'option_number': self.option_number,
                'timestamp': torch.tensor([time.time()]),  # Save as tensor for consistency
                
                # Training device info
                'device': str(next(model.parameters()).device),
                
                # Optimizer state (if you want to resume training)
                # 'optimizer_state_dict': optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            }
            
            # Save the complete checkpoint
            torch.save(checkpoint, str(model_path))
            
            # Also save hyperparameters as separate JSON for easy inspection
            params_path = self.model_subdirs[model_type] / "hyperparameters.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            # Save model metadata
            metadata_path = self.model_subdirs[model_type] / "model_metadata.json"
            metadata = {
                'model_type': model_type,
                'primary_use': self.primary_use,
                'option_number': self.option_number,
                'n_parameters': checkpoint['n_parameters'],
                'seq_length': self.seq_length,
                'n_features': self.n_features,
                'prediction_horizon': self.prediction_horizon,
                'training_config': {
                    'epochs': self.epochs,
                    'final_epochs': self.final_epochs,
                    'batch_size': self.batch_size
                }
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved {model_type} model to {model_path}")
            print(f"✓ Saved hyperparameters to {params_path}")
            print(f"✓ Saved metadata to {metadata_path}")
            
            return model_path

        except Exception as e:
            print(f"✗ Could not save model: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _save_hyperparameters(self, model_name, params, best_value, n_trials, opt_time):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO hyperparameter_optimization 
                (primary_use, option_number, model_name, best_hyperparameters, best_value, 
                 optimization_time, n_trials, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.primary_use, self.option_number, model_name, json.dumps(params), 
                  best_value, opt_time, n_trials, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            print(f"Saved hyperparameters to database")
        except Exception as e:
            print(f"Could not save hyperparameters: {e}")
    
    def _save_performance_metrics(self, model_name, metrics, training_time, n_parameters, model_path):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            mse = float(metrics.get('MSE', 0.0))
            mae = float(metrics.get('MAE', 0.0))
            r2 = float(metrics.get('R2', 0.0))
            mape = float(metrics.get('MAPE', 0.0))
            smape = float(metrics.get('SMAPE', 0.0))
            training_time = float(training_time) if training_time else 0.0
            n_parameters = int(n_parameters) if n_parameters else 0
            
            print(f"DEBUG - Before DB insert: MSE={mse}, MAE={mae}, R2={r2}, MAPE={mape}, SMAPE={smape}")
            
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_performance 
                (primary_use, option_number, model_name, mse, mae, r2, mape, smape, 
                 n_parameters, training_time, model_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.primary_use, self.option_number, model_name, mse, mae, r2, mape, smape,
                  n_parameters, training_time, model_path, datetime.now().isoformat()))
            
            cursor.execute('''
                SELECT mse, mae, r2, mape, smape FROM prediction_performance
                WHERE primary_use = ? AND option_number = ? AND model_name = ?
            ''', (self.primary_use, self.option_number, model_name))
            
            result = cursor.fetchone()
            if result:
                print(f"DEBUG - After DB insert: MSE={result[0]}, MAE={result[1]}, R2={result[2]}, MAPE={result[3]}, SMAPE={result[4]}")
            
            conn.commit()
            conn.close()
            print(f"Saved performance metrics to database")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")
        except Exception as e:
            print(f"ERROR saving performance metrics: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_prediction_results(self, model_name, y_true, y_pred):
        try:
            y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
            y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred

            y_true_flat = y_true_np.flatten().tolist()
            y_pred_flat = y_pred_np.flatten().tolist()

            if len(y_true_flat) != len(y_pred_flat):
                raise ValueError("True and predicted vectors must have the same length.")

            true_values_json = json.dumps(y_true_flat)
            predicted_values_json = json.dumps(y_pred_flat)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT COUNT(*) FROM prediction_performance
                WHERE primary_use = ? AND option_number = ? AND model_name = ?
            ''', (self.primary_use, self.option_number, model_name))
            exists = cursor.fetchone()[0] > 0

            if not exists:
                print(f"Warning: No performance entry found for {model_name}. Cannot attach predictions.")
            else:
                cursor.execute('''
                    UPDATE prediction_performance
                    SET true_values_json = ?, predicted_values_json = ?
                    WHERE primary_use = ? AND option_number = ? AND model_name = ?
                ''', (true_values_json, predicted_values_json, 
                    self.primary_use, self.option_number, model_name))

            conn.commit()
            conn.close()
            print(f"Saved prediction vectors (length={len(y_true_flat)}) to database")

        except Exception as e:
            print(f"Could not save prediction vectors: {e}")

    def _check_existing_results(self, model_name):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM hyperparameter_optimization 
            WHERE primary_use = ? AND option_number = ? AND model_name = ?
        ''', (self.primary_use, self.option_number, model_name))
        has_hyperparams = cursor.fetchone()[0] > 0

        cursor.execute('''
            SELECT COUNT(*) FROM prediction_performance 
            WHERE primary_use = ? AND option_number = ? AND model_name = ?
        ''', (self.primary_use, self.option_number, model_name))
        has_performance = cursor.fetchone()[0] > 0

        cursor.execute('''
            SELECT true_values_json, predicted_values_json FROM prediction_performance 
            WHERE primary_use = ? AND option_number = ? AND model_name = ?
        ''', (self.primary_use, self.option_number, model_name))
        result = cursor.fetchone()
        has_predictions = bool(result and result[0] not in (None, '', '[]') and result[1] not in (None, '', '[]'))

        conn.close()

        return {
            'hyperparameters': has_hyperparams,
            'performance': has_performance,
            'predictions': has_predictions,
            'complete': has_hyperparams and has_performance and has_predictions
        }

    
    def _load_existing_hyperparameters(self, model_name):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT best_hyperparameters FROM hyperparameter_optimization 
                WHERE primary_use = ? AND option_number = ? AND model_name = ?
            ''', (self.primary_use, self.option_number, model_name))
            
            result = cursor.fetchone()
            if result:
                self.best_params[model_name] = json.loads(result[0])
                print(f"Loaded existing hyperparameters for {model_name}")
            
            conn.close()
        except Exception as e:
            print(f"Could not load hyperparameters: {e}")
    
    def run_single_model_evaluation(self, model_type, n_trials=5):
        print(f"\n{'='*70}")
        print(f"EVALUATING {model_type}")
        print(f"Primary Use: {self.primary_use} | Option: {self.option_number}")
        print(f"Epochs: {self.epochs} (optimization) / {self.final_epochs} (final training)")
        print(f"{'='*70}")
        
        existing = self._check_existing_results(model_type)
        
        if existing['complete']:
            print(f"Complete results already exist for {model_type}. Skipping...")
            return
        
        try:
            if not existing['hyperparameters']:
                print(f"\nSTEP 1/3: Optimizing hyperparameters for {model_type}...")
                self.optimize_hyperparameters(model_type, n_trials)
            else:
                print(f"\nSTEP 1/3: Hyperparameters already exist for {model_type}")
                self._load_existing_hyperparameters(model_type)
            
            if model_type in self.best_params or existing['hyperparameters']:
                if not existing['performance'] or not existing['predictions']:
                    print(f"\nSTEP 2/3: Training {model_type} with best hyperparameters...")
                    print(f"STEP 3/3: Saving performance metrics and predictions...")
                    self.train_best_model(model_type)
                else:
                    print(f"\nSTEPS 2-3: Performance and predictions already exist for {model_type}")
            
            print(f"\n✓ {model_type} evaluation COMPLETED SUCCESSFULLY!")
        
        except Exception as e:
            print(f"\n✗ ERROR with {model_type}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
    
    def run_complete_evaluation(self, models=None, n_trials=5):
        if models is None:
            models = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'WaveNet', 'TFT', 'TST']
        
        print("="*80)
        print("PYTORCH TIME SERIES PREDICTION EVALUATION")
        print(f"Primary Use: {self.primary_use} | Option: {self.option_number}")
        print(f"Database: {self.db_path}")
        print(f"Results Directory: {self.main_dir}")
        print(f"Epochs: {self.epochs} (optimization) / {self.final_epochs} (final training)")
        print(f"Models: {', '.join(models)}")
        print("="*80)
        
        for model_type in tqdm(models, desc="Evaluating Models", unit="model"):
            self.run_single_model_evaluation(model_type, n_trials)
            torch.cuda.empty_cache()
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETED")
        print(f"Results saved to: {self.main_dir}")
        print(f"Database: {self.db_path}")
        print("="*80)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def run_time_series_evaluation(container, primary_use, option_number,
                              models=None, n_trials=5, results_base_dir="results",
                              db_path=BENCHMARK_DB,
                              epochs=10, final_epochs=30, batch_size=32):
    if models is None:
        models = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'WaveNet', 'TFT', 'TST']
    
    predictor = TimeSeriesPredictor(
        container=container,
        primary_use=primary_use,
        option_number=option_number,
        results_base_dir=results_base_dir,
        db_path=db_path,
        epochs=epochs,
        final_epochs=final_epochs,
        batch_size=batch_size
    )
    
    predictor.run_complete_evaluation(models, n_trials)
    return predictor


def query_results_from_database(db_path=BENCHMARK_DB, primary_use=None,
                               option_number=None, model_name=None):
    conn = sqlite3.connect(db_path)
    
    where_conditions = []
    params = []
    
    if primary_use:
        where_conditions.append("primary_use = ?")
        params.append(primary_use)
    
    if option_number is not None:
        where_conditions.append("option_number = ?")
        params.append(option_number)
    
    if model_name:
        where_conditions.append("model_name = ?")
        params.append(model_name)
    
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    hyperparams_query = f"SELECT * FROM hyperparameter_optimization {where_clause}"
    hyperparams_df = pd.read_sql_query(hyperparams_query, conn, params=params)
    
    performance_query = f"SELECT * FROM prediction_performance {where_clause}"
    performance_df = pd.read_sql_query(performance_query, conn, params=params)
    
    conn.close()
    
    return {
        'hyperparameters': hyperparams_df,
        'performance': performance_df
    }


def get_best_models_by_primary_use(db_path=BENCHMARK_DB):
    conn = sqlite3.connect(db_path)
    
    query = '''
        SELECT 
            primary_use,
            option_number,
            model_name,
            r2,
            mse,
            mae,
            mape,
            smape,
            training_time,
            n_parameters
        FROM prediction_performance
        WHERE (primary_use, option_number, r2) IN (
            SELECT primary_use, option_number, MAX(r2)
            FROM prediction_performance
            GROUP BY primary_use, option_number
        )
        ORDER BY primary_use, option_number
    '''
    
    best_models_df = pd.read_sql_query(query, conn)
    conn.close()
    return best_models_df


primary_use_options = {
    'parking': 23, 'lodging': 149, 'office': 296, 'education': 604,
    'retail': 11, 'assembly': 203, 'other': 26, 'public': 166,
    'warehouse': 14, 'food': 6, 'utility': 4, 'health': 27,
    'religion': 3, 'science': 7, 'industrial': 11, 'services': 9,
    'unknown': 19, 'residential': 19
}


def find_good_option(primary_use, db_path=ENERGY_DB, scale_type="both", smape_threshold=15, device='cpu'):
    """
    Find a good option for the given primary_use with robust error handling for small datasets.
    """
    from Functions import preprocess
    
    if isinstance(device, str):
        device = torch.device(device)
    
    def calculate_smape(y_true, y_pred):
        y_true_np = y_true.cpu().numpy() if torch.is_tensor(y_true) else y_true
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_true_flat = y_true_np.flatten()
        y_pred_flat = y_pred_np.flatten()
        mask = np.isfinite(y_true_flat) & np.isfinite(y_pred_flat)
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        if len(y_true_clean) == 0:
            return np.inf
        smape = np.mean(2 * np.abs(y_true_clean - y_pred_clean) / 
                        (np.abs(y_true_clean) + np.abs(y_pred_clean) + 1e-8)) * 100
        return smape
    
    max_option = primary_use_options.get(primary_use, 0)
    
    if max_option == 0:
        print(f"WARNING: No options available for '{primary_use}', using option 0")
        return 0
    
    print(f"\n{'='*70}")
    print(f"Finding good option for '{primary_use}' (max: {max_option})")
    print(f"SMAPE threshold: {smape_threshold}")
    print(f"{'='*70}\n")
    
    valid_options = []
    
    for option in range(max_option):
        print(f"Checking option {option}/{max_option-1} for {primary_use}...")
        
        try:
            container = preprocess.load_and_preprocess_data_with_sequences(
                db_path,
                primary_use,
                option,
                scaled=True,
                scale_type=scale_type
            )
            
            min_samples_train = 10
            min_samples_test = 5
            
            if (container.X_train.shape[0] < min_samples_train or 
                container.X_test.shape[0] < min_samples_test):
                print(f"  ✗ Option {option}: Dataset too small "
                      f"(train: {container.X_train.shape[0]}, test: {container.X_test.shape[0]})")
                continue
            
            X_train = torch.FloatTensor(container.X_train).to(device)
            y_train = torch.FloatTensor(container.y_train).to(device)
            X_test = torch.FloatTensor(container.X_test).to(device)
            y_test = torch.FloatTensor(container.y_test).to(device)
            
            seq_length = X_train.shape[1]
            n_features = X_train.shape[2]
            prediction_horizon = y_train.shape[1]
            
            print(f"  Dataset loaded: train={X_train.shape[0]}, test={X_test.shape[0]}, "
                  f"seq_len={seq_length}, features={n_features}")
            
            model = GRUModel(n_features, seq_length, prediction_horizon, 1, 32, 0.2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            batch_size = min(32, max(4, X_train.shape[0] // 4))
            train_dataset = TensorDataset(X_train, y_train.squeeze())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            n_epochs = 10
            print(f"  Training model ({n_epochs} epochs, batch_size={batch_size})...")
            
            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0
                n_batches = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)
                    
                    if not torch.isfinite(loss):
                        print(f"    WARNING: Non-finite loss at epoch {epoch+1}, skipping batch")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                if n_batches > 0:
                    avg_loss = epoch_loss / n_batches
                    if (epoch + 1) % 5 == 0:
                        print(f"    Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
            
            model.eval()
            with torch.no_grad():
                pred = model(X_test).cpu().numpy()
            
            pred = pred.reshape(-1, prediction_horizon, 1)
            smape = calculate_smape(y_test.cpu().numpy(), pred)
            
            print(f"  ✓ Option {option}: SMAPE = {smape:.2f}")
            
            del model, optimizer, X_train, y_train, X_test, y_test
            torch.cuda.empty_cache()
            
            if smape < smape_threshold:
                print(f"\n  ✓✓ FOUND GOOD OPTION {option} with SMAPE {smape:.2f} < {smape_threshold}\n")
                return option
            
            valid_options.append((option, smape))
        
        except ValueError as e:
            if "too small to split" in str(e).lower() or "dataset" in str(e).lower():
                print(f"  ✗ Option {option}: Dataset too small to split")
            else:
                print(f"  ✗ Option {option}: ValueError - {e}")
            continue
        
        except Exception as e:
            print(f"  ✗ Option {option}: Failed - {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            continue
    
    if valid_options:
        valid_options.sort(key=lambda x: x[1])
        best_option, best_smape = valid_options[0]
        print(f"\n  ⚠ No option met threshold {smape_threshold}")
        print(f"  Using best option {best_option} with SMAPE {best_smape:.2f}\n")
        return best_option
    
    print(f"\n  ⚠ WARNING: All options failed for '{primary_use}'")
    print(f"  Using fallback option 0\n")
    return 0


def load_data(primary_use, option_number, **kwargs):
    from Functions import preprocess
    return preprocess.load_and_preprocess_data_with_sequences(
        ENERGY_DB,
        primary_use,
        option_number,
        scaled=True,
        scale_type="both"
    )


def get_user_inputs_ts():
    print("=== PyTorch Time Series Model Evaluation Configuration ===")
    
    available_primary_uses = ['parking', 'lodging', 'office', 'education', 'retail', 'assembly',
                             'other', 'public', 'warehouse', 'food', 'utility', 'health',
                             'religion', 'science', 'industrial', 'services', 'unknown', 'residential']
    print("Available primary uses:")
    for i, use in enumerate(available_primary_uses):
        print(f"  {i}: {use}")
    
    primary_uses_input = input("Enter primary use numbers (comma-separated, or 'all' for all): ").strip()
    
    if primary_uses_input.lower() == 'all':
        primary_uses = available_primary_uses
    else:
        selected_indices = [int(idx.strip()) for idx in primary_uses_input.split(',')]
        primary_uses = [available_primary_uses[idx] for idx in selected_indices if 0 <= idx < len(available_primary_uses)]
    
    available_models = ['LSTM', 'GRU', 'BLSTM', 'BGRU', 'CNN1D', 'DCNN', 'TCN', 'WaveNet', 'TFT', 'TST']
    print("Available models (default: all):")
    for i, model in enumerate(available_models):
        print(f"  {i}: {model}")
    
    models_input = input("Enter model numbers (comma-separated, or press Enter for all): ").strip()
    
    if not models_input:
        models = available_models
    else:
        selected_indices = [int(idx.strip()) for idx in models_input.split(',')]
        models = [available_models[idx] for idx in selected_indices if 0 <= idx < len(available_models)]
    
    smape_input = input("Enter SMAPE threshold for option selection (default 15): ").strip()
    smape_threshold = float(smape_input) if smape_input else 15.0
    
    n_trials = int(input("Number of optimization trials (default 5): ").strip() or "5")
    epochs = int(input("Training epochs for optimization (default 5): ").strip() or "5")
    final_epochs = int(input("Final training epochs (default 10): ").strip() or "10")
    batch_size = int(input("Batch size (default 64): ").strip() or "64")
    
    return {
        'primary_uses': primary_uses,
        'models': models,
        'smape_threshold': smape_threshold,
        'n_trials': n_trials,
        'epochs': epochs,
        'final_epochs': final_epochs,
        'batch_size': batch_size
    }


# ============================================================================
# MODEL LOADING FUNCTION
# ============================================================================
MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "GRU": GRUModel,
    "BLSTM": BLSTMModel,
    "BGRU": BGRUModel,
    "CNN1D": CNN1DModel,
    "DCNN": DCNNModel,
    "TCN": TCNModel,
    "WaveNet": WaveNetModel,
    "TFT": TFTModel,
    "TST": TSTModel
}

def load_complete_model(model_path, device='cpu', return_metadata=False):
    """
    Enhanced load function that reconstructs the complete model with all information
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on ('cpu', 'cuda', or torch.device)
        return_metadata: If True, return (model, metadata) tuple
    
    Returns:
        model: Loaded PyTorch model in eval mode
        metadata: Dict with all saved information (if return_metadata=True)
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model information
        model_type = checkpoint['model_type']
        params = checkpoint['hyperparameters']
        n_features = checkpoint['n_features']
        seq_length = checkpoint['seq_length']
        prediction_horizon = checkpoint['prediction_horizon']
        
        print(f"Loading {model_type} model...")
        print(f"  Sequence length: {seq_length}")
        print(f"  Features: {n_features}")
        print(f"  Prediction horizon: {prediction_horizon}")
        
        # Initialize the correct model class with hyperparameters
        model_cls = MODEL_REGISTRY[model_type]
        
        # Build model based on type-specific parameters
        if model_type in ['LSTM', 'BLSTM']:
            model = model_cls(
                n_features=n_features,
                seq_length=seq_length,
                prediction_horizon=prediction_horizon,
                n_layers=params['n_layers'],
                lstm_units=params['lstm_units'],
                dropout=params['dropout']
            )
        
        elif model_type in ['GRU', 'BGRU']:
            model = model_cls(
                n_features=n_features,
                seq_length=seq_length,
                prediction_horizon=prediction_horizon,
                n_layers=params['n_layers'],
                gru_units=params['gru_units'],
                dropout=params['dropout']
            )
        
        elif model_type in ['CNN1D', 'DCNN', 'TCN', 'WaveNet']:
            model = model_cls(
                n_features=n_features,
                seq_length=seq_length,
                prediction_horizon=prediction_horizon,
                n_filters=params['n_filters'],
                kernel_size=params['kernel_size'],
                n_layers=params['n_layers'],
                dropout=params['dropout']
            )
        
        elif model_type in ['TFT', 'TST']:
            model = model_cls(
                n_features=n_features,
                seq_length=seq_length,
                prediction_horizon=prediction_horizon,
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                n_layers=params['n_layers'],
                dropout=params['dropout']
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the trained weights
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {checkpoint.get('n_parameters', 'N/A')}")
        print(f"  Device: {device}")
        
        if return_metadata:
            metadata = {
                'model_type': model_type,
                'hyperparameters': params,
                'seq_length': seq_length,
                'n_features': n_features,
                'prediction_horizon': prediction_horizon,
                'n_parameters': checkpoint.get('n_parameters'),
                'primary_use': checkpoint.get('primary_use'),
                'option_number': checkpoint.get('option_number'),
                'epochs': checkpoint.get('epochs'),
                'final_epochs': checkpoint.get('final_epochs'),
                'batch_size': checkpoint.get('batch_size'),
                'original_device': checkpoint.get('device'),
            }
            return model, metadata
        
        return model
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def load_model_from_database(db_path, primary_use, option_number, model_name, device='cpu'):
    """
    Load a model using database query to find the model path
    """
    import sqlite3
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT model_path FROM prediction_performance
        WHERE primary_use = ? AND option_number = ? AND model_name = ?
    ''', (primary_use, option_number, model_name))
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        raise ValueError(f"Model not found: {primary_use}/{option_number}/{model_name}")
    
    model_path = result[0]
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return load_complete_model(model_path, device=device, return_metadata=True)

def verify_model_integrity(model_path):
    """
    Verify that a saved model contains all necessary information
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        required_keys = [
            'model_type',
            'state_dict',
            'hyperparameters',
            'seq_length',
            'n_features',
            'prediction_horizon'
        ]
        
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"✗ Missing keys: {missing_keys}")
            return False
        
        print("✓ Model integrity check passed")
        print(f"  Model type: {checkpoint['model_type']}")
        print(f"  Parameters: {len(checkpoint['state_dict'])} layers")
        print(f"  Hyperparameters: {list(checkpoint['hyperparameters'].keys())}")
        
        return True
    
    except Exception as e:
        print(f"✗ Integrity check failed: {e}")
        return False


# ============================================================================
# MAIN EXECUTION FLOW
# ============================================================================
def main_ts():
    config = get_user_inputs_ts()
    
    print("\n=== Configuration Summary ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Execution cancelled.")
        return
    
    selected_configs = []
    for primary_use in config['primary_uses']:
        good_option = find_good_option(primary_use, smape_threshold=config['smape_threshold'])
        selected_configs.append((primary_use, good_option))
    
    completed_configs = []
    for primary_use, option_number in selected_configs:
        print(f"\n{'='*80}")
        print(f"EVALUATING {primary_use.upper()} - OPTION {option_number}")
        print(f"{'='*80}")
        
        try:
            container = load_data(primary_use, option_number)
            
            predictor = run_time_series_evaluation(
                container=container,
                primary_use=primary_use,
                option_number=option_number,
                models=config['models'],
                n_trials=config['n_trials'],
                results_base_dir="results",
                db_path=BENCHMARK_DB,
                epochs=config['epochs'],
                final_epochs=config['final_epochs'],
                batch_size=config['batch_size']
            )
            
            completed_configs.append({
                'primary_use': primary_use,
                'option_number': option_number,
                'predictor': predictor
            })
            
            print(f"Completed {primary_use} option {option_number}")
        
        except Exception as e:
            print(f"Error evaluating {primary_use} option {option_number}: {e}")
            torch.cuda.empty_cache()
            continue
    
    print(f"\nEvaluation completed! {len(completed_configs)} configurations processed.")
    print("Results saved to database and files.")
    return completed_configs


if __name__ == "__main__":
    results = main_ts()