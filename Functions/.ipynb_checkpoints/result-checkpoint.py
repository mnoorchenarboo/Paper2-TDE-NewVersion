import numpy as np

def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))

def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2.0 * np.abs(y_true - y_pred) / np.clip(denominator, 1e-8, None))
