import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

def replace_rare_categories(df, columns, threshold=0.01):
    if columns is None:
        return df
    for col in columns:
        value_counts = df[col].value_counts(normalize=True)
        categories_to_replace = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: f'{col}_other' if x in categories_to_replace else x)
    return df

def one_hot_encode(df, categorical_columns): 
    if categorical_columns is None:
        return df
    encoder = OneHotEncoder(drop=None, sparse_output=False)
    categorical_features = df[categorical_columns]
    encoded_features = encoder.fit_transform(categorical_features)
    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    
    for col in categorical_columns:
        col_prefix = f"{col}_"
        other_columns = [c for c in encoded_features_df.columns if c.startswith(col_prefix) and 'other' in c.lower()]
        if other_columns:
            encoded_features_df = encoded_features_df.drop(columns=other_columns[0])

    df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_features_df], axis=1)
    return df_encoded

def fill_missing_values(df, numerical_columns, categorical_columns=None, binary_columns=None):
    df_filled = df.copy()
    
    if numerical_columns is not None:
        for col in numerical_columns:
            for i in df.index:
                if pd.isna(df_filled.at[i, col]):
                    neighbors = []
                    if i > 1:
                        neighbors.append(df_filled.at[i-2, col])
                    if i > 0:
                        neighbors.append(df_filled.at[i-1, col])
                    if i < len(df)-1:
                        neighbors.append(df_filled.at[i+1, col])
                    if i < len(df)-2:
                        neighbors.append(df_filled.at[i+2, col])
                    neighbors = [n for n in neighbors if not pd.isna(n)]
                    if neighbors:
                        df_filled.at[i, col] = np.mean(neighbors)
    
    for col in (categorical_columns or []) + (binary_columns or []):
        for i in df.index:
            if pd.isna(df_filled.at[i, col]):
                neighbors = []
                if i > 1 and not pd.isna(df_filled.at[i-2, col]):
                    neighbors.append(df_filled.at[i-2, col])
                if i > 0 and not pd.isna(df_filled.at[i-1, col]):
                    neighbors.append(df_filled.at[i-1, col])
                if i < len(df)-1 and not pd.isna(df_filled.at[i+1, col]):
                    neighbors.append(df_filled.at[i+1, col])
                if i < len(df)-2 and not pd.isna(df_filled.at[i+2, col]):
                    neighbors.append(df_filled.at[i+2, col])
                if neighbors:
                    df_filled.at[i, col] = pd.Series(neighbors).mode()[0]
    
    return df_filled

def remove_leading_trailing_zeros(df, column_name):
    while len(df) > 0 and df[column_name].iloc[0] == 0:
        df = df.iloc[1:]
    while len(df) > 0 and df[column_name].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df

def replace_middle_zeros(df, column_name):
    values = df[column_name].values
    for i in range(1, len(values) - 1):
        if values[i] == 0:
            prev_non_zero = next((values[j] for j in range(i-1, -1, -1) if values[j] != 0), None)
            next_non_zero = next((values[j] for j in range(i+1, len(values)) if values[j] != 0), None)
            if prev_non_zero is not None and next_non_zero is not None:
                values[i] = (prev_non_zero + next_non_zero) / 2
    df[column_name] = values
    return df

def pre_process(df, numerical_columns, categorical_columns=None, binary_columns=None, datetime_column=None, target_column=None):
    if datetime_column:
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.sort_values(by=datetime_column)

    df = replace_rare_categories(df, categorical_columns)
    df = one_hot_encode(df, categorical_columns)
    df = fill_missing_values(df, numerical_columns, categorical_columns, binary_columns)
    
    if target_column:
        df = remove_leading_trailing_zeros(df, target_column)
        df = replace_middle_zeros(df, target_column)
    
    df = df.sort_values(by=datetime_column)
    
    return df


def add_cos_transformations(df, col, period):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    df.drop(columns=[col], inplace=True)
    return df

def get_datetime_features(df, datetime_column, cos_sin = False):
    # Ensure Datetime is in datetime format
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    # Extracting date-time features
    df['Hour'] = df[datetime_column].dt.hour
    df['DayOfWeek'] = df[datetime_column].dt.dayofweek
    df['DayOfMonth'] = df[datetime_column].dt.day
    df['Month'] = df[datetime_column].dt.month
    df['DayOfYear'] = df[datetime_column].dt.dayofyear

    # Binary representation for weekends
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    if cos_sin:
        # Apply cosine transformations for periodic features
        df = add_cos_transformations(df, 'Hour', 24)
        df = add_cos_transformations(df, 'DayOfWeek', 7)
        df = add_cos_transformations(df, 'DayOfMonth', 31)  # Approximate number of days in a month
        df = add_cos_transformations(df, 'Month', 12)
        df = add_cos_transformations(df, 'DayOfYear', 365)

    # Dropping Datetime column if not needed anymore
    df.set_index(datetime_column, inplace=True)

    return df


# Function to create sequences
def seq_data(df, input_seq_length=48, output_seq_length=24, target='Energy'):
    X, y = [], []
    max_idx = len(df) - input_seq_length - output_seq_length + 1
    for i in range(max_idx):
        X.append(df.iloc[i:i+input_seq_length].values)
        y.append(df.iloc[i+input_seq_length:i+input_seq_length+output_seq_length][target].values)
    return np.array(X), np.array(y).reshape(max_idx, output_seq_length, 1)

from sklearn.preprocessing import MinMaxScaler

# Function to load and preprocess data with sequences
def load_and_preprocess_data_with_sequences(df, target='Energy', scaled=False, scale_type='features', val_ratio=0.1, test_ratio=0.1, input_seq_length=48, output_seq_length=24):
    """
    Parameters:
    - df: DataFrame containing the data
    - target: Target variable for prediction
    - scaled: Boolean flag to apply scaling
    - scale_type: Specify scaling for 'features', 'output', or 'both'
    - val_ratio: Ratio of the dataset to be used for validation
    - test_ratio: Ratio of the dataset to be used for testing
    - input_seq_length: Number of time steps in the input sequences
    - output_seq_length: Number of time steps in the output sequences
    
    Returns:
    - X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled
    """

    # Drop NA values or consider imputation
    df.dropna(inplace=True)

    # Calculate the number of rows to lose at the beginning and end
    rows_to_lose = input_seq_length + output_seq_length - 1
    effective_rows = len(df) - rows_to_lose

    # Calculate sizes for test and validation
    test_size = int(effective_rows * test_ratio)
    val_size = int(effective_rows * val_ratio)
    train_size = effective_rows - test_size - val_size

    # Check for edge cases
    if test_size == 0 or val_size == 0 or train_size <= 0:
        raise ValueError("Dataset is too small to be split into train, validation, and test sets with the given ratios.")

    if scaled:
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        if scale_type == 'features' or scale_type == 'both':
            # Fit and transform the features scaler on the training data
            feature_columns = df.columns[df.columns != target]
            feature_scaler.fit(df[feature_columns].iloc[:train_size + input_seq_length + output_seq_length - 1])
            df[feature_columns] = feature_scaler.transform(df[feature_columns])
        
        if scale_type == 'output' or scale_type == 'both':
            # Fit and transform the output scaler on the training data
            output_scaler.fit(df[[target]].iloc[:train_size + input_seq_length + output_seq_length - 1])
            df[[target]] = output_scaler.transform(df[[target]])

        df_scaled = df
    else:
        df_scaled = df

    # Generate sequences
    X, y = seq_data(df_scaled, input_seq_length, output_seq_length, target)

    # Split sequences into training, validation, and test sets
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test, X, y, df_scaled
