from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import sqlite3
import numpy as np


def remove_leading_trailing_zeros(df, column_name):
    """
    Removes leading and trailing rows where the specified column contains
    either 0 or NaN. Keeps the rest of the data intact to preserve time continuity.
    """
    series = df[column_name]
    valid_mask = ~(series.isna() | (series == 0))

    if not valid_mask.any():
        return df.iloc[0:0]  # Return empty DataFrame if all are invalid

    first_valid = valid_mask.idxmax()
    last_valid = valid_mask[::-1].idxmax()

    return df.loc[first_valid:last_valid]


def clean_missing_values(df):
    threshold = 0.2  # 20% threshold for missing values

    for column in df.columns:
        missing_ratio = df[column].isna().mean()

        if missing_ratio > threshold:
            print(f"Dropping column '{column}' with {missing_ratio*100:.2f}% missing values.")
            df.drop(columns=[column], inplace=True)
        # else:
            # print(f"Filling missing values in column '{column}' with {missing_ratio*100:.2f}% missing values.")

            def fill_with_neighbors_avg(series):
                filled_series = series.copy()

                for i, value in enumerate(series):
                    if pd.isna(value):
                        neighbors = pd.concat([
                            series[max(i - 5, 0):i].dropna(),
                            series[i + 1:i + 6].dropna()
                        ])
                        if len(neighbors) > 0:
                            filled_series[i] = neighbors.mean()
                return filled_series

            df[column] = fill_with_neighbors_avg(df[column])

    return df


def get_datetime_features(df, cos_sin=False):
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    df['Month'] = df.index.month
    df['DayOfYear'] = df.index.dayofyear
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

    if cos_sin:
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)

    return df


def print_primary_use_summary(db_path):
    conn = sqlite3.connect(db_path)

    try:
        bdg_df = pd.read_sql("SELECT * FROM BDG2_electricity", conn)
    except:
        bdg_df = pd.DataFrame()

    try:
        lh_df = pd.read_sql("SELECT * FROM London_Hydro", conn)
    except:
        lh_df = pd.DataFrame()

    conn.close()

    summary = {}

    if not bdg_df.empty:
        for col in bdg_df.columns:
            parts = col.split("_")
            if len(parts) == 3:
                primary = parts[1]
                summary.setdefault(primary, []).append(col)

    if not lh_df.empty and "dataset" in lh_df.columns:
        summary.setdefault("residential", lh_df["dataset"].unique())

    print("Primary Use Summary:")
    for primary_use, items in summary.items():
        print(f" - {primary_use}: {len(items)} options")


def load_and_preprocess_from_sqlite(db_path, primary_use, option_number=0):
    conn = sqlite3.connect(db_path)

    if primary_use.lower() == "residential":
        df = pd.read_sql("SELECT * FROM London_Hydro", conn)
        conn.close()

        unique_datasets = sorted(df["dataset"].unique())

        if option_number >= len(unique_datasets):
            print(f"Option number {option_number} out of range. Available: 0 to {len(unique_datasets) - 1}")
            return None

        selected_dataset = unique_datasets[option_number]
        df = df[df["dataset"] == selected_dataset].copy()
        df.drop(columns=["dataset"], inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        df = remove_leading_trailing_zeros(df, "energy_consumption")
        df = get_datetime_features(df)
        df = clean_missing_values(df)

        time_features = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'DayOfYear', 'IsWeekend']
        weather_features = ['temperature', 'humidity', 'wind_speed']
        target = ['energy_consumption']

        ordered_cols = [col for col in time_features + weather_features + target if col in df.columns]
        df = df[ordered_cols]

        summary = f"Source: London_Hydro | Dataset: {selected_dataset} | Primary Use: residential"
        return df, summary

    else:
        df = pd.read_sql("SELECT * FROM BDG2_electricity", conn)
        weather_df = pd.read_sql("SELECT * FROM BDG2_weather", conn)
        conn.close()

        columns = [col for col in df.columns if "_" in col and col.split("_")[1] == primary_use]

        if not columns:
            print(f"No columns found for primary use '{primary_use}'")
            return None

        if option_number >= len(columns):
            print(f"Option number {option_number} out of range. Available: 0 to {len(columns) - 1}")
            return None

        col_name = columns[option_number]
        site_id, _, building = col_name.split("_")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        energy_df = df[[col_name]].copy().rename(columns={col_name: "energy_consumption"})
        energy_df = remove_leading_trailing_zeros(energy_df, "energy_consumption")
        energy_df["energy_consumption"] = energy_df["energy_consumption"].ffill().bfill()
        energy_df = get_datetime_features(energy_df)

        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        site_weather = weather_df[weather_df["site_id"] == site_id].copy()
        site_weather.set_index("timestamp", inplace=True)

        weather_features = ['airTemperature', 'dewTemperature', 'seaLvlPressure', 'windSpeed']
        available_weather = [col for col in weather_features if col in site_weather.columns]
        site_weather = site_weather[available_weather]

        for col in site_weather.columns:
            site_weather = remove_leading_trailing_zeros(site_weather, col)

        site_weather = site_weather.ffill().bfill()

        merged = pd.merge(energy_df, site_weather, left_index=True, right_index=True, how="inner")
        cleaned = clean_missing_values(merged)

        time_features = ['Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'DayOfYear', 'IsWeekend']
        target = ['energy_consumption']
        ordered_cols = [col for col in time_features + weather_features + target if col in cleaned.columns]
        cleaned = cleaned[ordered_cols]

        summary = f"Source: BDG2 | Primary Use: {primary_use}, Building: {building}, SiteID: {site_id}"
        return cleaned, summary



def seq_data(df, input_seq_length=48, output_seq_length=24, target='energy_consumption'):
    X, y = [], []
    max_idx = len(df) - input_seq_length - output_seq_length + 1
    for i in range(max_idx):
        X.append(df.iloc[i:i+input_seq_length].values)
        y.append(df.iloc[i+input_seq_length:i+input_seq_length+output_seq_length][target].values)
    return np.array(X), np.array(y).reshape(max_idx, output_seq_length, 1)


class DataContainer:
    """
    A container class to store and access training, validation, testing splits, and full DataFrame.
    """
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, X, y, original_data, scaled_data, data_type, more_info, feature_names):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y
        self.original_data = original_data
        self.scaled_data = scaled_data
        self.data_type = data_type
        self.more_info = more_info
        self.feature_names = feature_names


def load_and_preprocess_data_with_sequences(
    db_path,
    primary_use,
    option_number=0,
    target='energy_consumption',
    scaled=False,
    scale_type='features',
    val_ratio=0.1,
    test_ratio=0.1,
    input_seq_length=48,
    output_seq_length=24
):
    """
    Prepares time-series data for training, validation, and testing using sliding window.
    Returns sequences and scaled DataFrame.
    """
    
    df_original, data_info = load_and_preprocess_from_sqlite(db_path=db_path, primary_use=primary_use, option_number=option_number)
    
    df = df_original.copy()
    df.dropna(inplace=True)

    rows_to_lose = input_seq_length + output_seq_length - 1
    effective_rows = len(df) - rows_to_lose

    test_size = int(effective_rows * test_ratio)
    val_size = int(effective_rows * val_ratio)
    train_size = effective_rows - test_size - val_size

    if test_size == 0 or val_size == 0 or train_size <= 0:
        raise ValueError("Dataset is too small to split into train/val/test with given ratios.")

    if scaled:
        feature_scaler = MinMaxScaler()
        output_scaler = MinMaxScaler()

        feature_columns = [col for col in df.columns if col != target]

        scaler_end_idx = train_size + input_seq_length + output_seq_length - 1
        training_slice = df.iloc[:scaler_end_idx]

        if scale_type in ['features', 'both']:
            feature_scaler.fit(training_slice[feature_columns])
            df[feature_columns] = feature_scaler.transform(df[feature_columns])

        if scale_type in ['output', 'both']:
            output_scaler.fit(training_slice[[target]])
            df[[target]] = output_scaler.transform(df[[target]])

        df_scaled = df
    else:
        df_scaled = df

    # Generate sequences
    X, y = seq_data(df_scaled, input_seq_length, output_seq_length, target)

    # Final slicing
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Optional shape print
    print("\nShapes:")
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Wrap in container
    return DataContainer(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        X=X,
        y=y,
        original_data=df_original,
        scaled_data=df_scaled,
        data_type=primary_use,
        more_info=data_info,
        feature_names=df.columns.tolist()
    )