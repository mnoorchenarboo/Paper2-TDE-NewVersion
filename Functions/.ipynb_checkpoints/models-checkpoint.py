from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dropout, Add, Dense, LayerNormalization, Reshape
from tensorflow.keras.models import Model


def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))


def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2.0 * np.abs(y_true - y_pred) / np.clip(denominator, 1e-8, None))


def implement_tcn(data,
                  num_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8],
                  dropout_rate=0.2, learning_rate=1e-3, epochs=20, batch_size=32):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    input_shape = X_train.shape[1:]
    target_length = y_train.shape[1]

    # Start timer
    start_time = time.time()

    # Define TCN block
    def tcn_block(input_layer, filters, kernel_size, dilation_rate, dropout_rate):
        x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", activation="relu")(
            input_layer)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        skip_connection = layers.Conv1D(filters, 1, padding="same")(input_layer)  # Skip connection
        return layers.Add()([x, skip_connection])

    # Build TCN model
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for dilation_rate in dilation_rates:
        x = tcn_block(x, num_filters, kernel_size, dilation_rate, dropout_rate)
    x = layers.Conv1D(target_length, 1, activation="relu")(x)  # Reduce to target sequence length
    x = layers.Reshape((target_length, -1))(x)
    outputs = layers.Dense(y_train.shape[-1])(x)
    model = models.Model(inputs, outputs)

    # Compile the model with a specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate and compute metrics
    y_pred = model.predict(X_test)
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)
    mape_value = calculate_mape(y_test_flat, y_pred_flat)
    smape_value = calculate_smape(y_test_flat, y_pred_flat)

    # End timer
    end_time = time.time()
    elapsed_time = end_time - start_time

    metrics = {
        "MSE": mse,
        "MAE": mae,
        "MAPE": mape_value,
        "SMAPE": smape_value,
        "R2": r2,
        "Runtime (seconds)": elapsed_time
    }

    return model, metrics, history
