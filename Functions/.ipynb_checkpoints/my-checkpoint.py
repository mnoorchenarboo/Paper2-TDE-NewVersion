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


def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None)))

def calculate_smape(y_true, y_pred):
    denominator = np.abs(y_true) + np.abs(y_pred)
    return 100 * np.mean(2.0 * np.abs(y_true - y_pred) / np.clip(denominator, 1e-8, None))

def implement_tcn(X_train, y_train, X_val, y_val, X_test, y_test,
                  num_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8],
                  dropout_rate=0.2, learning_rate=1e-3, epochs=20, batch_size=32):
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




def implement_tcn_with_attention(X_train, y_train, X_val, y_val, X_test, y_test,
                                 num_filters=64, kernel_size=3, dilation_rates=[1, 2, 4, 8],
                                 dropout_rate=0.3, num_heads=2, key_dim=32, 
                                 learning_rate=1e-3, epochs=20, batch_size=32):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, Dropout, Add, Dense, LayerNormalization, Reshape
    from tensorflow.keras.models import Model

    input_shape = X_train.shape[1:]
    target_length = y_train.shape[1]

    # Start timer
    start_time = time.time()

    # Define TCN block
    def tcn_block(input_layer, filters, kernel_size, dilation_rate, dropout_rate):
        x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", activation="relu")(input_layer)
        x = Dropout(dropout_rate)(x)
        x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal", activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        # Ensure skip connection has the same shape
        skip_connection = Conv1D(filters, 1, padding="same")(input_layer)
        return Add()([x, skip_connection])

    # Build TCN model
    inputs = Input(shape=input_shape)
    x = inputs

    for dilation_rate in dilation_rates:
        x = tcn_block(x, num_filters, kernel_size, dilation_rate, dropout_rate)

    # Project TCN output to match key_dim for attention
    x = Dense(key_dim)(x)

    # Add Multi-Head Attention
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    attn_output, attn_scores = attention_layer(x, x, return_attention_scores=True)  # Self-attention

    # Residual connection and normalization
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    # Reduce to target sequence length
    x = Conv1D(target_length, 1, activation="relu")(x)
    x = Reshape((target_length, -1))(x)
    outputs = Dense(y_train.shape[-1])(x)
    model = Model(inputs, outputs)

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

    # Extract attention weights for explainability
    attention_extractor = Model(inputs=model.input, outputs=attn_scores)
    attention_scores = attention_extractor.predict(X_test)

    return model, metrics, attention_scores, history


import matplotlib.pyplot as plt


def plot_loss_curves(history_tcn, history_tcn_with_attention):
    # Extract loss and validation loss
    tcn_loss = history_tcn.history['loss']
    tcn_val_loss = history_tcn.history['val_loss']
    tcn_with_attention_loss = history_tcn_with_attention.history['loss']
    tcn_with_attention_val_loss = history_tcn_with_attention.history['val_loss']

    # Plot
    plt.figure(figsize=(10, 6))

    # TCN Loss
    plt.plot(tcn_loss, label='TCN Training Loss', linestyle='-', marker='o')
    plt.plot(tcn_val_loss, label='TCN Validation Loss', linestyle='--', marker='x')

    # TCN with Attention Loss
    plt.plot(tcn_with_attention_loss, label='TCN+Attention Training Loss', linestyle='-', marker='s')
    plt.plot(tcn_with_attention_val_loss, label='TCN+Attention Validation Loss', linestyle='--', marker='d')

    # Labels and Title
    plt.title("Training and Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()