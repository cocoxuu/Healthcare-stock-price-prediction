# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import json
from market_analysis import add_technical_indicators

def prepare_training_data(df, forecast_period=5):
    """
    Prepare data for model training, generally from the prospect of the whole healthcare sector.
    
    X variables include:
    - Basic price data (open, high, low, close, volume)
    - Technical indicator(avg_sma_50, avg_ma_20, 
                        avg_bb_upper, avg_bb_lower, 
                        avg_rsi, avg_macd, avg_macd_signal) 
                            
    Y variable is:
    - Future Closing Price (5 days)

    """
    df = add_technical_indicators(df)
    
    # set dependent variables
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                   'Avg_SMA_50', 'Avg_MA_20', 'Avg_BB_Upper', 'Avg_BB_Lower',
                   'Avg_RSI', 'Avg_MACD', 'Avg_MACD_Signal'] 

    # check any col actually exists
    available_features = [col for col in feature_cols if col in df.columns]
    print("\nAvailable features:", available_features)

    features = df[available_features].copy()
    features = features.dropna()
    
    # create independent variable
    # forcast period = 5 days
    target = features['Close'].shift(-forecast_period)
    features = features[:-forecast_period]
    target = target[:-forecast_period]
    
    # create and fit a MinMaxScaler
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

    print("\nFinal features set:", list(features.columns))
    print(f"\nNumber of samples: {len(features)}")
    print(f"\nForecast period: {forecast_period} days" )

    target_scaler.forecast_period = forecast_period
    
    return features, target, target_scaled, target_scaler

def save_training_results(results, model, scaler, base_path='saved_models'):
    """
    Save training results, model and scaler
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    # save model and scaler
    model.save(f'{base_path}/lstm_model.h5') 
    with open(f'{base_path}/scaler.pkl', 'wb') as f: 
        pickle.dump(scaler, f)
        
    # save other results
    serializable_results = {
        'metrics': results['metrics'],
        'predictions': results['predictions'].tolist(),
        'true_values': results['true_values'].tolist(),
        'history': {
            'loss': results['history'].history['loss'],
            'val_loss': results['history'].history['val_loss']
        }
    }
    
    with open(f'{base_path}/results.json', 'w') as f:
        json.dump(serializable_results, f)  

def load_training_results(base_path='saved_models'):
    """
    Load saved training results, model and scaler
    This function is generated with the help of ChatGPT-4o.
    """
    try:
        # load model, scaler and other results
        model = load_model(f'{base_path}/lstm_model.h5')
        with open(f'{base_path}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{base_path}/results.json', 'r') as f:
            results = json.load(f)
            
        # convert lists back to numpy arrays
        results['predictions'] = np.array(results['predictions'])
        results['true_values'] = np.array(results['true_values'])
        
        return True, {'model': model, 'scaler': scaler, **results}
    
    except:
        return False, None


def train_lstm(features, target_scaled, target_scaler, force_retrain=False, timesteps=10, epochs=50):
    """
    Train an LSTM model and visualize the results.
    """
    # get forecast period from scaler
    forecast_period = getattr(target_scaler, 'forecast_period', 5) # default is 5
    
    if not force_retrain:
        success, saved_results = load_training_results() # try to load saved results first
        if success:
            return saved_results
    
    # scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # prepare serial data
    num_samples = scaled_features.shape[0] - timesteps + 1
    X = np.zeros((num_samples, timesteps, scaled_features.shape[1]))
    y = np.zeros((num_samples, 1))

    # create time series samples
    for i in range(num_samples):
        X[i] = scaled_features[i:i+timesteps]
        y[i] = target_scaled[i+timesteps-1]
        
    # split training & testing set
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # construct the LSTM model
    model = Sequential([
        Input(shape=(timesteps, features.shape[1])),
        LSTM(20, activation='tanh', return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        LSTM(32, activation='tanh',
             kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='relu')
    ])
    
    model.compile(optimizer='adam', loss='mse') 
    
    # add k-fold validification
    cv_scores = []
    kfold = TimeSeriesSplit(n_splits=5)

    # storing the best validation results
    best_history = None
    best_val_loss = float('inf')
    
    for train_index, val_index in kfold.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
        history = model.fit(
            X_fold_train, y_fold_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_fold_val, y_fold_val),  
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
        )
    

        val_loss = min(history.history['val_loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_history = history
        
        score = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        cv_scores.append(score)

    print(f"Mean CV Score: {np.mean(cv_scores):.4f}")

    # make predictions
    predictions = model.predict(X_test)
    true_prices = target_scaler.inverse_transform(y_test)  # inverse transform scaled values
    predicted_prices = target_scaler.inverse_transform(predictions)
    
    print("\nPrice Ranges:")
    print(f"Predicted: {float(predicted_prices.min()):.2f} - {float(predicted_prices.max()):.2f}")
    print(f"True: {float(true_prices.min()):.2f} - {float(true_prices.max()):.2f}")
    print(f"\nForecast Period: {forecast_period} days into the future.")

    # calculate performance metrics
    mse = mean_squared_error(true_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_prices, predicted_prices)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'forecast_period': forecast_period
    }
    
    return {
        'model':model,
        'predictions': predicted_prices,
        'true_values': true_prices,
        'metrics':metrics,
        'history': best_history,
        'feature_scaler': scaler,
        'target_scaler': target_scaler,
        'test_data':X_test,
    }
