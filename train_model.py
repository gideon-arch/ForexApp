import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# 1. Setup
os.makedirs('models_new', exist_ok=True)
pairs = ['ETHUSD', 'BTCUSD']

# Match the 17 features used in lstm_model.py
feature_cols = ["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower", "atr", "adx", "obv", "returns", "volatility"]

def train_and_save():
    for pair in pairs:
        print(f"--- Training {pair} ---")
        
        # 2. Mock Data (Replacing 1 column with 17 columns to match your production logic)
        num_rows = 1000
        data = pd.DataFrame(np.random.rand(num_rows, len(feature_cols)), columns=feature_cols)
        
        # 3. Scaling (Fitted on ALL 17 columns)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 4. Prepare X and y (Keeping 60 sequence length)
        X, y = [], []
        seq_len = 60
        for i in range(seq_len, len(scaled_data)):
            X.append(scaled_data[i-seq_len:i, :]) # Taking all columns
            y.append(np.random.randint(0, 3)) # Mock labels for classification
        X, y = np.array(X), np.array(y)

        # 5. Build and Train (Updated for 17 features)
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            BatchNormalization(), Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(), Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
        model.fit(X, y, epochs=5, batch_size=32, verbose=1)

        # 6. Save
        model_path = f'models_new/{pair}_lstm_best.h5'
        scaler_path = f'models_new/{pair}_scaler.pkl'
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Success! {pair} saved with 17-feature configuration.")

if __name__ == "__main__":
    train_and_save()