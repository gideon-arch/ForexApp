import os
import joblib
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Setup
os.makedirs('models_new', exist_ok=True)
pairs = ['ETHUSD', 'BTCUSD']

def train_and_save():
    for pair in pairs:
        print(f"--- Training {pair} ---")
        # 2. Mock Data
        data = np.random.rand(1000, 1) 
        
        # 3. Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # 4. Prepare X and y
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # 5. Build and Train
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=2, batch_size=32, verbose=0)

        # 6. Save
        model_path = f'models_new/{pair}_lstm_best.h5'
        scaler_path = f'models_new/{pair}_scaler.pkl'
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Success! {pair} saved.")

if __name__ == "__main__":
    train_and_save()