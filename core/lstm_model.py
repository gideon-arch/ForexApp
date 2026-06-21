# core/lstm_model.py
import os
import shutil
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, BatchNormalization, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam

class UniversalLSTMPredictor:
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5, model_path: str = None):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # PRODUCTION FIX: Use current working directory to locate the folder
        if model_path is None:
            self.model_path = Path(os.getcwd()) / "models_new"
        else:
            self.model_path = Path(model_path)
            
        os.makedirs(self.model_path, exist_ok=True)
        self.threshold = float(os.getenv("LSTM_THRESHOLD_PERCENT", 0.1)) / 100
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_cols = ["open", "high", "low", "close", "volume", "rsi", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_middle", "bb_lower", "atr", "adx", "obv", "returns", "volatility"]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()
        df["rsi"] = ta.momentum.rsi(df["close"], window=14)
        macd = ta.trend.MACD(df["close"])
        df["macd"], df["macd_signal"], df["macd_hist"] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"])
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])
        df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"]) if "volume" in df.columns else 0
        return df.dropna()

    def _create_sequences(self, data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i : (i + self.sequence_length)])
            future_return = (data[i + self.sequence_length + self.prediction_horizon - 1, 3] - data[i + self.sequence_length - 1, 3]) / data[i + self.sequence_length - 1, 3]
            label = 1 if future_return > threshold else (2 if future_return < -threshold else 0)
            y.append(label)
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(), Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(), Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(), Dropout(0.2),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, df: pd.DataFrame, pair: str, epochs: int = 35) -> Dict:
        train_threshold = max(self.threshold, 0.005) if "BTC" in pair.upper() or "ETH" in pair.upper() else self.threshold
        df_features = self._engineer_features(df)
        feature_data = df_features[self.feature_cols].values
        split_idx = int(len(feature_data) * 0.8)
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit with feature names by using the DataFrame directly
        scaled_data = self.scaler.fit_transform(df_features[self.feature_cols])
        X, y = self._create_sequences(scaled_data, train_threshold)
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        clean_pair = pair.replace("/", "")
        h5_path = self.model_path / f"{clean_pair}_lstm_best.h5"
        scaler_path = self.model_path / f"{clean_pair}_scaler.pkl"

        history = self.model.fit(X[:split_idx], y[:split_idx], epochs=epochs, batch_size=64, validation_data=(X[split_idx:], y[split_idx:]), verbose=1)
        
        self.model.save(str(h5_path))
        joblib.dump(self.scaler, str(scaler_path))
        
        self.is_trained = True
        return {"accuracy": float(history.history['accuracy'][-1])}

    def load_model(self, pair: str) -> bool:
        clean_pair = pair.replace("/", "").upper()
        h5_file = self.model_path / f"{clean_pair}_lstm_best.h5"
        pkl_file = self.model_path / f"{clean_pair}_scaler.pkl"
        
        if h5_file.exists() and pkl_file.exists():
            self.model = load_model(str(h5_file), compile=False)
            self.scaler = joblib.load(str(pkl_file))
            self.is_trained = True
            return True
        return False

    def predict(self, df: pd.DataFrame, pair: str = "ETHUSD") -> Dict:
        if not self.load_model(pair):
            raise FileNotFoundError(f"No valid pre-trained model found at {self.model_path}")
        
        df_features = self._engineer_features(df)
        # Keep as DataFrame to preserve feature names for the scaler
        last_data = df_features[self.feature_cols].tail(self.sequence_length)
        
        # Transform using the scaler (now with matching feature names)
        X_scaled = self.scaler.transform(last_data)
        X_pred = np.array([X_scaled])
        
        pred = self.model.predict(X_pred, verbose=0)[0]
        return {"signal": ["HOLD", "BUY", "SELL"][np.argmax(pred)], "confidence": float(np.max(pred))}

class EnsemblePredictor:
    def __init__(self):
        self.lstm = UniversalLSTMPredictor()
        self.weights = {"lstm": 0.4, "technical": 0.35, "fundamental": 0.25}

    def predict(self, df: pd.DataFrame, tech_score: float, fund_score: float, pair: str) -> Dict:
        lstm_pred = self.lstm.predict(df, pair=pair)
        score = ({"SELL": -1, "HOLD": 0, "BUY": 1}[lstm_pred["signal"]] * lstm_pred["confidence"] * self.weights["lstm"]) + (tech_score * self.weights["technical"]) + (fund_score * self.weights["fundamental"])
        final_signal = "BUY" if score > 0.1 else ("SELL" if score < -0.1 else "HOLD")
        return {"signal": final_signal, "confidence": float(lstm_pred["confidence"])}