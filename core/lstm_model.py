# core/lstm_model.py
# LSTM Ensemble Predictor

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
import ta
import logging

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM disabled.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

class UniversalLSTMPredictor:
    FEATURE_COLS = [
        "open", "high", "low", "close", "volume",
        "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "atr", "adx", "obv", "returns", "volatility"
    ]
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5, model_path: str = None):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_path = Path(model_path) if model_path else Path(os.getcwd()) / "models_new"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.threshold = float(os.getenv("LSTM_THRESHOLD_PERCENT", 0.1)) / 100
        self.model = None
        self.scaler = None
        self.is_trained = False
        logger.info(f"LSTM Predictor | Seq: {sequence_length} | Path: {self.model_path}")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20, min_periods=1).std().fillna(0)
        df["rsi"] = ta.momentum.rsi(df["close"], window=14).fillna(50)
        
        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd().fillna(0)
        df["macd_signal"] = macd.macd_signal().fillna(0)
        df["macd_hist"] = macd.macd_diff().fillna(0)
        
        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband().fillna(df["close"])
        df["bb_middle"] = bb.bollinger_mavg().fillna(df["close"])
        df["bb_lower"] = bb.bollinger_lband().fillna(df["close"])
        
        df["atr"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"]).fillna(0)
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"]).fillna(25)
        
        if "volume" in df.columns and df["volume"].sum() > 0:
            df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"]).fillna(0)
        else:
            df["obv"] = 0.0
        
        for col in self.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        
        df.dropna(inplace=True)
        return df

    def _create_sequences(self, data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i:(i + self.sequence_length)])
            future_price = data[i + self.sequence_length + self.prediction_horizon - 1, 3]
            current_price = data[i + self.sequence_length - 1, 3]
            future_return = (future_price - current_price) / current_price if current_price > 0 else 0
            
            if future_return > threshold: label = 1
            elif future_return < -threshold: label = 2
            else: label = 0
            
            y.append(label)
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow not available")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(), Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(), Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(), Dropout(0.2),
            Dense(3, activation="softmax")
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train(self, df: pd.DataFrame, pair: str, epochs: int = 25) -> Dict:
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available", "accuracy": 0.0}
        
        logger.info(f"Training {pair} model...")
        train_threshold = max(self.threshold, 0.005) if any(x in pair.upper() for x in ["BTC", "ETH"]) else self.threshold
        
        df_features = self._engineer_features(df)
        if len(df_features) < self.sequence_length + 50:
            return {"error": f"Insufficient data: {len(df_features)} rows", "accuracy": 0.0}
        
        feature_data = df_features[self.FEATURE_COLS].values
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(feature_data)
        
        X, y = self._create_sequences(scaled_data, train_threshold)
        if len(X) < 50:
            return {"error": f"Insufficient sequences: {len(X)}", "accuracy": 0.0}
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=min(64, len(X_train)),
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        clean_pair = pair.replace("/", "").upper()
        model_file = self.model_path / f"{clean_pair}_lstm.keras"
        scaler_file = self.model_path / f"{clean_pair}_scaler.pkl"
        
        self.model.save(str(model_file))
        joblib.dump(self.scaler, str(scaler_file))
        
        self.is_trained = True
        accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else 0.0
        
        logger.info(f"{pair} model trained | Accuracy: {accuracy:.2%}")
        return {"accuracy": accuracy}

    def load_model(self, pair: str) -> bool:
        if not TF_AVAILABLE:
            return False
        
        if self.is_trained and self.model is not None:
            return True
        
        clean_pair = pair.replace("/", "").upper()
        model_file = self.model_path / f"{clean_pair}_lstm.keras"
        scaler_file = self.model_path / f"{clean_pair}_scaler.pkl"
        
        if not model_file.exists():
            model_file = self.model_path / f"{clean_pair}_lstm_best.h5"
        
        if model_file.exists() and scaler_file.exists():
            try:
                self.model = load_model(str(model_file), compile=False)
                self.scaler = joblib.load(str(scaler_file))
                self.is_trained = True
                logger.info(f"Loaded model for {pair}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model for {pair}: {e}")
                return False
        
        logger.warning(f"No model found for {pair}")
        return False

    def predict(self, df: pd.DataFrame, pair: str = "ETHUSD") -> Dict:
        if not TF_AVAILABLE or not self.is_trained or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "Model not available"}
        
        try:
            df_features = self._engineer_features(df)
            if len(df_features) < self.sequence_length:
                return {"signal": "HOLD", "confidence": 0.0, "reason": f"Need {self.sequence_length} rows, got {len(df_features)}"}
            
            last_data = df_features[self.FEATURE_COLS].tail(self.sequence_length)
            X_scaled = self.scaler.transform(last_data)
            X_pred = np.array([X_scaled])
            
            pred = self.model.predict(X_pred, verbose=0)[0]
            signal_idx = np.argmax(pred)
            confidence = float(pred[signal_idx])
            signal = ["HOLD", "BUY", "SELL"][signal_idx]
            
            return {
                "signal": signal,
                "confidence": confidence,
                "raw_probs": {"HOLD": float(pred[0]), "BUY": float(pred[1]), "SELL": float(pred[2])}
            }
            
        except Exception as e:
            logger.error(f"Prediction error for {pair}: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "reason": str(e)}

class EnsemblePredictor:
    def __init__(self):
        self.lstm = UniversalLSTMPredictor()
        self.weights = {"lstm": 0.4, "technical": 0.35, "fundamental": 0.25}
    
    def predict(self, df: pd.DataFrame, tech_score: float, fund_score: float, pair: str) -> Dict:
        lstm_pred = self.lstm.predict(df, pair=pair)
        
        signal_map = {"SELL": -1, "HOLD": 0, "BUY": 1}
        lstm_score = signal_map.get(lstm_pred["signal"], 0) * lstm_pred["confidence"]
        
        ensemble_score = (
            lstm_score * self.weights["lstm"] +
            tech_score * self.weights["technical"] +
            fund_score * self.weights["fundamental"]
        )
        
        # CHANGED: Threshold lowered to 0.08 for increased sensitivity
        if ensemble_score > 0.08: final_signal = "BUY"
        elif ensemble_score < -0.08: final_signal = "SELL"
        else: final_signal = "HOLD"
        
        final_confidence = min(max(lstm_pred["confidence"], abs(ensemble_score) * 2), 1.0)
        
        return {
            "signal": final_signal,
            "confidence": float(final_confidence),
            "ensemble_score": float(ensemble_score),
            "lstm": lstm_pred,
            "technical_score": float(tech_score)
        }