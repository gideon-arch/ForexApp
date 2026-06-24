# core/lstm_model.py
# LSTM Model for Crypto Price Prediction - FIXED

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple
import joblib

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM predictions disabled.")

class LSTMPredictor:
    def __init__(self, sequence_length: int = 60, n_features: int = 10):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler() if TF_AVAILABLE else None
        self.model_path = os.path.join("models", "lstm")
        os.makedirs(self.model_path, exist_ok=True)

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features in exact same order for train AND predict"""
        features = pd.DataFrame(index=df.index)
        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df.get('volume', 0)
        features['rsi'] = df.get('rsi', 50.0)
        features['macd'] = df.get('macd', 0.0)
        features['macd_signal'] = df.get('macd_signal', 0.0)
        features['sma_20'] = df.get('sma_20', df['close'])
        features['sma_50'] = df.get('sma_50', df['close'])
        features['atr'] = df.get('atr', df['close'] * 0.01)
        features['bb_position'] = df.get('bb_position', 0.5)
        
        # Forward fill then backward fill, then fill remaining with 0
        features = features.ffill().bfill().fillna(0)
        return features

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def build_model(self) -> Sequential:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae']
        )
        return model

    def train(self, df: pd.DataFrame, pair: str, epochs: int = 50, batch_size: int = 32) -> Dict:
        if not TF_AVAILABLE:
            return {"status": "error", "message": "TensorFlow not available"}
        
        features_df = self._prepare_features(df)
        data = features_df.values
        
        if data.shape[1] != self.n_features:
            logger.error(f"Feature mismatch: got {data.shape[1]} features, expected {self.n_features}")
            return {"status": "error", "message": f"Feature mismatch: {data.shape[1]} vs {self.n_features}"}
        
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = self._create_sequences(scaled_data)
        if len(X) < 100:
            return {"status": "error", "message": f"Not enough sequences: {len(X)}"}
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model = self.build_model()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        predictions = self.model.predict(X_test, verbose=0)
        mae = np.mean(np.abs(predictions.flatten() - y_test))
        
        model_file = os.path.join(self.model_path, f"{pair}_lstm.h5")
        self.model.save(model_file)
        scaler_file = os.path.join(self.model_path, f"{pair}_scaler.pkl")
        joblib.dump(self.scaler, scaler_file)
        
        return {
            "status": "success",
            "accuracy": float(1 - mae),
            "epochs_trained": len(history.history['loss']),
            "final_loss": float(history.history['loss'][-1]),
            "model_path": model_file
        }

    def load_model(self, pair: str) -> bool:
        if not TF_AVAILABLE:
            return False
        
        model_file = os.path.join(self.model_path, f"{pair}_lstm.h5")
        scaler_file = os.path.join(self.model_path, f"{pair}_scaler.pkl")
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            return False
        
        try:
            self.model = load_model(model_file)
            self.scaler = joblib.load(scaler_file)
            logger.info(f"Model loaded for {pair}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model for {pair}: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Dict:
        if not TF_AVAILABLE or self.model is None:
            return {"signal": "HOLD", "confidence": 0.0, "predicted_price": 0.0}
        
        try:
            features_df = self._prepare_features(df)
            data = features_df.values
            
            if data.shape[1] != self.n_features:
                logger.error(f"Feature mismatch in predict: got {data.shape[1]} features, expected {self.n_features}")
                return {"signal": "HOLD", "confidence": 0.0, "predicted_price": 0.0}
            
            # Check we have enough data
            if len(data) < self.sequence_length:
                logger.warning(f"Not enough data for prediction: {len(data)} < {self.sequence_length}")
                return {"signal": "HOLD", "confidence": 0.0, "predicted_price": 0.0}
            
            scaled_data = self.scaler.transform(data)
            
            last_sequence = scaled_data[-self.sequence_length:]
            last_sequence = np.reshape(last_sequence, (1, self.sequence_length, self.n_features))
            
            predicted_scaled = self.model.predict(last_sequence, verbose=0)[0][0]
            
            # Inverse transform the prediction
            dummy = np.zeros((1, self.n_features))
            dummy[0, 0] = predicted_scaled
            predicted_price = self.scaler.inverse_transform(dummy)[0, 0]
            
            current_price = df['close'].iloc[-1]
            price_change = (predicted_price - current_price) / current_price if current_price > 0 else 0
            
            threshold = float(os.getenv('LSTM_THRESHOLD_PERCENT', '0.05')) / 100
            
            if price_change > threshold:
                signal = "BUY"
                confidence = min(abs(price_change) / (threshold * 3), 0.99)
            elif price_change < -threshold:
                signal = "SELL"
                confidence = min(abs(price_change) / (threshold * 3), 0.99)
            else:
                signal = "HOLD"
                confidence = 0.0
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "predicted_price": float(predicted_price),
                "current_price": float(current_price),
                "price_change_pct": float(price_change * 100)
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {"signal": "HOLD", "confidence": 0.0, "predicted_price": 0.0}

class EnsemblePredictor:
    def __init__(self):
        self.lstm = LSTMPredictor()
        self.technical_weight = 0.4
        self.lstm_weight = 0.6

    def predict(self, df: pd.DataFrame, tech_score: float = 0.0, fund_score: float = 0.0, pair: str = "") -> Dict:
        lstm_result = self.lstm.predict(df)
        
        lstm_signal = lstm_result.get("signal", "HOLD")
        lstm_confidence = lstm_result.get("confidence", 0.0)
        
        if lstm_signal == "HOLD" or lstm_confidence < 0.1:
            return {"signal": "HOLD", "confidence": 0.0, "reason": "LSTM uncertain"}
        
        combined_confidence = (lstm_confidence * self.lstm_weight) + (abs(tech_score) * self.technical_weight)
        combined_confidence = min(combined_confidence, 0.99)
        
        if tech_score > 0.3 and lstm_signal == "BUY":
            final_signal = "BUY"
        elif tech_score < -0.3 and lstm_signal == "SELL":
            final_signal = "SELL"
        elif abs(tech_score) < 0.2:
            final_signal = lstm_signal
        else:
            final_signal = "HOLD"
            combined_confidence = 0.0
        
        return {
            "signal": final_signal,
            "confidence": float(combined_confidence),
            "lstm_signal": lstm_signal,
            "lstm_confidence": lstm_confidence,
            "tech_score": tech_score,
            "reason": f"LSTM:{lstm_signal}({lstm_confidence:.2f}) + Tech:{tech_score:.2f}"
        }