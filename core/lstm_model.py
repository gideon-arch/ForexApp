# core/lstm_model.py
# LSTM with REALISTIC 0.1% threshold (Dynamic Environment Mapping)

import os
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
import ta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Clean top-level Keras imports to clear VS Code linter errors
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, BatchNormalization, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam


class UniversalLSTMPredictor:
    """Universal LSTM Predictor optimized for:
    - ETH/USD & BTC/USD (Crypto Streams)
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        model_path: str = None,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # FIXED: Use pathlib to step out of core/ and anchor directly to the project root models/ directory
        if model_path is None:
            current_dir = Path(__file__).resolve().parent
            self.model_path = current_dir.parent / "models"
        else:
            self.model_path = Path(model_path)
            
        # Ensure the physical model directory exists on the host
        os.makedirs(str(self.model_path), exist_ok=True)

        # LINKED TO ENVIRONMENT: Defaults to 0.1% if threshold variable is missing
        self.threshold = float(os.getenv("LSTM_THRESHOLD_PERCENT", 0.1)) / 100

        self.model = None
        self.scaler = None
        self.is_trained = False

        self.feature_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "adx",
            "obv",
            "returns",
            "volatility",
        ]

        print(
            f"🧠 LSTM Universal Engine Initialized | Absolute Save Path: {self.model_path} | Sensitivity Threshold Target: {self.threshold * 100}%"
        )

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=20).std()

        df["rsi"] = ta.momentum.rsi(df["close"], window=14)

        macd = ta.trend.MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        bb = ta.volatility.BollingerBands(df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()

        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"]
        )
        df["adx"] = ta.trend.adx(df["high"], df["low"], df["close"])

        if "volume" in df.columns:
            df["obv"] = ta.volume.on_balance_volume(df["close"], df["volume"])
        else:
            df["obv"] = 0

        df.dropna(inplace=True)
        return df

    def _create_sequences(
        self, data: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []

        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            X.append(data[i : (i + self.sequence_length)])

            future_prices = data[
                i
                + self.sequence_length : i
                + self.sequence_length
                + self.prediction_horizon,
                3,  # Index 3 corresponds to the scaled 'close' price
            ]
            current_price = data[i + self.sequence_length - 1, 3]

            # Avoid division by zero anomalies
            if current_price == 0:
                future_return = 0
            else:
                future_return = (
                    future_prices[-1] - current_price
                ) / current_price

            if future_return > threshold:
                label = 1  # Buy
            elif future_return < -threshold:
                label = 2  # Sell
            else:
                label = 0  # Hold

            y.append(label)

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=input_shape),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(64, return_sequences=False),
                BatchNormalization(),
                Dropout(0.3),
                Dense(32, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(3, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train(self, df: pd.DataFrame, pair: str, epochs: int = 35) -> Dict:
        train_threshold = self.threshold
        if "BTC" in pair.upper() or "ETH" in pair.upper():
            train_threshold = max(
                self.threshold, 0.005
            )  # 0.5% standard for cryptos

        print(
            f"🧠 Training Universal LSTM for {pair} (Aligned Threshold: {train_threshold * 100}%)..."
        )

        df_features = self._engineer_features(df)
        feature_data = df_features[self.feature_cols].values

        # Core sequential chronological partition split to block data leaks
        split_idx = int(len(feature_data) * 0.8)
        train_features = feature_data[:split_idx]
        test_features = feature_data[split_idx:]

        # Fit scale baseline *only* on training blocks
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = self.scaler.fit_transform(train_features)
        scaled_test = self.scaler.transform(test_features)

        # Re-pool structural matrix
        scaled_data = np.vstack((scaled_train, scaled_test))

        X, y = self._create_sequences(scaled_data, train_threshold)

        # Splitting indices manually to mirror validation sequences cleanly
        X_train = X[: split_idx - self.sequence_length]
        y_train = y[: split_idx - self.sequence_length]
        X_test = X[split_idx - self.sequence_length :]
        y_test = y[split_idx - self.sequence_length :]

        if len(X_train) < 100:
            raise ValueError(
                f"Need 100+ training sequences, got {len(X_train)}"
            )

        self.model = self.build_model((X.shape[1], X.shape[2]))

        clean_pair = pair.replace("/", "")
        # CHANGED: Reverted backup checkpoint strategy to use high-compression .h5 file footprints
        h5_save_path = self.model_path / f"{clean_pair}_lstm_best.h5"
        scaler_save_path = self.model_path / f"{clean_pair}_scaler.pkl"
        
        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            ),
            ModelCheckpoint(
                str(h5_save_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=0,
            ),
        ]

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        joblib.dump(self.scaler, str(scaler_save_path))

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.is_trained = True

        print(
            f"✅ Training complete for {pair} | Test Accuracy: {accuracy:.2%}"
        )

        return {
            "pair": pair,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "epochs": len(history.history["loss"]),
            "threshold": train_threshold,
        }

    def load_model(self, pair: str) -> bool:
        clean_pair = pair.replace("/", "")
        
        # CHANGED: Standardized cross-platform lookups targeting raw .h5 extensions
        model_upper = self.model_path / f"{clean_pair.upper()}_lstm_best.h5"
        model_lower = self.model_path / f"{clean_pair.lower()}_lstm_best.h5"
        
        scaler_upper = self.model_path / f"{clean_pair.upper()}_scaler.pkl"
        scaler_lower = self.model_path / f"{clean_pair.lower()}_scaler.pkl"

        # Dynamically determine casing styles present in repository layout
        if model_upper.exists() and scaler_upper.exists():
            model_file, scaler_file = model_upper, scaler_upper
        elif model_lower.exists() and scaler_lower.exists():
            model_file, scaler_file = model_lower, scaler_lower
        else:
            print(
                f"❌ Missing files for {pair}. Looked for absolute variants inside directory tree:\n"
                f"  - {model_upper}\n"
                f"  - {model_lower}"
            )
            return False

        try:
            # Added compile=False to bypass cross-platform container validation quirks on Linux host
            self.model = load_model(str(model_file), compile=False)
            
            # Recompile structural layers manually to prepare tracking states safely
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            
            self.scaler = joblib.load(str(scaler_file))
            self.is_trained = True
            print(f"✅ Successfully loaded model and pre-fitted scaler pipeline from: {model_file}")
            return True
        except Exception as e:
            print(f"⚠️ Error loading saved model files for {pair}: {e}")
            return False

    def predict(self, df: pd.DataFrame, pair: str = "ETHUSD") -> Dict:
        # Check training status or try loading the model directly
        if not self.is_trained or self.scaler is None:
            if not self.load_model(pair):
                raise FileNotFoundError(
                    f"❌ Model state or scaler pipeline for {pair} is unavailable, and training is blocked on production servers."
                )

        # Pass complete un-sliced raw dataframe stream to calculate technical indices safely
        df_features = self._engineer_features(df)

        if len(df_features) < self.sequence_length:
            raise ValueError(
                f"Need {self.sequence_length} valid indicators candles, got {len(df_features)}. Feed a larger dataframe buffer history window."
            )

        # Extract sequence slice safely after the math transforms have resolved
        last_data = (
            df_features[self.feature_cols].tail(self.sequence_length).values
        )
        scaled_sequence = self.scaler.transform(last_data)
        X_pred = np.array([scaled_sequence])

        prediction = self.model.predict(X_pred, verbose=0)[0]

        classes = ["HOLD", "BUY", "SELL"]
        predicted_class = np.argmax(prediction)
        confidence = float(prediction[predicted_class])

        current_price = df["close"].iloc[-1]
        volatile = df_features["atr"].iloc[-1]

        asset_threshold = self.threshold
        if "BTC" in pair.upper() or "ETH" in pair.upper():
            asset_threshold = max(self.threshold, 0.005)

        if predicted_class == 1:  # Buy
            target = current_price * (1 + asset_threshold)
            stop_loss = current_price - (volatile * 1.5)
        elif predicted_class == 2:  # Sell
            target = current_price * (1 - asset_threshold)
            stop_loss = current_price + (volatile * 1.5)
        else:
            target = current_price
            stop_loss = current_price

        return {
            "signal": classes[predicted_class],
            "confidence": confidence,
            "probabilities": {
                "hold": float(prediction[0]),
                "buy": float(prediction[1]),
                "sell": float(prediction[2]),
            },
            "current_price": float(current_price),
            "predicted_target": float(target),
            "suggested_stop_loss": float(stop_loss),
            "risk_reward_ratio": abs(target - current_price) / abs(stop_loss - current_price) if stop_loss != current_price else 0,
            "threshold": asset_threshold,
        }


class EnsemblePredictor:

    def __init__(self):
        self.lstm = UniversalLSTMPredictor()
        self.weights = {"lstm": 0.4, "technical": 0.35, "fundamental": 0.25}

    def predict(
        self, df: pd.DataFrame, tech_score: float, fund_score: float, pair: str
    ) -> Dict:
        # CRITICAL FIX: Kill the fallback server training loop entirely
        if not self.lstm.is_trained:
            if not self.lstm.load_model(pair):
                raise FileNotFoundError(
                    f"❌ Environment Halt: Pre-trained binaries for {pair} not found in 'models/'. "
                    f"Execution stopped to prevent a 512MB Render RAM crash."
                )

        lstm_pred = self.lstm.predict(df, pair=pair)

        signal_map = {"SELL": -1, "HOLD": 0, "BUY": 1}
        lstm_score = signal_map[lstm_pred["signal"]] * lstm_pred["confidence"]

        combined_score = (
            lstm_score * self.weights["lstm"]
            + tech_score * self.weights["technical"]
            + fund_score * self.weights["fundamental"]
        )

        threshold = float(os.getenv("ENSEMBLE_THRESHOLD_PERCENT", 0.1)) / 100

        if combined_score > threshold:
            final_signal = "BUY"
        elif combined_score < -threshold:
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        composite_confidence = (
            lstm_pred["confidence"] * self.weights["lstm"]
            + abs(tech_score) * self.weights["technical"]
            + abs(fund_score) * self.weights["fundamental"]
        )

        return {
            "signal": final_signal,
            "confidence": float(composite_confidence),
            "combined_score": float(combined_score),
            "threshold": threshold,
            "component_scores": {
                "lstm": float(lstm_score),
                "technical": float(tech_score),
                "fundamental": float(fund_score),
            },
            "lstm_details": lstm_pred,
            "weights_used": self.weights,
        }