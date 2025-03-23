import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
import time
from RiskFreeCurve import RiskFreeCurve
from BlackScholesPricer import BlackScholesPricer
import pandas as pd

class NeuralNetworkPricer:
    def __init__(self, model_path="nn_model.keras"):
        self.model_path = model_path
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.curve = RiskFreeCurve()
        self.curve.fetch_data()
        self.model = self._build_model()

    def _build_model(self):
        input_layer = Input(shape=(4,))  # moneyness, sigma, T, r
        x = Dense(128, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        output_layer = Dense(1, activation='linear')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse', metrics=['mse'])
        return model

    def generate_data(self, n_samples=100000):
        S = np.random.uniform(300, 500, n_samples)
        K = np.random.uniform(300, 500, n_samples)
        T = np.random.uniform(0.05, 2.0, n_samples)
        sigma = np.random.uniform(0.05, 0.5, n_samples)
        r = np.array([self.curve.get_zero_rate(t) for t in T])

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
        call_price = np.clip(call_price, 1e-5, None)

        moneyness = S / K
        X = np.vstack([moneyness, sigma, T, r]).T
        y = np.log(call_price).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = self.x_scaler.fit_transform(X_train)
        X_test = self.x_scaler.transform(X_test)
        y_train = self.y_scaler.fit_transform(y_train)
        y_test = self.y_scaler.transform(y_test)

        return X_train, X_test, y_train, y_test

    def train(self, n_samples=100000):
        X_train, X_test, y_train, y_test = self.generate_data(n_samples)
        start = time.time()
        self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=256,
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
            ]
        )
        train_time = time.time() - start
        preds = self.y_scaler.inverse_transform(self.model.predict(X_test, verbose=0))
        y_true = self.y_scaler.inverse_transform(y_test)
        rmse = np.sqrt(mean_squared_error(np.exp(y_true), np.exp(preds)))
        print(f"[Neural Net] RMSE = {rmse:.4f}")
        print(f"[Neural Net] Training time = {train_time:.2f} seconds")
        save_model(self.model, self.model_path)
        return rmse, train_time

    def predict(self, S, K, sigma, T):
        r = self.curve.get_zero_rate(T)
        moneyness = S / K
        x = np.array([[moneyness, sigma, T, r]])
        x_scaled = self.x_scaler.transform(x)
        y_scaled = self.model.predict(x_scaled, verbose=0)
        return np.exp(self.y_scaler.inverse_transform(y_scaled).item())

if __name__ == "__main__":
    print("--- Training Neural Network (no PCA, with r) ---")
    model = NeuralNetworkPricer()
    rmse, train_time = model.train(n_samples=100000)

    print("\n--- Testing vs Black-Scholes on 200 samples ---")
    n_tests = 200
    errors = []
    bs_prices = []
    nn_prices = []

    S = np.random.uniform(300, 500, n_tests)
    K = np.random.uniform(300, 500, n_tests)
    T = np.random.uniform(0.05, 2.0, n_tests)
    sigma = np.random.uniform(0.05, 0.5, n_tests)
    curve = model.curve

    for i in range(n_tests):
        r = curve.get_zero_rate(T[i])
        bs = BlackScholesPricer(S[i], K[i], T[i], r, sigma[i], option_type="call", q=0.0).price()
        pred = model.predict(S[i], K[i], sigma[i], T[i])

        bs_prices.append(bs)
        nn_prices.append(pred)
        errors.append(abs(pred - bs))

    print(f"Mean BS price: {np.mean(bs_prices):.4f}")
    print(f"Mean NN price: {np.mean(nn_prices):.4f}")
    print(f"Mean abs error: {np.mean(errors):.4f}")

    df_nn = pd.DataFrame({
        "S": S,
        "K": K,
        "T": T,
        "sigma": sigma,
        "bs_price": bs_prices,
        "nn_price": nn_prices,
        "nn_error": errors,
        "nn_time": [train_time] * n_tests,
        "Characteristics": [f"NN | RMSE={rmse:.4f} | TrainTime={train_time:.2f}s"] * n_tests
    })
    df_nn.to_csv("nn_benchmark_results.csv", index=False)