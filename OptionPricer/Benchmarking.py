import time
import numpy as np
import pandas as pd
from BlackScholesPricer import BlackScholesPricer
from MonteCarloPricer import MonteCarloPricer
from BinomialTreePricer import BinomialTreePricer
from RiskFreeCurve import RiskFreeCurve

class OptionBenchmarker:
    def __init__(self):
        self.curve = RiskFreeCurve()
        self.curve.fetch_data()
        self.nn_results = pd.read_csv("nn_benchmark_results.csv")
        self.nn_results.columns = self.nn_results.columns.str.strip()

    def test_all_methods_vs_bs(self):
        n_tests = len(self.nn_results)
        results = []

        for i in range(n_tests):
            S = self.nn_results.loc[i, "S"]
            K = self.nn_results.loc[i, "K"]
            T = self.nn_results.loc[i, "T"]
            sigma = self.nn_results.loc[i, "sigma"]
            bs_price = self.nn_results.loc[i, "bs_price"]
            nn_price = self.nn_results.loc[i, "nn_price"]
            nn_time = self.nn_results.loc[i, "nn_time"]
            nn_char = self.nn_results.loc[i, "Characteristics"]

            r = self.curve.get_zero_rate(T)

            # Black-Scholes
            results.append({
                "Method": "Black-Scholes",
                "Price": bs_price,
                "Time": 0.0,
                "Error": 0.0,
                "Details": "Closed-form"
            })

            # Monte Carlo - Serial
            start = time.time()
            mc = MonteCarloPricer(S, K, T, r, sigma, option_type="call", q=0.0)
            mc_price_serial, _ = mc.price(N=10000)
            mc_time_serial = (time.time() - start) * 1000
            results.append({
                "Method": "Monte Carlo (Serial, 10k)",
                "Price": mc_price_serial,
                "Time": mc_time_serial,
                "Error": abs(mc_price_serial - bs_price),
                "Details": "Serial | 10,000 simulations"
            })

            # Monte Carlo - Parallel
            start = time.time()
            mc_price_parallel, _, _ = mc.benchmark(N=10000, n_jobs=4)
            mc_time_parallel = (time.time() - start) * 1000
            results.append({
                "Method": "Monte Carlo (Parallel, 10k)",
                "Price": mc_price_parallel,
                "Time": mc_time_parallel,
                "Error": abs(mc_price_parallel - bs_price),
                "Details": "Parallel | 10,000 simulations"
            })

            # Binomial Tree - Vectorized
            start = time.time()
            bt = BinomialTreePricer(S, K, T, r, sigma, N=100, option_type="call", dividend_type="continuous", q=0.0)
            bt_price_vec, _ = bt.vectorized_price()
            bt_time_vec = (time.time() - start) * 1000
            results.append({
                "Method": "Binomial Tree (Vectorized)",
                "Price": bt_price_vec,
                "Time": bt_time_vec,
                "Error": abs(bt_price_vec - bs_price),
                "Details": "Vectorized | N=100"
            })

            # Binomial Tree - Recursive
            bt_rec = BinomialTreePricer(S, K, T, r, sigma, N=20, option_type="call", dividend_type="continuous", q=0.0)
            start = time.time()
            bt_price_r = bt_rec.recursive_price()
            bt_time_r = (time.time() - start) * 1000
            results.append({
                "Method": "Binomial Tree (Recursive)",
                "Price": bt_price_r,
                "Time": bt_time_r,
                "Error": abs(bt_price_r - bs_price),
                "Details": "Recursive | N=20"
            })

            # Neural Network
            results.append({
                "Method": "Neural Network",
                "Price": nn_price,
                "Time": nn_time,
                "Error": abs(nn_price - bs_price),
                "Details": nn_char
            })

        df = pd.DataFrame(results)
        summary = df.groupby("Method")[["Price", "Time", "Error"]].mean().round(4)

        print("\n--- Benchmark on 200 Random Samples (vs BS) ---")
        print(summary)
        print("\n--- Details (first 5 rows) ---")
        print(df.head())

if __name__ == "__main__":
    benchmark = OptionBenchmarker()
    benchmark.test_all_methods_vs_bs()