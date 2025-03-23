#  Intelligent Option Pricer for S&P 500 Index

This project implements a robust and modular pricing engine for European options on indices such as the S&P 500. It combines classic pricing models (Black-Scholes, Monte Carlo, Binomial Tree), modern machine learning approximation, real market data retrieval, and benchmarking — all coded from scratch in Python.

##  Objectives

- Implement pricing models without using specialized financial libraries
- Retrieve real market data (spot, volatility, yield curve)
- Train a neural network to approximate option prices
- Compare pricing methods in terms of speed and accuracy
- Ensure modular, efficient, and well-documented code structure

---

##  Project Structure

```bash
.
├── BlackScholesPricer.py         # Analytical Black-Scholes model
├── BinomialTreePricer.py        # Binomial Tree (CRR) method
├── MonteCarloPricer.py          # Monte Carlo method with optimizations
├── NeuralNetworkPricer.py       # ML-based pricer using Keras
├── RiskFreeCurve.py             # Yield curve from market data (FRED)
├── VolatilitySurface.py         # Implied volatility surface with bicubic interpolation
├── Benchmarking.py              # Benchmarks all pricing methods
├── nn_model.keras               # Pre-trained neural network model
├── nn_benchmark_results.csv     # Benchmark results of the ML model
├── requirements.txt             # All required Python dependencies
├── .gitignore                   # Ignore unnecessary files in Git
└── Intelligent_Option_Pricer.pdf # Detailed report
```

---

##  Pricers Implemented

### 1. Black-Scholes (Closed-Form)
- From-scratch implementation
- Handles constant dividend yield (e.g., 1.15% for S&P 500)
- Vectorized for fast computation

### 2. Monte Carlo Simulation
- Lazy path generation via generators
- Antithetic variates for variance reduction
- Parallel computation using `concurrent.futures`
- Delta and Vega via finite differences

### 3. Binomial Tree (CRR)
- Recursive and vectorized implementations
- Supports continuous or discrete dividends
- Efficient backward induction algorithm

### 4. Neural Network Approximation
- MLP with ReLU, Dropout, BatchNorm
- Trained on 100,000 synthetic Black-Scholes prices
- Inputs: (S/K, σ, T, r)
- Output: log(option price)
- Trained model stored as `nn_model.keras`
- Benchmark outputs in `nn_benchmark_results.csv`

---

##  Market Data Integration

### Spot Price & Volatility
- Retrieved from Yahoo Finance (SPY ETF)
- `VolatilitySurface.py` builds the implied volatility surface
- Bicubic interpolation using `RectBivariateSpline`

### Yield Curve
- Data from FRED (US Treasury zero-coupon rates)
- Cubic spline interpolation
- Discounting curve constructed with `QuantLib`

---

##  Benchmarking

Run all pricers on a set of 200 test options and compare:

| Method                  | Price     | Time (ms) | Error (%) |
|-------------------------|-----------|-----------|-----------|
| Black-Scholes           | 61.12     | 0.00      | 0.00      |
| Binomial Tree (Vector)  | 61.13     | 0.97      | 0.01      |
| Monte Carlo (Serial)    | 61.20     | 55.77     | 0.13      |
| Monte Carlo (Parallel)  | 61.10     | 341.47    | 0.03      |
| Binomial Tree (Rec.)    | 61.25     | 1.98      | 0.22      |
| Neural Network          | 61.89     | 235.00    | 1.25      |

---

##  Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/intelligent-option-pricer.git
cd intelligent-option-pricer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run benchmark (no need to retrain the neural network)

```bash
python Benchmarking.py
```

### 4. Optional: Retrain Neural Network ( Slow)

```bash
python NeuralNetworkPricer.py  # Regenerates model + benchmark CSV
```

---

##  Notes

- All pricing models were implemented from scratch (no QuantLib pricing)
- The `nn_model.keras` and `nn_benchmark_results.csv` are provided to avoid re-training
- No need for Docker or virtual machines — everything runs locally with Python 3.8+
- Risk-free rate and volatility surfaces are interpolated from real market data

---

##  Report

For detailed methodology, design choices, equations, and critiques, see:

 `Intelligent_Option_Pricer.pdf`

---

##  Author

**Amina Alami**  
M2 El Karoui (École Polytechnique / Sorbonne)  
March 2025

