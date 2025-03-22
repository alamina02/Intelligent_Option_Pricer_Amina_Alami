import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    """
    Black-Scholes pricer for European options on indices with a constant dividend yield.

    Parameters:
    ----------
    S0 : float or np.ndarray
        Spot price of the underlying asset
    K : float or np.ndarray
        Strike price of the option
    T : float or np.ndarray
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annualized)
    sigma : float or np.ndarray
        Volatility of the underlying asset (annualized)
    option_type : str
        'call' or 'put'
    q : float, optional
        Constant dividend yield (default is 0.0)
    """

    def __init__(self, S0, K, T, r, sigma, option_type="call", q=0.0):
        self.S0 = np.array(S0, dtype=float)
        self.K = np.array(K, dtype=float)
        self.T = np.array(T, dtype=float)
        self.r = r
        self.sigma = np.array(sigma, dtype=float)
        self.option_type = option_type.lower()
        self.q = q
        self._validate_inputs()

    def _validate_inputs(self):
        assert self.option_type in ["call", "put"], "option_type must be 'call' or 'put'"
        assert np.all(self.S0 > 0) and np.all(self.K > 0), "S0 and K must be positive"
        assert np.all(self.T > 0) and np.all(self.sigma > 0), "T and sigma must be positive"

    def d1(self):
        return (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        d1 = self.d1()
        d2 = self.d2()

        if self.option_type == "call":
            # European call formula
            price = self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            # European put formula
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)

        return price

# Example usage
if __name__ == "__main__":
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.1
    sigma = 0.2

    call = BlackScholesPricer(S0, K, T, r, sigma, option_type="call", q=0.015)
    put = BlackScholesPricer(S0, K, T, r, sigma, option_type="put", q=0.015)

    print(f"Call price: {call.price():.4f}")
    print(f"Put price: {put.price():.4f}")

    # Vectorized example with constant dividend yield
    S0_arr = np.array([90, 100, 110])
    K_arr = np.array([100, 100, 100])
    T_arr = np.array([1, 1, 1])
    sigma_arr = np.array([0.2, 0.2, 0.2])

    pricer_vec = BlackScholesPricer(S0_arr, K_arr, T_arr, r, sigma_arr, option_type="call", q=0.02)
    print("Vectorized call prices with q=2%:", pricer_vec.price())
