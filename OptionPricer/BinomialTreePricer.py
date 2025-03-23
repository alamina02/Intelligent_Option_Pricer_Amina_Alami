import numpy as np
import time

class BinomialTreePricer:
    """
    Cox-Ross-Rubinstein Binomial Tree pricer for European options.

    Supports:
    - Recursive and vectorized implementations
    - Continuous or discrete dividends
    - Comparison of accuracy and speed

    Parameters:
    ----------
    S0 : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility (annualized)
    N : int
        Number of time steps in the tree
    option_type : str
        'call' or 'put'
    dividend_type : str
        'continuous' or 'discrete'
    q : float or list
        If continuous: dividend yield (float), if discrete: list of (time, amount)
    """

    def __init__(self, S0, K, T, r, sigma, N=100, option_type="call", dividend_type="continuous", q=0.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.dividend_type = dividend_type
        self.q = q

    def _apply_dividends(self, S, t):
        """
        Adjust spot price based on discrete dividends.
        """
        if self.dividend_type == "discrete" and isinstance(self.q, list):
            for (t_div, amount) in self.q:
                if t >= t_div:
                    S -= amount
        return S

    def vectorized_price(self):
        """
        Vectorized implementation of the CRR binomial tree.

        Returns:
            option_price : float
            computation_time : float
        """
        start = time.time()

        dt = self.T / self.N
        q = self.q if self.dividend_type == "continuous" else 0.0

        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp((self.r - q) * dt)
        p = (a - d) / (u - d)

        S = np.array([self.S0 * (u**j) * (d**(self.N - j)) for j in range(self.N + 1)])
        if self.dividend_type == "discrete":
            S = np.array([self._apply_dividends(s, self.T) for s in S])

        if self.option_type == "call":
            V = np.maximum(S - self.K, 0)
        else:
            V = np.maximum(self.K - S, 0)

        disc = np.exp(-self.r * dt)
        for i in range(self.N - 1, -1, -1):
            V = disc * (p * V[1:] + (1 - p) * V[:-1])

        end = time.time()
        return V[0], end - start

    def recursive_price(self):
        """
        Recursive implementation of the CRR binomial tree (slow, for illustration only).
        """
        dt = self.T / self.N
        q = self.q if self.dividend_type == "continuous" else 0.0
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp((self.r - q) * dt)
        p = (a - d) / (u - d)
        disc = np.exp(-self.r * dt)

        def value(i, j):
            # i: time step, j: number of up moves
            S = self.S0 * (u ** j) * (d ** (i - j))
            if self.dividend_type == "discrete":
                S = self._apply_dividends(S, i * dt)

            if i == self.N:
                return max(S - self.K, 0) if self.option_type == "call" else max(self.K - S, 0)
            else:
                return disc * (p * value(i + 1, j + 1) + (1 - p) * value(i + 1, j))

        return value(0, 0)

    def compare_methods(self):
        """
        Compare recursive and vectorized methods in terms of accuracy and speed.
        """
        vec_price, vec_time = self.vectorized_price()
        start = time.time()
        rec_price = self.recursive_price()
        rec_time = time.time() - start

        print("Comparison of Binomial Tree Methods")
        print("----------------------------------")
        print(f"Vectorized Price: {vec_price:.4f}, Time: {vec_time:.4f} s")
        print(f"Recursive  Price: {rec_price:.4f}, Time: {rec_time:.4f} s")
        print(f"Absolute Error: {abs(vec_price - rec_price):.6f}")

# Example usage
if __name__ == "__main__":
    # Test with continuous dividend
    pricer = BinomialTreePricer(S0=100, K=100, T=1, r=0.1, sigma=0.2, N=20, option_type="call", dividend_type="continuous", q=0.015)
    pricer.compare_methods()

    # Test with discrete dividend
    discrete_divs = [(0.5, 2.0), (0.75, 1.5)]  # (time, amount)
    pricer2 = BinomialTreePricer(S0=100, K=100, T=1, r=0.1, sigma=0.2, N=20, option_type="call", dividend_type="discrete", q=discrete_divs)
    price, time_spent = pricer2.vectorized_price()
    print(f"\nPrice with discrete dividends: {price:.4f}, Time: {time_spent:.4f} s")
    
    #Test only with Vectorized price to compare with BlackScholes and MonteCarlo
    pricer_vec = BinomialTreePricer(S0=100, K=100, T=1, r=0.1, sigma=0.2, N=100, option_type="call", dividend_type="continuous", q=0)
    price_vec, _ = pricer_vec.vectorized_price()
    print(f"Vectorized Price: {price_vec:.4f}")