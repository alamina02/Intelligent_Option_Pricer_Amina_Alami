import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

class MonteCarloPricer:
    """
    Monte Carlo pricer for European options using Black-Scholes dynamics.

    Features:
    - Lazy generation via generator
    - Antithetic variates for variance reduction
    - Random matrix for approximating Greeks (delta, vega)
    - Parallelized pricing with multiprocessing
    - Benchmarking of computation time

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
    option_type : str
        'call' or 'put'
    q : float
        Dividend yield (default = 0.0)
    """

    def __init__(self, S0, K, T, r, sigma, option_type="call", q=0.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.q = q

    def _payoff(self, ST):
        """Compute the option payoff for final price ST."""
        if self.option_type == "call":
            return np.maximum(ST - self.K, 0)
        else:
            return np.maximum(self.K - ST, 0)

    def _generate_paths(self, N):
        """
        Lazy generator yielding antithetic variates for Z ~ N(0,1).
        Each iteration yields a pair: (Z, -Z)
        """
        for _ in range(N // 2):
            Z = np.random.randn()
            yield Z
            yield -Z

    def price(self, N=100_000):
        """
        Serial Monte Carlo pricing using antithetic variables.

        Returns:
            price : float
            std_error : float
        """
        dt = self.T
        disc = np.exp(-self.r * self.T)
        payoffs = []

        for Z in self._generate_paths(N):
            ST = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)
            payoffs.append(self._payoff(ST))

        payoffs = np.array(payoffs)
        price = disc * np.mean(payoffs)
        std_error = disc * np.std(payoffs, ddof=1) / np.sqrt(N)
        return price, std_error

    def _single_worker(self, seed, chunk):
        """
        Worker function for parallel pricing. Used in multiprocessing.
        """
        np.random.seed(seed)
        samples = []
        for Z in np.random.randn(chunk // 2):
            for z in [Z, -Z]:  # antithetic variates
                ST = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * z)
                samples.append(self._payoff(ST))
        return samples

    def price_parallel(self, N=1_000_000, n_jobs=4):
        """
        Parallel Monte Carlo pricing using multiprocessing.

        Returns:
            price : float
            std_error : float
        """
        chunks = [N // n_jobs] * n_jobs
        seeds = np.random.randint(0, 1e6, size=n_jobs)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = executor.map(self._single_worker, seeds, chunks)

        all_payoffs = np.concatenate(list(results))
        price = np.exp(-self.r * self.T) * np.mean(all_payoffs)
        std_error = np.exp(-self.r * self.T) * np.std(all_payoffs, ddof=1) / np.sqrt(N)
        return price, std_error

    def benchmark(self, N=1_000_000, n_jobs=4):
        """
        Run parallel pricing and measure computation time.

        Returns:
            price : float
            std_error : float
            time_elapsed : float (in seconds)
        """
        start = time.time()
        price, error = self.price_parallel(N, n_jobs)
        end = time.time()
        return price, error, end - start

    def greeks(self, N=100_000, h=1e-2):
        """
        Approximate Delta and Vega using finite differences and common random numbers.

        Parameters:
            N : int
                Number of Monte Carlo paths
            h : float
                Perturbation step size

        Returns:
            delta : float
            vega : float
        """
        np.random.seed(42)
        Zs = list(self._generate_paths(N))

        # Delta
        S_up = self.S0 + h
        S_down = self.S0 - h

        ST_up = S_up * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * np.array(Zs))
        ST_down = S_down * np.exp((self.r - self.q - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * np.array(Zs))

        payoff_up = self._payoff(ST_up)
        payoff_down = self._payoff(ST_down)

        delta = np.exp(-self.r * self.T) * (np.mean(payoff_up - payoff_down)) / (2 * h)

        # Vega
        sigma_up = self.sigma + h
        sigma_down = self.sigma - h

        ST_up = self.S0 * np.exp((self.r - self.q - 0.5 * sigma_up**2) * self.T + sigma_up * np.sqrt(self.T) * np.array(Zs))
        ST_down = self.S0 * np.exp((self.r - self.q - 0.5 * sigma_down**2) * self.T + sigma_down * np.sqrt(self.T) * np.array(Zs))

        payoff_up = self._payoff(ST_up)
        payoff_down = self._payoff(ST_down)

        vega = np.exp(-self.r * self.T) * (np.mean(payoff_up - payoff_down)) / (2 * h)

        return delta, vega

# Example usage
if __name__ == "__main__":
    pricer = MonteCarloPricer(S0=100, K=100, T=1, r=0.1, sigma=0.2, option_type="call")

    # Serial pricing
    price, err = pricer.price(N=100_000)
    print(f"[Serial] Price: {price:.4f}, Std Error: {err:.4f}")

    # Parallel pricing
    price_p, err_p, t = pricer.benchmark(N=1_000_000, n_jobs=4)
    print(f"[Parallel] Price: {price_p:.4f}, Std Error: {err_p:.4f}, Time: {t:.2f} sec")

    # Greeks
    delta, vega = pricer.greeks(N=100_000)
    print(f"Delta ≈ {delta:.4f}, Vega ≈ {vega:.4f}")