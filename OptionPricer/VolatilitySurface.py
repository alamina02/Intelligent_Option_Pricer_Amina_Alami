import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import yfinance as yf
from datetime import datetime

class VolatilitySurface:
    """
    Constructs and interpolates a volatility surface using bicubic interpolation.
    Based on live implied volatilities from SPY options via Yahoo Finance.
    Automatically adapts the spline order if there are not enough data points.
    """

    def __init__(self, strikes, maturities, vol_matrix, spot):
        self.strikes = strikes
        self.maturities = maturities
        self.vol_matrix = vol_matrix
        self.spot = spot  # Save the spot price (S0) for use in pricing models

        # Automatically adjust interpolation order based on data size
        kx = min(3, len(strikes) - 1)
        ky = min(3, len(maturities) - 1)
        if kx < 1 or ky < 1:
            raise ValueError("Not enough data points for interpolation.")

        self.spline = RectBivariateSpline(maturities, strikes, vol_matrix, kx=kx, ky=ky)

    def get_vol(self, T, K):
        """Interpolate implied volatility for a given maturity T and strike K."""
        return float(self.spline(T, K))

    def get_spot(self):
        """Return the spot price (S0) used to build the surface."""
        return self.spot

    def plot_surface(self):
        """Plot the implied volatility surface in 3D."""
        K_grid, T_grid = np.meshgrid(self.strikes, self.maturities)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(K_grid, T_grid, self.vol_matrix, cmap='viridis', edgecolor='k', alpha=0.9)
        ax.set_title("Implied Volatility Surface")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity (Years)")
        ax.set_zlabel("Implied Volatility")
        plt.tight_layout()
        plt.show()

# Example usage with market data from Yahoo Finance
if __name__ == "__main__":
    # Fetch SPY spot price (ETF tracking S&P 500)
    spy = yf.Ticker("SPY")
    spot = spy.history(period="1d")['Close'].iloc[-1]
    print(f"Spot price (SPY): {spot:.2f}")

    # Select option expiries and build strike grid
    expiries = spy.options[:6]  # take up to 6 maturities
    strikes = np.round(np.linspace(spot * 0.95, spot * 1.05, 7), 2)  # 7 strikes around spot

    maturities = []
    vol_matrix = []

    for expiry in expiries:
        try:
            chain = spy.option_chain(expiry)
            calls = chain.calls
        except:
            continue  # skip bad expiry

        vols = []
        for K in strikes:
            # Match strike within a small tolerance (±2)
            match = calls[np.isclose(calls['strike'], K, atol=2)]
            if not match.empty:
                vols.append(match['impliedVolatility'].iloc[0])
            else:
                vols.append(np.nan)

        # Keep only fully valid rows
        if not any(np.isnan(vols)):
            vol_matrix.append(vols)
            dt = (datetime.strptime(expiry, "%Y-%m-%d") - datetime.today()).days / 365
            maturities.append(dt)

    # Build volatility surface if enough valid data
    if len(maturities) >= 2 and len(strikes) >= 4:
        strikes = np.array(strikes)
        maturities = np.array(maturities)
        vol_matrix = np.array(vol_matrix)

        surface = VolatilitySurface(strikes, maturities, vol_matrix, spot)
        T_test, K_test = maturities[1], strikes[len(strikes)//2]
        print(f"Spot: {surface.get_spot():.2f} | Vol(K={K_test:.2f}, T={T_test:.2f}): {surface.get_vol(T_test, K_test):.2%}")
        surface.plot_surface()
    else:
        print("Not enough valid data to build the volatility surface.")
