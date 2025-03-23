import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import QuantLib as ql
import yfinance as yf

class RiskFreeCurve:
    """
    Construct a risk-free zero-coupon yield curve from US Treasury yields
    using cubic spline interpolation and QuantLib discounting curve.

    Data source: Yahoo Finance (via yfinance)
    """

    def __init__(self):
        self.maturities = None   # in years
        self.rates = None        # zero-coupon rates
        self.curve = None        # CubicSpline object
        self.ql_curve = None     # QuantLib curve (ZeroCurve)

    def fetch_data(self):
        """
        Fetch treasury rates using Yahoo Finance tickers:
        ^IRX (13W), ^FVX (5Y), ^TNX (10Y), ^TYX (30Y)
        """
        tickers = {
            '^IRX': 0.25,
            '^FVX': 5,
            '^TNX': 10,
            '^TYX': 30
        }

        maturities = []
        rates = []

        for ticker, mat in tickers.items():
            data = yf.Ticker(ticker)
            hist = data.history(period="1d")
            if not hist.empty:
                last_yield = hist['Close'].iloc[-1] / 100  # yields are in percentage points
                maturities.append(mat)
                rates.append(last_yield)

        self.maturities = np.array(maturities)
        self.rates = np.array(rates)
        self.curve = CubicSpline(self.maturities, self.rates)
        self.build_quantlib_curve()

    def build_quantlib_curve(self):
        """
        Build a QuantLib zero rate curve from the fetched data.
        """
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        dates = [today + ql.Period(int(m * 12 + 0.5), ql.Months) for m in self.maturities]
        self.ql_curve = ql.YieldTermStructureHandle(
            ql.ZeroCurve(dates, self.rates, ql.Actual365Fixed())
        )

    def get_zero_rate(self, T):
        if self.curve is None:
            raise ValueError("Risk-free curve not initialized. Call fetch_data() first.")
        return float(self.curve(T))

    def get_discount_factor(self, T):
        if self.ql_curve is None:
            raise ValueError("QuantLib curve not built. Call fetch_data() first.")
        return self.ql_curve.discount(T)

    def plot(self):
        if self.curve is None:
            raise ValueError("Risk-free curve not initialized. Call fetch_data() first.")
        T_grid = np.linspace(min(self.maturities), max(self.maturities), 300)
        plt.figure(figsize=(8, 4))
        plt.plot(T_grid, self.curve(T_grid), label="Interpolated Curve")
        plt.scatter(self.maturities, self.rates, color='red', label="Input Points")
        plt.title("US Treasury Zero-Coupon Yield Curve (Yahoo Finance)")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Zero Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    curve = RiskFreeCurve()
    curve.fetch_data()
    print(f"Zero rate for 5Y: {curve.get_zero_rate(5):.4%}")
    print(f"Discount factor for 5Y: {curve.get_discount_factor(5):.6f}")
    curve.plot()