import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import QuantLib as ql
import pandas_datareader.data as web

class RiskFreeCurve:
    """
    Constructs a risk-free zero-coupon yield curve from US Treasury yields,
    using FRED data, cubic spline interpolation, and a QuantLib curve.
    """

    def __init__(self):
        self.maturities = None   # in years
        self.rates = None        # zero-coupon rates (in decimals)
        self.curve = None        # CubicSpline object
        self.ql_curve = None     # QuantLib curve (ZeroCurve)

    def fetch_data(self):
        """
        Retrieves US Treasury yields from FRED for an extended range of maturities.
        The FRED tickers used are:
         - 'DGS1MO': 1 month,
         - 'DGS3MO': 3 months,
         - 'DGS6MO': 6 months,
         - 'DGS1'  : 1 year,
         - 'DGS2'  : 2 years,
         - 'DGS3'  : 3 years,
         - 'DGS5'  : 5 years,
         - 'DGS7'  : 7 years,
         - 'DGS10' : 10 years,
         - 'DGS20' : 20 years,
         - 'DGS30' : 30 years.
        """
        start = datetime.datetime(2023, 1, 1)
        end = datetime.datetime.today()

        tickers = {
            'DGS1MO': 1/12,   # 1 month ≈ 0.0833 year
            'DGS3MO': 0.25,   # 3 months
            'DGS6MO': 0.5,    # 6 months
            'DGS1': 1,        # 1 year
            'DGS2': 2,        # 2 years
            'DGS3': 3,        # 3 years
            'DGS5': 5,        # 5 years
            'DGS7': 7,        # 7 years
            'DGS10': 10,      # 10 years
            'DGS20': 20,      # 20 years
            'DGS30': 30       # 30 years
        }

        maturities = []
        rates = []

        for ticker, mat in tickers.items():
            # Retrieve data from FRED
            series = web.DataReader(ticker, 'fred', start, end).dropna()
            # Take the last available value
            last_value = series.iloc[-1, 0]
            maturities.append(mat)
            # Convert percentage to decimal
            rates.append(last_value / 100)

        # Sort the data by maturity to ensure proper interpolation
        sorted_data = sorted(zip(maturities, rates))
        maturities, rates = zip(*sorted_data)

        self.maturities = np.array(maturities)
        self.rates = np.array(rates)
        self.curve = CubicSpline(self.maturities, self.rates)
        self.build_quantlib_curve()

    def build_quantlib_curve(self):
        """
        Builds a QuantLib zero-coupon curve from the retrieved yields.
        """
        today = ql.Date.todaysDate()
        ql.Settings.instance().evaluationDate = today

        # Approximate conversion of maturities to dates
        dates = [today + ql.Period(int(mat * 12 + 0.5), ql.Months) for mat in self.maturities]
        self.ql_curve = ql.YieldTermStructureHandle(
            ql.ZeroCurve(dates, list(self.rates), ql.Actual365Fixed())
        )

    def get_zero_rate(self, T):
        """
        Returns the interpolated zero-coupon rate for a maturity T (in years).
        """
        return float(self.curve(T))

    def get_discount_factor(self, T):
        """
        Returns the discount factor for a maturity T (in years).
        """
        return self.ql_curve.discount(T)

    def plot(self):
        """
        Plots the interpolated curve and displays the input data points.
        """
        T_grid = np.linspace(min(self.maturities), max(self.maturities), 300)
        plt.figure(figsize=(8, 4))
        plt.plot(T_grid, self.curve(T_grid), label="Interpolated Curve")
        plt.scatter(self.maturities, self.rates, color='red', label="Input Points")
        plt.title("US Treasury Zero-Coupon Yield Curve (FRED Data)")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Zero Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    curve = RiskFreeCurve()
    curve.fetch_data()
    print(f"Zero rate for 5 years: {curve.get_zero_rate(5):.4%}")
    print(f"Discount factor for 5 years: {curve.get_discount_factor(5):.6f}")
    curve.plot()
