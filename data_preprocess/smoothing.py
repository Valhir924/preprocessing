import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
from numpy.polynomial.polynomial import polyfit


class Expotional_smoothing:
    def __init__(self, data):
        self.data = data

    def exponential_smoothing(self, seasonal_periods, trend=None, seasonal=None, initialization_method='estimated',
                              optimized=False, use_boxcox=False, damped_trend=False):
        model = ExponentialSmoothing(self.data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal,
                                     initialization_method=initialization_method, optimized=optimized,
                                     use_boxcox=use_boxcox, damped_trend=damped_trend)
        fit = model.fit()
        return fit.fittedvalues

    def simple_exp_smoothing(self, smoothing_level=None, initialization_method='estimated'):
        model = SimpleExpSmoothing(self.data, initialization_method=initialization_method)
        fit = model.fit(smoothing_level=smoothing_level)
        return fit.fittedvalues

    def holt(self, exponential=False, damped_trend=False, smoothing_level=None, smoothing_trend=None,
             initialization_method='estimated', optimized=False):
        model = Holt(self.data, exponential=exponential, damped_trend=damped_trend,
                     initialization_method=initialization_method, optimized=optimized)
        fit = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend)
        return fit.fittedvalues


class Polynomial_smoothing:
    def __init__(self, data_y, data_x=None):
        if data_x is None:
            self.data_x = np.arange(len(data_y))
        self.data_y = data_y

    def polynomial_smoothing(self, degree):
        coefficients = polyfit(self.data_x, self.data_y, degree)
        polynomial = np.poly1d(coefficients[::-1])
        return self.data_x, polynomial(self.data_x)
