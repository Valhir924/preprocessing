import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GaussianProcessFilter:
    def __init__(self, length_scale=1.0,
                 length_scale_bounds=(1e-1, 10.0), noise_level=1, noise_level_bounds=(1e-10, 1e+1), alpha=0,
                 normalize_y=True):
        self.kernel = RBF(length_scale, length_scale_bounds) + WhiteKernel(
            noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        self.gp = GaussianProcessRegressor(kernel=self.kernel, alpha=alpha, normalize_y=normalize_y)

    def fit(self, x, y):
        self.gp.fit(x, y)

    def predict(self, x, return_std=True):
        return self.gp.predict(x, return_std=return_std)
