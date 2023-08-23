import numpy as np


def noise_generator(time_series, var=1, m_dim=1):
    if m_dim == 1:
        noise = np.random.normal(0, var, size=len(time_series))
        noisy_time_series = time_series + noise

        return noisy_time_series
    else:
        pass
