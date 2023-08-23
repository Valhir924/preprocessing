import pywt
import numpy as np


class WaveletTransform:
    def __init__(self, wavelet_name, data, level, mode):
        self.wavelet_name = wavelet_name
        self.data = data
        self.level = level
        self.mode = mode

    def forward_transform(self):
        coeffs = pywt.wavedec(self.data, self.wavelet_name, mode=self.mode, level=self.level, axis=-1)
        return coeffs

    def thresholding(self, coeffs, mode='soft'):
        if mode == 'soft':
            value = np.sqrt(2 * np.log(len(self.data))) * np.median(abs(coeffs[-1])) / 0.6745
        else:
            value = None
        for i in range(1, len(coeffs)):
            if mode == 'hard':
                threshold = 0.5
                value = threshold * max(coeffs[i])
            coeffs[i] = pywt.threshold(coeffs[i], value)
        return coeffs

    def inverse_transform(self, coeffs):
        rec_data = pywt.waverec(coeffs, self.wavelet_name, mode='symmetric', axis=-1)
        return rec_data


