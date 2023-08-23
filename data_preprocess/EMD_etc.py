from PyEMD import EMD, EEMD, CEEMDAN
from scipy.signal import savgol_filter
import numpy as np


class EMD_filter:
    def __init__(self, max_imf=None):
        self.max_imf = max_imf
        self.emd = EMD()

    def filter(self, signal):
        imfs = self.emd(signal)
        imfs_smoothed = self.emd(signal, max_imf=self.max_imf)
        filtered_signal = imfs_smoothed[-1]
        for i in range(len(imfs_smoothed) - 1):
            filtered_signal += imfs_smoothed[i]
        return imfs, imfs_smoothed, filtered_signal


class EEMD_filter:
    def __init__(self, extrema_detection="parabol"):
        self.eemd = EEMD()
        self.eemd.EMD.extrema_detection = extrema_detection

    def filter(self, t, s):
        eIMFs = self.eemd.eemd(s, t)
        rec = np.sum(eIMFs, axis=0)
        return eIMFs, rec


class CEEMDAN_filter:
    def __init__(self):
        self.ceemdan = CEEMDAN()

    def filter(self, s, window_length=5, polyorder=2):
        imfs = self.ceemdan(s)
        filtered_imfs = []
        for imf in imfs:
            filtered_imfs.append(savgol_filter(imf, window_length, polyorder))
        s_rec = np.sum(filtered_imfs, axis=0)
        return imfs, filtered_imfs, s_rec

