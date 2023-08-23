from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints, ExtendedKalmanFilter
import numpy as np


class Kalman_Filter:
    def __init__(self, dim_x, dim_z, x, f, h, p, r, q):
        self.kf = KalmanFilter(dim_x, dim_z)
        self.kf.x = x
        self.kf.F = f
        self.kf.H = h
        self.kf.P = p
        self.kf.R = r
        self.kf.Q = q

    def fit(self, measurements):
        KF_fit, Covs, _, _ = self.kf.batch_filter(measurements)
        RTS_fit, Ps, _, _ = self.kf.rts_smoother(KF_fit, Covs)
        return KF_fit, RTS_fit


class Unscented_Kalman_Filter:
    def __init__(self, dim_x, dim_z, dt, fx, x, hx, r, q, n, alpha, beta, kappa):
        sigma_points = MerweScaledSigmaPoints(n=n, alpha=alpha, beta=beta, kappa=kappa)
        self.ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=dim_z, dt=dt, fx=fx, hx=hx, points=sigma_points)
        self.ukf.x = x
        self.ukf.R = r
        self.ukf.Q = q

    def fit(self, measurements):
        self.ukf.predict()
        self.ukf.update(measurements[0])
        UKF_fit, covs = self.ukf.batch_filter(measurements)
        RTS_fit, P, K = self.ukf.rts_smoother(UKF_fit, covs)
        return UKF_fit, RTS_fit


class Extended_Kalman_Filter:
    def __init__(self, dim_x, dim_z, x, p, r, q, fx, dt):
        self.dim_x = dim_x
        self.ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        self.ekf.x = x
        self.ekf.P = p
        self.ekf.Q = q
        self.ekf.R = r
        self.ekf.F = fx(dt)

    def fit(self, measurements, HJacobian, hx):
        EKF_fit = np.zeros((len(measurements), self.dim_x))
        for i in range(len(measurements)):
            self.ekf.predict()
            self.ekf.update(measurements[i], HJacobian=HJacobian, Hx=hx)
            EKF_fit[i] = self.ekf.x
        return EKF_fit

