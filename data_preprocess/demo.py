import numpy as np
import pywt
import pandas as pd
from filterpy.common import Q_discrete_white_noise
from EMD_etc import EMD_filter, EEMD_filter, CEEMDAN_filter
import matplotlib.pyplot as plt
from noise_generating import noise_generator
from smoothing import Polynomial_smoothing, Expotional_smoothing
from kalman_filter import Kalman_Filter, Unscented_Kalman_Filter, Extended_Kalman_Filter
from wavelet_transform import WaveletTransform
from gaussian_process_regression import GaussianProcessFilter
from feature_engineering import Feature_engineering_SN, Feature_engineering_DMR


def Polynomial_demo():
    # Generate noisy data
    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    s = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    s += 5 * sin(11, 2.7)
    s += 3 * sin(14, 1.6)
    s += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    s += t ** 2.1 - t
    s_noise = noise_generator(s, 1)
    degree = 3

    model = Polynomial_smoothing(data_y=s_noise, data_x=t)
    t, S_edited = model.polynomial_smoothing(degree)

    plt.figure(figsize=(12, 4))
    plt.title("polynomial contrast")
    plt.plot(t, s_noise, 'b', t, S_edited, 'r')
    plt.legend(['original', 'poltnimial'])
    plt.show()


def Expotional_demo():
    # Generate noisy data
    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    s = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    s += 5 * sin(11, 2.7)
    s += 3 * sin(14, 1.6)
    s += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    s += t ** 2.1 - t
    testing_data = noise_generator(s, 1)

    model = Expotional_smoothing(testing_data)
    edited_SIMPLE = model.simple_exp_smoothing(initialization_method='estimated')
    edited_holt_linear = model.holt(initialization_method="estimated",
                                    smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
    edited_holt_exp = model.holt(exponential=True, initialization_method="estimated",
                                 smoothing_level=0.8, smoothing_trend=0.2, optimized=False)
    edited_holt_damp = model.holt(damped_trend=True, initialization_method="estimated",
                                  smoothing_level=0.8, smoothing_trend=0.2)
    Holts_Winter_aa = model.exponential_smoothing(
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        use_boxcox=True,
        initialization_method="estimated",
    )
    Holts_Winter_am = model.exponential_smoothing(
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        use_boxcox=True,
        initialization_method="estimated",
    )
    Holts_Winter_da = model.exponential_smoothing(
        seasonal_periods=4,
        trend="add",
        seasonal="add",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated",
    )
    Holts_Winter_dm = model.exponential_smoothing(
        seasonal_periods=4,
        trend="add",
        seasonal="mul",
        damped_trend=True,
        use_boxcox=True,
        initialization_method="estimated",
    ).fit()

    # draw the contrast picture

    plt.figure(figsize=(12, 4))
    plt.title("expotional contrast")
    plt.plot(np.arange(len(testing_data)), testing_data, 'b',
             np.arange(len(testing_data)), edited_SIMPLE, 'r--',
             np.arange(len(testing_data)), edited_holt_damp, 'g--',
             np.arange(len(testing_data)), Holts_Winter_dm, 'y--')
    plt.legend(['original data', 'Simple ExpSmoothing', 'Additive damped trend holt', 'holt winters'])
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.title('Comparison between Holts')
    plt.plot(np.arange(len(testing_data)), testing_data, 'b',
             np.arange(len(testing_data)), edited_holt_linear, 'r--',
             np.arange(len(testing_data)), edited_holt_exp, 'g--',
             np.arange(len(testing_data)), edited_holt_damp, 'y--')
    plt.legend(['original data', 'linear', 'exp', 'damp trend'])
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.title('Comparison between Holt Winters')
    plt.plot(np.arange(len(testing_data)), testing_data, 'b',
             np.arange(len(testing_data)), Holts_Winter_aa, 'r--',
             np.arange(len(testing_data)), Holts_Winter_am, 'g--',
             np.arange(len(testing_data)), Holts_Winter_da, 'y--',
             np.arange(len(testing_data)), Holts_Winter_dm, 'm--')
    plt.legend(['original data', 'add trend, add seasonal of period', 'add trend, mul seasonal of period',
                'damped trend, add seasonal of period', 'damped trend, mul seasonal of period'])
    plt.show()


def KF_demo():
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.25)

    R = 30
    itr = 100

    real_state = []
    x = np.array([10, 5]).reshape(2, 1)

    for i in range(itr):
        real_state.append(x[0, 0])
        x = np.dot(F, x) + np.random.multivariate_normal(mean=(0, 0), cov=Q).reshape(2, 1)

    measurements = noise_generator(real_state, var=R)

    # initialization
    P = np.diag([10., 10.])
    x = np.array([10, 5]).reshape(2, 1)

    # filter
    kf = Kalman_Filter(dim_x=2, dim_z=1, x=x, f=F, h=np.array([[1., 0.]]), p=P, r=R,
                       q=Q_discrete_white_noise(dim=2, dt=dt, var=1e-2))

    Xs, Ms = kf.fit(measurements)
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, len(measurements)), measurements[:], label='Measurements')
    plt.plot(range(0, len(real_state)), real_state[:], label='Real statement')
    plt.plot(range(0, len(Xs)), np.array(Xs)[:, 0], label='Kalman Filter')
    plt.plot(range(0, len(Ms)), np.array(Ms)[:, 0], label='RTS position')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Position', fontsize=14)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.axhline(5, label='Real statement')  # , label='$GT_x(real)$'
    plt.plot(range(0, len(Xs)), np.array(Xs)[:, 1], 'g', label='Kalman Filter')
    plt.plot(range(0, len(Ms)), np.array(Ms)[:, 1], 'r', label='RTS')
    plt.legend()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('velocity', fontsize=14)
    plt.show()


def UKF_demo():
    def fx(x_fx, dt_fx):
        F = np.array([[1, dt_fx, 0.5 * dt_fx * dt_fx],
                      [0, 1, dt_fx],
                      [0, 0, 1]])
        return F @ x_fx

    def hx(x_hx):
        H = np.array([[1, 0, 0]])
        return H @ x_hx

    # 初始化参数
    dt = 0.1  # 采样时间
    q_std = 2  # 系统噪声的标准差
    r_std = 10  # 测量噪声的标准差
    x = np.array([0, 3, 5])  # 初始状态，[位置、速度、加速度]

    # 创建 UKF 滤波器
    ukf = Unscented_Kalman_Filter(dim_x=3, dim_z=1, dt=dt, fx=fx, x=x, hx=hx, r=np.diag([r_std ** 2]),
                                  q=np.diag([q_std, q_std, q_std]) ** 2, n=3, alpha=0.1, beta=2., kappa=0.)

    # 模拟数据
    time = np.arange(0, 10, dt)
    n = len(time)
    xs = np.zeros((n, 3))
    zs = np.zeros((n, 1))

    for i in range(n):
        t = time[i]
        # 模拟状态转移
        x = fx(x, dt)
        # 加上噪声（过程噪声）并保存
        xs[i] = x + np.random.randn(3) * q_std
        # 模拟观测
        z = hx(x)
        # 加上噪声（测量噪声）并保存
        zs[i] = z + np.random.randn() * r_std

    # 执行UKF滤波
    xs, Ms = ukf.fit(measurements=zs)

    plt.plot(time, xs[:, 0], label='filter')
    plt.plot(time, zs, label='measurements')
    plt.plot(time, Ms[:, 0], label='RTS')
    plt.legend(loc='best')
    plt.show()


def EKF_demo():
    # 定义状态转移函数 f(x)
    def fx(x_fx, dt_fx):
        F = np.array([[1, dt_fx, 0.5 * dt_fx * dt_fx],
                      [0, 1, dt_fx],
                      [0, 0, 1]])
        return F @ x_fx

    # 定义观测函数 h(x)
    def hx(x_hx):
        H = np.array([[1, 0, 0]])
        return H @ x_hx

    # 定义状态转移函数的雅可比矩阵 Fx
    def Fx(dt_Fx):
        F = np.array([[1, dt_Fx, .5 * dt_Fx * dt_Fx],
                      [0, 1, dt_Fx],
                      [0, 0, 1]])
        return F

    # 定义观测函数的雅可比矩阵 HJacobian
    def HJacobian(x):
        H = np.array([[1, 0, 0]])
        return H

    # 初始化参数
    dt = 0.1  # 采样时间
    q_std = 2  # 系统噪声的标准差
    r_std = 10  # 测量噪声的标准差
    x = np.array([0, 3, 5])  # 初始状态，[位置、速度、加速度]
    P = np.diag([10, 1, 0.1])  # 初始协方差矩阵

    # 定义 EKF 滤波器
    ekf = Extended_Kalman_Filter(dim_x=3, dim_z=1, x=x, p=P, r=np.diag([r_std ** 2]),
                                 q=Q_discrete_white_noise(dim=3, dt=dt, var=q_std ** 2), fx=Fx(dt), dt=dt)

    # 模拟数据
    time = np.arange(0, 10, dt)
    n = len(time)
    xs = np.zeros((n, 3))
    zs = np.zeros((n, 1))

    for i in range(n):
        t = time[i]
        # 模拟状态转移
        x = fx(x, dt)
        # 加上噪声（过程噪声）并保存
        xs[i] = x + np.random.randn(3) * q_std
        # 模拟观测
        z = hx(x)
        # 加上噪声（测量噪声）并保存
        zs[i] = z + np.random.randn() * r_std

    xs = ekf.fit(measurements=zs, HJacobian=HJacobian, hx=hx)

    plt.plot(time, xs[:, 0], label='filter')
    plt.plot(time, zs, label='measurements')
    plt.legend(loc='best')
    plt.show()


def wavelet_demo():
    wavelet = 'db4'
    data = pywt.data.ecg()
    # 添加高斯噪声
    data_noise = noise_generator(data, 1)
    filter = WaveletTransform(wavelet, data, level=6, mode='symmetric')
    # 进行Wavelet Packet分解
    coeffs_noise = filter.forward_transform()

    # 绘制分解得到的系数，包括approximation coefficients和detail coefficients
    fig, axarr = plt.subplots(len(coeffs_noise), figsize=(10, 15))
    for i in range(len(coeffs_noise)):
        axarr[i].plot(coeffs_noise[i])
        if i == 0:
            axarr[i].set_title("Approximation Coefficients")
        else:
            axarr[i].set_title("Detail Coefficients {}".format(i))

    coeffs_noise = filter.thresholding(coeffs=coeffs_noise, mode='soft')

    # 将分解得到的系数进行重构
    reconstructed_signal = filter.inverse_transform(coeffs_noise)

    # 绘制处理后的信号和原信号（包括噪声）
    plt.figure(figsize=[12, 6])
    plt.plot(np.arange(len(data)), data, 'r')
    plt.plot(np.arange(len(data)), data_noise, 'b')
    plt.plot(np.arange(len(data)), reconstructed_signal, 'g')
    plt.legend(['original data', 'noisy data', 'reconstructed data'], loc='best')
    # 输出重构后的信号和原信号（包括噪声）的均方误差
    mse = np.mean((reconstructed_signal - data_noise) ** 2)
    plt.title('CONTRACT')
    print("MSE: ", mse)
    plt.show()


def gaussian_demo():
    # Generate noisy data
    X = np.linspace(0, 10, 100)
    y = noise_generator(np.sin(X), var=0.2)

    filter = GaussianProcessFilter(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), noise_level=1,
                                   noise_level_bounds=(1e-10, 1e+1), alpha=0, normalize_y=True)

    filter.fit(X.reshape(-1, 1), y)

    # Generate test data
    X_test = np.linspace(0, 10, 1000)

    # Predict mean and standard deviation of test data
    y_pred, sigma = filter.predict(X_test.reshape(-1, 1), return_std=True)

    # Plot noisy data, true function, and predicted function with confidence interval
    plt.scatter(X, y, c='r', label='Noisy data')
    plt.plot(X_test, np.sin(X_test), 'k:', label='True function')
    plt.plot(X_test, y_pred, 'b-', label='Predicted function')
    plt.fill_between(X_test, y_pred - 2 * sigma, y_pred + 2 * sigma, alpha=0.2, color='blue')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def EMD_demo():
    # Generate noisy data
    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    s = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    s += 5 * sin(11, 2.7)
    s += 3 * sin(14, 1.6)
    s += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    s += t ** 2.1 - t
    s_noise = noise_generator(s, 1)

    emd = EMD_filter(5)

    # Extract intrinsic mode functions (IMFs)
    IMFs, IMFs_smoothed, s_rec = emd.filter(s_noise)

    # Plot IMFs
    plt.figure(figsize=[12, 6])
    # plt.title('IMFs')
    plt.subplot(IMFs.shape[0] + 1, 1, 1)
    plt.plot(t, s_noise, 'r')
    plt.title('Noisy signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    for n, imf in enumerate(IMFs):
        plt.subplot(IMFs.shape[0] + 1, 1, n + 2)
        plt.plot(t, imf, 'g')
        plt.title('IMF ' + str(n + 1))
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # Plot smoothed IMFs
    plt.figure(figsize=[12, 6])
    # plt.title('SMOOTHED IMFs')
    plt.subplot(IMFs_smoothed.shape[0] + 1, 1, 1)
    plt.plot(t, s_noise, 'r')
    plt.title('Noisy signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    for n, imf in enumerate(IMFs_smoothed):
        plt.subplot(IMFs_smoothed.shape[0] + 1, 1, n + 2)
        plt.plot(t, imf, 'g')
        plt.title('IMF ' + str(n + 1))
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.tight_layout()
    plt.show()

    plt.figure(figsize=[12, 6])
    plt.title('conclusion')
    plt.plot(t, s, 'r')
    plt.plot(t, s_noise, 'k')
    plt.plot(t, s_rec, 'y')
    plt.legend(['original data', 'noisy data', 'rec data'], loc='best')
    plt.show()


def EEMD_demo():
    # Define signal
    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    s = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    s += 5 * sin(11, 2.7)
    s += 3 * sin(14, 1.6)
    s += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    s += t ** 2.1 - t
    s_noise = noise_generator(s, 1)

    # Assign EEMD to `eemd` variable
    eemd = EEMD_filter()
    eIMFs = eemd.filter(t, s_noise)
    nIMFs = eIMFs.shape[0]

    # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs + 1, 1, 1)
    plt.plot(t, s, 'r')

    for n in range(nIMFs):
        plt.subplot(nIMFs + 1, 1, n + 2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" % (n + 1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('eemd_example', dpi=120)
    plt.show()
    s_rec = np.sum(eIMFs, axis=0)
    plt.figure(figsize=[12, 6])
    plt.plot(t, s, 'r', t, s_noise, 'b', t, s_rec, 'g')
    plt.legend(['original data', 'noise_data', 'rec_data'], loc='best')
    plt.show()


def CEEMDAN_demo():
    # Generate test signal
    t = np.linspace(0, 1, 200)

    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    s = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    s += 5 * sin(11, 2.7)
    s += 3 * sin(14, 1.6)
    s += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    s += t ** 2.1 - t
    s_noise = noise_generator(s, 1)

    # Define EMD parameters
    ceemdan = CEEMDAN_filter()

    # Decompose signal into IMFs
    imfs, filtered_imfs, s_rec = ceemdan.filter(s_noise)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(t, s, 'r')
    plt.title("Original signal")
    plt.xlabel("Time (s)")
    for i, imf in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, i + 2)
        plt.plot(t, imf, 'g')
        plt.title("IMF " + str(i + 1))
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

    # Plot the reconstructed signal
    plt.figure(figsize=[12, 6])
    plt.plot(t, s, 'r--', t, s_noise, 'b', t, s_rec, 'g--')
    plt.legend(['original data', 'noise_data', 'rec_data'], loc='best')
    plt.show()


def Normalization_demo():
    data = np.array([[1.2, 3.3], [2.1, 4.5], [3.2, 1.3]])

    # 创建MinMaxScaler对象
    scaler = Feature_engineering_SN(method='normalization')

    # 使用fit_transform进行归一化
    scaler.fit(data)
    scaled_data = scaler.transform(data)

    print("Original data:\n", data)
    print("Scaled data:\n", scaled_data)


def Standardization_demo():
    # 生成一个5×3的随机矩阵作为原始数据
    data = np.random.rand(5, 3)

    # 创建StandardScaler对象并计算均值和标准差
    scaler = Feature_engineering_SN(method='standardization')
    scaler.fit(data)

    # 对数据进行标准化
    data_std = scaler.transform(data)
    print(data_std)


def Duplicate_values_handling():
    # 创建一个包含数值型数据的DataFrame
    df = pd.DataFrame({'A': [1, 1, 3, 4, 1], 'B': [1, 1, 3, 4, 1], 'C': [100, 100, 300, 400, 100]})

    scaler = Feature_engineering_DMR(method='duplicate values handling')

    # 删除重复的数据行
    df_deduped = scaler.transform(df)
    print("Deduplicated DataFrame:")
    print(df_deduped)


def Missing_value_handling_demo():
    # 创建包含缺失值的DataFrame对象
    data = {'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, np.nan, 8],
            'C': [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    print(df)

    model = Feature_engineering_DMR(method='missing value handling')
    df = model.transform(df, delete_missing_row=True)
    print(df)


def resample_demo():
    index = pd.date_range('1/1/2000', periods=9, freq='T')
    series = pd.Series(range(9), index=index)
    model = Feature_engineering_DMR(method='resample')
    print(model.transform(series, resample_fq='30S'))


