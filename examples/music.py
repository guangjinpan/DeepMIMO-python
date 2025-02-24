import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# 导向矢量函数定义
# ---------------------------
def steering_vector_aoa(theta, M=32, d=32, wavelength=1):

    """
    均匀线阵AOA导向矢量
    theta：入射角（弧度）
    返回 (M,) 维度的复数向量
    """
    c = 3e8                   # 光速
    fc = 3.5e9                # 中心频率 3.5 GHz
    wavelength = c / fc        # 波长
    M = 32                    # 天线数
    N = 64                    # 子载波数
    BW = 10e6                 # 带宽 10 MHz

    # 子载波频率（假设对称分布在[-BW/2, BW/2]）
    freqs = np.linspace(0, BW, N)

    # 均匀线阵，天线间距取半波长
    d = wavelength / 2

    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d * m * np.sin(theta) / wavelength)


def steering_vector_aoa3d(theta, phi, M=32, d=32, wavelength=1):

    """
    均匀线阵AOA导向矢量
    theta：入射角（弧度）
    返回 (M,) 维度的复数向量
    """
    c = 3e8                   # 光速
    fc = 3.5e9                # 中心频率 3.5 GHz
    wavelength = c / fc        # 波长
    M = 32                    # 天线数
    N = 64                    # 子载波数
    BW = 10e6                 # 带宽 10 MHz

    # 子载波频率（假设对称分布在[-BW/2, BW/2]）
    freqs = np.linspace(0, BW, N)

    # 均匀线阵，天线间距取半波长
    d = wavelength / 2

    m = np.arange(M)

    # theta = -77.921093 /180*np.pi
    # print(phi*180/np.pi)
    # phi = 96.89779687 /180*np.pi

    vec = np.exp(-1j * 2 * np.pi * d * m * np.sin(phi) * np.sin(theta) / wavelength)
    # print(vec)
    # print(-1j * np.pi* np.sin(phi) * np.sin(theta))

    return vec

def steering_vector_toa(tau, freqs=3.5e9):
    """
    TOA导向矢量（基于子载波频率）
    tau：延时（秒）
    返回 (N,) 维度的复数向量
    """
    return np.exp(-1j * 2 * np.pi * freqs * tau)


# ---------------------------
# OMP算法实现
# ---------------------------
def omp(D, y, num_paths, tol=1e-6):
    """
    D: 字典矩阵 (M*N, num_candidates)
    y: 测量向量 (M*N,)
    num_paths: 要恢复的稀疏系数个数（即路径数）
    tol: 残差停止阈值
    返回选取的字典列索引以及对应的稀疏系数
    """
    residual = y.copy()
    indices = []
    selected_atoms = np.zeros((D.shape[0], 0), dtype=complex)
    
    for _ in range(num_paths):
        # 计算所有字典列与当前残差的相关性
        correlations = np.abs(np.dot(D.conj().T, residual))
        idx = np.argmax(correlations)
        indices.append(idx)
        # 将选中的字典原子加入到 selected_atoms 中
        selected_atoms = np.column_stack([selected_atoms, D[:, idx]])
        # 解决最小二乘问题，更新估计的系数
        x_hat, _, _, _ = np.linalg.lstsq(selected_atoms, y, rcond=None)
        # 更新残差
        residual = y - np.dot(selected_atoms, x_hat)
        if np.linalg.norm(residual) < tol:
            break
    return indices, x_hat

def calculate_distance_and_aoa(BSlocation, UElocation):
    """
    计算基站与用户之间的距离和到达角 (AOA)
    :param BSlocation: (x_b, y_b) 基站坐标
    :param UElocation: (x_u, y_u) 用户坐标
    :return: (distance, aoa)  距离 (m) 和 AOA (degree)
    """
    x_b, y_b = BSlocation
    x_u, y_u = UElocation
    
    # 计算欧几里得距离
    distance = np.sqrt((x_u - x_b)**2 + (y_u - y_b)**2)
    
    # 计算AOA，atan2 确保正确象限
    aoa_rad = np.arctan2(y_u - y_b, x_u - x_b)
    aoa_deg = np.degrees(aoa_rad)  # 转换为度
    
    return distance, aoa_deg

def music_f(y,phi):
    # ---------------------------
    # 构造联合字典（AOA-TOA联合字典）
    # ---------------------------
    # 为了降低字典规模，这里选用较为粗糙的网格
    # ---------------------------
    # 参数设置
    # ---------------------------
    c = 3e8                   # 光速
    fc = 3.5e9                # 中心频率 3.5 GHz
    wavelength = c / fc        # 波长
    M = 32                    # 天线数
    N = 64                    # 子载波数
    BW = 10e6                 # 带宽 10 MHz

    # 子载波频率（假设对称分布在[-BW/2, BW/2]）
    freqs = np.linspace(0, BW, N)

    # 均匀线阵，天线间距取半波长
    d = wavelength / 2


    theta_grid_omp = np.linspace(-np.pi/2, np.pi/2, 91)   # -90°到90°，共91个样本
    tau_grid_omp = np.linspace(0, 1000e-9, 100)            # 0到300 ns，共31个样本

    dictionary = []
    param_list = []  # 用于记录每一列对应的 (theta, tau)
    for theta in theta_grid_omp:
        a_aoa = steering_vector_aoa(theta, M=M, d=d, wavelength=wavelength)
        # a_aoa = steering_vector_aoa3d(theta, phi, M=M, d=d, wavelength=wavelength)
        for tau in tau_grid_omp:
            a_toa = steering_vector_toa(tau, freqs=freqs)
            # 联合导向矢量（Kronecker积），尺寸 (M*N,)
            a = np.kron(a_toa, a_aoa)
            dictionary.append(a)
            param_list.append((theta, tau))
    # 构造字典矩阵，尺寸为 (M*N, num_candidates)
    D = np.column_stack(dictionary)
    print("字典尺寸：", D.shape)

    # ---------------------------
    # 模拟多径信道数据
    # ---------------------------
    # 假设信道有两个路径
    P = 5
    true_params = [
        ((-117.833+180) * np.pi/180, 128/3e8),   # 路径1：30°，100 ns
        # (-2 * np.pi/180, 550e-9)   # 路径2：-20°，150 ns
    ]

    # 构造信道测量样本 y
    # y = np.zeros(M * N, dtype=complex)
    # for (theta, tau) in true_params:
    #     a_aoa = steering_vector_aoa(theta, M=M, d=d, wavelength=wavelength)
    #     a_aoa = steering_vector_aoa3d(theta, phi, M=M, d=d, wavelength=wavelength)
    #     a_toa = steering_vector_toa(tau, freqs=freqs)
    #     a = np.kron(a_toa, a_aoa)
    #     alpha = np.exp(1j * 2 * np.pi * np.random.rand())  # 随机复数增益
    #     print(a.shape)
    #     y += 1 * a
    # y1 = y.reshape((N,M))
    # # x=np.fft.ifft(y1[0,:])
    # # plt.plot(np.abs(x))
    # x1 = np.angle(y1[5,:])
    # print(x1.shape)
    # plt.plot(x1[1:]-x1[0])
    # plt.show()

    # #加入噪声
    # noise_level = 0.1
    # y += noise_level * (np.random.randn(M * N) + 1j * np.random.randn(M * N))



    # 设定要恢复的路径数
    y = y.reshape((-1))
    selected_indices, x_hat = omp(D, y.reshape((-1)), num_paths=P)
    estimated_params = [param_list[idx] for idx in selected_indices]

    print("估计的参数（AOA 单位：度，TOA 单位：ns）：")
    for theta, tau in estimated_params:
        print("AOA: {:.2f}°, TOA: {:.2f} ns".format(theta * 180/np.pi, tau * 1e9))

    # ---------------------------
    # 可选：绘制估计谱（通过投影字典列相关性展示）
    # ---------------------------
    # corr_values = np.abs(np.dot(D.conj().T, y))
    # corr_matrix = corr_values.reshape(len(theta_grid_omp), len(tau_grid_omp))
    # print(corr_matrix.shape)
    # plt.figure(figsize=(8, 6))
    # plt.imshow(10*np.log10(corr_matrix.T), extent=[theta_grid_omp[0]*180/np.pi, theta_grid_omp[-1]*180/np.pi,
    #         tau_grid_omp[-1]*1e9, tau_grid_omp[0]*1e9], aspect='auto', cmap='jet')
    # plt.xlabel('AOA (度)')
    # plt.ylabel('TOA (ns)')
    # plt.title('OMP 字典相关性谱')
    # plt.colorbar(label='相关性 (dB)')
    # plt.show()
    