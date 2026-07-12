"""
问题1：双光束干涉确定碳化硅外延层厚度的数学模型
==================================================

物理模型：
- 空气 (n0=1) / 外延层 (n1(ν), 厚度d) / 衬底 (n2(ν))
- 红外光以入射角θ入射，仅考虑外延层-衬底界面的一次反射和透射
- 光束1：外延层表面反射
- 光束2：透射进入外延层 → 衬底表面反射 → 透射回空气

关键公式推导：
1. 相位差：δ = 4π·n1·d·cosθ1·ν
2. Snell定律：sinθ = n1·sinθ1
3. 反射率：R(ν) = |r01 + t01·t10·r12·e^(iδ)|²
4. 极值条件：2·n1(ν)·d·cosθ1·ν = m (整数) → 极大值
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar, minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一部分：SiC折射率模型
# ============================================================

# 物理常数
e_charge = 1.602e-19       # 电子电荷 (C)
eps0 = 8.854e-12           # 真空介电常数 (F/m)
c_light = 2.998e8          # 光速 (m/s)
m0 = 9.109e-31             # 电子静止质量 (kg)

# 4H-SiC 材料参数 (ordinary ray)
EPS_INF = 6.71             # 高频介电常数
NU_TO = 797.0              # TO声子频率 (cm⁻¹)
NU_LO = 972.0              # LO声子频率 (cm⁻¹)
M_STAR_RATIO = 0.42        # 电子有效质量比 m*/m0

def n_SiC_intrinsic(nu):
    """
    本征(轻掺杂)4H-SiC折射率 (ordinary ray)
    基于介电函数模型: ε(ν) = ε_∞·(ν²-ν_LO²)/(ν²-ν_TO²)

    参数: nu - 波数 (cm⁻¹), 标量或数组
    返回: 折射率
    注意: 仅在 ν > ν_LO 的透明区域有效
    """
    nu = np.asarray(nu, dtype=float)
    n_sq = EPS_INF * (nu**2 - NU_LO**2) / (nu**2 - NU_TO**2)
    return np.sqrt(np.maximum(n_sq, 1.0))


def n_SiC_doped(nu, N_cm3=1e18):
    """
    掺杂4H-SiC折射率 (含Drude自由载流子修正)

    ε(ν) = ε_∞·(ν²-ν_LO²)/(ν²-ν_TO²) - ωp²/ω²

    参数:
      nu - 波数 (cm⁻¹)
      N_cm3 - 掺杂浓度 (cm⁻³)
    """
    nu = np.asarray(nu, dtype=float)
    n_sq_intrinsic = EPS_INF * (nu**2 - NU_LO**2) / (nu**2 - NU_TO**2)

    # Drude修正: Δε = -ωp²/ω² = -νp²/ν²
    # νp² = N·e²/(4π²·ε0·m*·c²)  (νp单位: cm⁻¹)
    m_star = M_STAR_RATIO * m0
    c_cm = c_light * 100  # cm/s
    nu_p_sq = (N_cm3 * 1e6) * e_charge**2 / (4 * np.pi**2 * eps0 * m_star * c_cm**2)
    # nu_p_sq 的单位是 (rad/s)² / (rad/s per cm⁻¹)² = cm⁻²
    # 实际上: νp² = Ne²/(4π²ε₀m*(c_cm)²) 这里c_cm用cm/s

    n_sq = n_sq_intrinsic - nu_p_sq / nu**2
    return np.sqrt(np.maximum(n_sq, 0.01))


# ============================================================
# 第二部分：双光束干涉反射率模型
# ============================================================

def fresnel_coefficients(n0, n1, n2, theta0):
    """
    计算双光束干涉所需的Fresnel系数 (s偏振)

    返回: r01, t01, t10, r12, cos_theta1, cos_theta2
    """
    # Snell定律
    sin_theta1 = n0 * np.sin(theta0) / n1
    cos_theta1 = np.sqrt(np.maximum(1 - sin_theta1**2, 0))
    sin_theta2 = n0 * np.sin(theta0) / n2
    cos_theta2 = np.sqrt(np.maximum(1 - sin_theta2**2, 0))

    # s偏振Fresnel系数
    r01 = (n0 * np.cos(theta0) - n1 * cos_theta1) / (n0 * np.cos(theta0) + n1 * cos_theta1)
    t01 = 2 * n0 * np.cos(theta0) / (n0 * np.cos(theta0) + n1 * cos_theta1)
    t10 = 2 * n1 * cos_theta1 / (n1 * cos_theta1 + n0 * np.cos(theta0))
    r12 = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)

    return r01, t01, t10, r12, cos_theta1, cos_theta2


def reflectance_double_beam(nu, d, theta0_deg, N_epi=1e15, N_sub=1e18):
    """
    双光束干涉反射率模型

    R(ν) = |r01 + t01·t10·r12·exp(iδ)|²

    其中相位差: δ = 4π·n1·d·cosθ1·ν

    参数:
      nu - 波数数组 (cm⁻¹)
      d - 外延层厚度 (cm)
      theta0_deg - 入射角 (度)
      N_epi - 外延层掺杂浓度 (cm⁻³)
      N_sub - 衬底掺杂浓度 (cm⁻³)
    返回:
      R - 反射率 (%)
    """
    nu = np.asarray(nu, dtype=float)
    theta0 = np.radians(theta0_deg)
    n0 = 1.0  # 空气

    # 折射率
    n1 = n_SiC_doped(nu, N_epi)
    n2 = n_SiC_doped(nu, N_sub)

    # Fresnel系数
    r01, t01, t10, r12, cos_theta1, cos_theta2 = fresnel_coefficients(
        n0, n1, n2, theta0
    )

    # 相位差
    delta = 4 * np.pi * n1 * d * cos_theta1 * nu

    # 总反射场
    r_total = r01 + t01 * t10 * r12 * np.exp(1j * delta)

    R = np.abs(r_total)**2
    return R * 100  # 转换为百分比


def reflectance_simplified(nu, d, theta0_deg, N_epi=1e15, N_sub=1e18):
    """
    简化的双光束干涉反射率模型（展为余弦形式）

    R(ν) = R_avg + ΔR · cos(4π·n1·d·cosθ1·ν)

    适用于反射率较小的情形
    """
    nu = np.asarray(nu, dtype=float)
    theta0 = np.radians(theta0_deg)
    n0 = 1.0

    n1 = n_SiC_doped(nu, N_epi)
    n2 = n_SiC_doped(nu, N_sub)

    r01, t01, t10, r12, cos_theta1, cos_theta2 = fresnel_coefficients(
        n0, n1, n2, theta0
    )

    R1 = r01**2
    R2 = r12**2
    T1 = 1 - R1  # t01*t10 ≈ 1-r01²

    R_avg = R1 + T1**2 * R2
    delta_R = 2 * np.abs(r01) * T1 * np.abs(r12)

    delta = 4 * np.pi * n1 * d * cos_theta1 * nu

    R = R_avg + delta_R * np.cos(delta)
    return R * 100


# ============================================================
# 第三部分：从干涉条纹确定厚度的算法
# ============================================================

def find_interference_extrema(nu, R, min_prominence=0.5):
    """
    从反射率光谱中找干涉条纹的极大值和极小值位置

    参数:
      nu - 波数数组 (cm⁻¹)
      R - 反射率数组 (%)
      min_prominence - 峰的最小突出度
    返回:
      nu_maxima, R_maxima, nu_minima, R_minima
    """
    # 找极大值
    peaks_max, props_max = find_peaks(R, prominence=min_prominence, distance=20)
    # 找极小值（对-R找峰）
    peaks_min, props_min = find_peaks(-R, prominence=min_prominence, distance=20)

    return (nu[peaks_max], R[peaks_max],
            nu[peaks_min], R[peaks_min])


def thickness_from_adjacent_peaks(nu_peaks, theta0_deg, n_func=None, N_doping=1e15):
    """
    从相邻极大值(或极小值)的间距计算厚度

    对相邻同类型极值(阶数差=1):
    2·n(ν_i)·d·cosθ1(ν_i)·ν_i = m_i
    2·n(ν_{i+1})·d·cosθ1(ν_{i+1})·ν_{i+1} = m_i + 1

    ⇒ d = 1 / {2·[n(ν_{i+1})·cosθ1(ν_{i+1})·ν_{i+1} - n(ν_i)·cosθ1(ν_i)·ν_i]}

    近似（n≈常数）: d = 1/(2·n·Δν)

    参数:
      nu_peaks - 极值波数位置 (cm⁻¹)
      theta0_deg - 入射角 (度)
      n_func - 折射率函数，默认为本征SiC
      N_doping - 掺杂浓度 (cm⁻³)
    返回:
      d_list - 各相邻峰对计算的厚度 (cm)
    """
    if n_func is None:
        n_func = lambda nu: n_SiC_doped(nu, N_doping)

    theta0 = np.radians(theta0_deg)
    d_list = []

    for i in range(len(nu_peaks) - 1):
        nu1, nu2 = nu_peaks[i], nu_peaks[i+1]

        n1 = n_func(nu1)
        n2 = n_func(nu2)

        # cosθ1 from Snell's law
        cos_theta1_1 = np.sqrt(1 - (np.sin(theta0) / n1)**2)
        cos_theta1_2 = np.sqrt(1 - (np.sin(theta0) / n2)**2)

        # 光学厚度函数: OP(ν) = n(ν)·cosθ1(ν)·ν
        OP1 = n1 * cos_theta1_1 * nu1
        OP2 = n2 * cos_theta1_2 * nu2

        # d = 1 / (2·ΔOP)  其中 ΔOP = OP2 - OP1
        d = 1.0 / (2.0 * (OP2 - OP1))
        d_list.append(d)

    return np.array(d_list)


def thickness_from_all_peaks(nu_peaks, theta0_deg, n_func=None, N_doping=1e15):
    """
    利用所有极值位置，通过线性回归确定厚度

    原理: OP(ν_m)·2d = m + φ₀
    即: n(ν_m)·cosθ1(ν_m)·ν_m = m/(2d) + φ₀/(2d)

    对 OP(ν_m) vs m 做线性拟合，斜率 = 1/(2d)

    参数:
      nu_peaks - 极值波数位置 (cm⁻¹)
      theta0_deg - 入射角 (度)
    返回:
      d - 厚度 (cm), m_order - 干涉级次
    """
    if n_func is None:
        n_func = lambda nu: n_SiC_doped(nu, N_doping)

    theta0 = np.radians(theta0_deg)

    # 计算每个极值的光学厚度函数值
    OP_values = []
    for nu_val in nu_peaks:
        n_val = n_func(nu_val)
        cos_theta1 = np.sqrt(1 - (np.sin(theta0) / n_val)**2)
        OP = n_val * cos_theta1 * nu_val
        OP_values.append(OP)

    OP_values = np.array(OP_values)

    # 相邻极值的OP差应近似为 1/(2d)
    # 用OP差的平均值来估计d
    dOP = np.diff(OP_values)
    d_est = 1.0 / (2.0 * np.mean(dOP))

    # 确定干涉级次
    m_order = np.round(2 * d_est * OP_values).astype(int)

    # 用线性回归精确确定d
    # OP(ν_m) = m/(2d) + φ₀/(2d)
    # 对 OP_values vs m_order 做线性拟合
    A = np.vstack([m_order, np.ones(len(m_order))]).T
    result = np.linalg.lstsq(A, OP_values, rcond=None)
    slope = result[0][0]
    d_final = 1.0 / (2.0 * slope)

    return d_final, m_order


def thickness_from_FFT(nu, R, theta0_deg=10, n_ref=None, N_doping=1e15):
    """
    FFT方法确定厚度

    原理: R(ν) ≈ baseline + ΔR·cos(4π·n·d·cosθ1·ν)
    去除基线趋势后做FFT，主频 f₀ = 2·n·d·cosθ₁
    则 d = f₀ / (2·n·cosθ₁)

    参数:
      nu - 波数数组 (cm⁻¹)
      R - 反射率数组 (%)
      theta0_deg - 入射角 (度)
    返回:
      d - 厚度 (cm), freq, power
    """
    if n_ref is None:
        n_ref = np.mean(n_SiC_doped(nu, N_doping))

    # 等间距插值
    nu_uniform = np.linspace(nu.min(), nu.max(), len(nu))
    R_uniform = np.interp(nu_uniform, nu, R)

    # 多项式拟合去除基线趋势
    coeffs = np.polyfit(nu_uniform, R_uniform, 5)
    baseline = np.polyval(coeffs, nu_uniform)
    R_detrend = R_uniform - baseline

    # Hanning窗函数减少频谱泄漏
    window = np.hanning(len(R_detrend))
    R_windowed = R_detrend * window

    # FFT
    N = len(R_windowed)
    dnu = nu_uniform[1] - nu_uniform[0]  # 波数间隔 (cm⁻¹)
    freq = np.fft.rfftfreq(N, d=dnu)     # 频率轴 (cm)
    fft_vals = np.fft.rfft(R_windowed)
    power = np.abs(fft_vals)**2

    # 排除低频分量（前5个点可能是残余基线）
    power[:5] = 0

    # 主频
    idx_peak = np.argmax(power)
    f_dominant = freq[idx_peak]  # cm

    # f_dominant = 2·n·d·cosθ₁  →  d = f / (2·n·cosθ₁)
    theta0 = np.radians(theta0_deg)
    cos_theta1 = np.sqrt(1 - (np.sin(theta0) / n_ref)**2)

    d = f_dominant / (2 * n_ref * cos_theta1)

    return d, freq, power


# ============================================================
# 第四部分：读取数据并进行初步分析
# ============================================================

def load_data(filepath):
    """读取附件数据"""
    df = pd.read_excel(filepath, header=0)
    df.columns = ['nu', 'R']
    df['nu'] = pd.to_numeric(df['nu'], errors='coerce')
    df['R'] = pd.to_numeric(df['R'], errors='coerce')
    df = df.dropna()
    return df['nu'].values, df['R'].values


def analyze_data(filepath, theta0_deg, label, N_sub=1e18):
    """对单个附件数据进行分析"""
    nu, R = load_data(filepath)

    # 仅分析透明区域 (ν > ν_LO + buffer)
    mask = nu > 1200  # 避开reststrahlen带边缘效应
    nu_trans = nu[mask]
    R_trans = R[mask]

    # 找干涉极值
    nu_max, R_max, nu_min, R_min = find_interference_extrema(
        nu_trans, R_trans, min_prominence=0.3
    )

    print(f"\n{'='*60}")
    print(f"  {label} (入射角 {theta0_deg}°)")
    print(f"{'='*60}")
    print(f"  数据范围: ν = {nu_trans[0]:.1f} ~ {nu_trans[-1]:.1f} cm⁻¹")
    print(f"  反射率范围: R = {R_trans.min():.2f} ~ {R_trans.max():.2f} %")
    print(f"  检测到极大值: {len(nu_max)} 个")
    print(f"  检测到极小值: {len(nu_min)} 个")

    # 方法1：相邻极大值法
    if len(nu_max) >= 2:
        d_from_max = thickness_from_adjacent_peaks(nu_max, theta0_deg)
        print(f"\n  [方法1] 相邻极大值法:")
        print(f"    各峰对厚度 (μm): {np.array(d_from_max)*1e4}")
        print(f"    平均厚度: {np.mean(d_from_max)*1e4:.4f} μm")
        print(f"    标准差:   {np.std(d_from_max)*1e4:.4f} μm")
        print(f"    变异系数: {np.std(d_from_max)/np.mean(d_from_max)*100:.2f}%")

    # 方法1b：相邻极小值法
    if len(nu_min) >= 2:
        d_from_min = thickness_from_adjacent_peaks(nu_min, theta0_deg)
        print(f"\n  [方法1b] 相邻极小值法:")
        print(f"    平均厚度: {np.mean(d_from_min)*1e4:.4f} μm")
        print(f"    标准差:   {np.std(d_from_min)*1e4:.4f} μm")

    # 方法2：线性回归法（利用所有极大值）
    if len(nu_max) >= 3:
        d_lr, m_order = thickness_from_all_peaks(nu_max, theta0_deg)
        print(f"\n  [方法2] 线性回归法 (极大值):")
        print(f"    厚度: {d_lr*1e4:.4f} μm")
        print(f"    干涉级次范围: {m_order[0]} ~ {m_order[-1]}")

    if len(nu_min) >= 3:
        d_lr_min, m_order_min = thickness_from_all_peaks(nu_min, theta0_deg)
        print(f"\n  [方法2b] 线性回归法 (极小值):")
        print(f"    厚度: {d_lr_min*1e4:.4f} μm")

    # 方法3：FFT法
    d_fft, freq, power = thickness_from_FFT(nu_trans, R_trans, theta0_deg)
    print(f"\n  [方法3] FFT法:")
    print(f"    厚度: {d_fft*1e4:.4f} μm")

    return {
        'nu': nu, 'R': R,
        'nu_trans': nu_trans, 'R_trans': R_trans,
        'nu_max': nu_max, 'R_max': R_max,
        'nu_min': nu_min, 'R_min': R_min,
        'd_fft': d_fft, 'freq': freq, 'power': power
    }


# ============================================================
# 第五部分：可视化
# ============================================================

def plot_interference_spectrum(result, theta0_deg, label, save_path):
    """绘制干涉光谱和极值位置"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 完整反射率光谱
    ax1 = axes[0, 0]
    ax1.plot(result['nu'], result['R'], 'b-', linewidth=0.5, alpha=0.7)
    ax1.axvline(x=NU_TO, color='r', linestyle='--', alpha=0.5, label=f'ν_TO={NU_TO} cm⁻¹')
    ax1.axvline(x=NU_LO, color='g', linestyle='--', alpha=0.5, label=f'ν_LO={NU_LO} cm⁻¹')
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Reflectance (%)')
    ax1.set_title(f'{label} - Full Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2: 透明区域的干涉条纹 + 极值标记
    ax2 = axes[0, 1]
    ax2.plot(result['nu_trans'], result['R_trans'], 'b-', linewidth=0.8)
    ax2.plot(result['nu_max'], result['R_max'], 'rv', markersize=6, label='Maxima')
    ax2.plot(result['nu_min'], result['R_min'], 'g^', markersize=6, label='Minima')
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Reflectance (%)')
    ax2.set_title(f'{label} - Interference Fringes (θ={theta0_deg}°)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3: FFT频谱
    ax3 = axes[1, 0]
    # 只显示有意义的频率范围 (对应厚度1~20μm)
    # f = 2*n*d*cos(theta), d in [1e-4, 2e-3] cm, n~2.55
    f_min = 2 * 2.55 * 1e-4 * 0.995  # ~5e-4 cm
    f_max = 2 * 2.55 * 2e-3 * 0.995  # ~1e-2 cm
    mask_fft = (result['freq'] > f_min) & (result['freq'] < f_max)
    ax3.plot(result['freq'][mask_fft]*1e4, result['power'][mask_fft], 'b-', linewidth=0.8)
    # 标记主频
    valid_mask = (result['freq'] > f_min) & (result['freq'] < f_max)
    valid_power = result['power'].copy()
    valid_power[:5] = 0
    idx_peak = np.argmax(valid_power)
    d_fft = result['freq'][idx_peak] / (2 * 2.55 * np.sqrt(1-(np.sin(np.radians(theta0_deg))/2.55)**2))
    ax3.axvline(x=result['freq'][idx_peak]*1e4, color='r', linestyle='--',
                label=f'f₀={result["freq"][idx_peak]*1e4:.2f}×10⁻⁴cm, d={d_fft*1e4:.2f}μm')
    ax3.set_xlabel('Frequency f₀ (×10⁻⁴ cm)')
    ax3.set_ylabel('Power')
    ax3.set_title(f'{label} - FFT Analysis')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 图4: OP(ν) vs 级次 (线性回归验证)
    ax4 = axes[1, 1]
    if len(result['nu_max']) >= 3:
        theta0 = np.radians(theta0_deg)
        OP_vals = []
        for nu_val in result['nu_max']:
            n_val = n_SiC_intrinsic(nu_val)
            cos_theta1 = np.sqrt(1 - (np.sin(theta0) / n_val)**2)
            OP = n_val * cos_theta1 * nu_val
            OP_vals.append(OP)
        OP_vals = np.array(OP_vals)
        m_order = np.arange(len(OP_vals))
        # 线性拟合
        coeffs = np.polyfit(m_order, OP_vals, 1)
        slope = coeffs[0]
        d_est = 1.0 / (2.0 * slope)

        ax4.scatter(m_order, OP_vals, color='blue', s=30, zorder=5)
        ax4.plot(m_order, np.polyval(coeffs, m_order), 'r--',
                 label=f'Slope={slope:.2f}, d={d_est*1e4:.4f} μm')
        ax4.set_xlabel('Peak index')
        ax4.set_ylabel('n(ν)·cosθ₁·ν (cm⁻¹)')
        ax4.set_title(f'{label} - Linear Regression')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图已保存: {save_path}")


def plot_refractive_index(save_path):
    """绘制SiC折射率随波数的变化"""
    nu = np.linspace(1050, 4000, 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 本征SiC折射率
    n_intr = n_SiC_intrinsic(nu)
    ax1.plot(nu, n_intr, 'b-', linewidth=2, label='Intrinsic (light doping)')

    # 不同掺杂浓度的衬底折射率
    for N in [5e17, 1e18, 5e18, 1e19]:
        n_dop = n_SiC_doped(nu, N)
        ax1.plot(nu, n_dop, '--', linewidth=1.5,
                 label=f'N = {N:.0e} cm⁻³')

    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Refractive index n')
    ax1.set_title('4H-SiC Refractive Index Model')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 折射率对厚度计算的影响
    # 展示 n·cosθ₁·ν 随 ν 的变化
    theta0 = np.radians(10)
    n1 = n_SiC_intrinsic(nu)
    cos_theta1 = np.sqrt(1 - (np.sin(theta0) / n1)**2)
    OP = n1 * cos_theta1 * nu
    ax2.plot(nu, OP, 'b-', linewidth=2)
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('n(ν)·cosθ₁(ν)·ν')
    ax2.set_title('Optical Path Function OP(ν)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图已保存: {save_path}")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':

    print("=" * 70)
    print("  问题1：双光束干涉确定碳化硅外延层厚度的数学模型")
    print("=" * 70)

    # ---- 模型推导总结 ----
    print("\n" + "=" * 70)
    print("  数学模型推导")
    print("=" * 70)
    print("""
  【物理模型】
  三层介质系统：空气(n₀=1) / 外延层(n₁(ν), 厚度d) / 衬底(n₂(ν))
  入射角θ₀，仅考虑衬底界面的一次反射（双光束干涉）

  【关键公式】

  1) Snell定律:
     sinθ₀ = n₁·sinθ₁ = n₂·sinθ₂

  2) 两束光的相位差:
     δ(ν) = 4π·n₁(ν)·d·cosθ₁(ν)·ν

  3) Fresnel系数 (s偏振):
     r₀₁ = (n₀cosθ₀ - n₁cosθ₁) / (n₀cosθ₀ + n₁cosθ₁)
     t₀₁ = 2n₀cosθ₀ / (n₀cosθ₀ + n₁cosθ₁)
     t₁₀ = 2n₁cosθ₁ / (n₁cosθ₁ + n₀cosθ₀)
     r₁₂ = (n₁cosθ₁ - n₂cosθ₂) / (n₁cosθ₁ + n₂cosθ₂)

  4) 双光束干涉反射率:
     R(ν) = |r₀₁ + t₀₁·t₁₀·r₁₂·exp(iδ)|²

     展开为余弦形式:
     R(ν) = R₀ + ΔR·cos(δ)
     其中: R₀ = r₀₁² + (1-r₀₁²)²·r₁₂²
           ΔR = 2|r₀₁|·(1-r₀₁²)·|r₁₂|

  5) 干涉极值条件:
     极大值: 2·n₁(ν)·d·cosθ₁(ν)·ν = m        (m为整数)
     极小值: 2·n₁(ν)·d·cosθ₁(ν)·ν = m + 1/2

  6) 厚度计算公式:
     对相邻极值(阶数差1):
     d = 1 / {2·[OP(ν_{m+1}) - OP(ν_m)]}
     其中 OP(ν) = n₁(ν)·cosθ₁(ν)·ν

     近似(n₁≈常数):
     d = 1 / (2·n₁·cosθ₁·Δν)

  7) 折射率色散模型 (4H-SiC):
     n²(ν) = ε_∞·(ν²-ν_LO²)/(ν²-ν_TO²) - ν_p²/ν²

     其中: ε_∞=6.71, ν_TO=797 cm⁻¹, ν_LO=972 cm⁻¹
           ν_p² = Ne²/(4π²ε₀m*c²)  (Drude自由载流子修正)
    """)

    # ---- 折射率模型可视化 ----
    print("\n绘制折射率模型...")
    plot_refractive_index('q1_refractive_index.png')

    # ---- 数据分析 ----
    print("\n" + "=" * 70)
    print("  对附件数据进行初步分析")
    print("=" * 70)

    # 附件1: SiC, 10°
    result1 = analyze_data('附件/附件1.xlsx', 10, '附件1 (SiC, θ=10°)')
    plot_interference_spectrum(result1, 10, 'SiC θ=10°', 'q1_attachment1_analysis.png')

    # 附件2: SiC, 15°
    result2 = analyze_data('附件/附件2.xlsx', 15, '附件2 (SiC, θ=15°)')
    plot_interference_spectrum(result2, 15, 'SiC θ=15°', 'q1_attachment2_analysis.png')

    # ---- 两种入射角结果对比 ----
    print("\n" + "=" * 70)
    print("  两种入射角结果对比")
    print("=" * 70)

    # 用FFT法的结果做对比
    d1 = result1['d_fft']
    d2 = result2['d_fft']
    print(f"\n  FFT法厚度估计:")
    print(f"    附件1 (θ=10°):  d = {d1*1e4:.4f} μm")
    print(f"    附件2 (θ=15°):  d = {d2*1e4:.4f} μm")
    print(f"    差异: {abs(d1-d2)*1e4:.4f} μm ({abs(d1-d2)/((d1+d2)/2)*100:.2f}%)")

    # ---- 模型验证：用计算出的厚度生成理论光谱 ----
    print("\n" + "=" * 70)
    print("  模型验证：理论光谱与实测数据对比")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (result, theta_deg, ax) in enumerate([
        (result1, 10, axes[0]),
        (result2, 15, axes[1])
    ]):
        nu = result['nu_trans']
        R_meas = result['R_trans']

        # 用FFT法得到的厚度
        d_est = result['d_fft']

        # 生成理论反射率
        R_theory = reflectance_double_beam(nu, d_est, theta_deg)

        ax.plot(nu, R_meas, 'b-', linewidth=0.8, alpha=0.7, label='Measured')
        ax.plot(nu, R_theory, 'r-', linewidth=0.8, alpha=0.7, label='Model (double-beam)')
        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Reflectance (%)')
        ax.set_title(f'SiC θ={theta_deg}°, d={d_est*1e4:.4f} μm')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('q1_model_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  模型验证图已保存: q1_model_validation.png")

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("  问题1 总结")
    print("=" * 70)
    print("""
  已建立双光束干涉确定外延层厚度的数学模型:

  核心方程:
    R(ν) = |r₀₁ + t₀₁·t₁₀·r₁₂·exp(iδ)|²
    δ(ν) = 4π·n₁(ν)·d·cosθ₁(ν)·ν

  厚度确定方法:
    1) 相邻极值法:  d = 1/(2·ΔOP)  简单直观，受色散影响
    2) 线性回归法:  利用所有极值做OP(ν) vs m的线性拟合  更稳健
    3) FFT法:       对反射率谱做FFT提取主频  快速全局估计
    4) 曲线拟合法:  拟合理论R(ν)到实测数据  最精确(问题2展开)

  关键考虑:
    - SiC折射率随波数变化(色散)，不可忽略
    - 衬底掺杂浓度影响r₁₂，进而影响干涉条纹振幅
    - 仅ν > ν_LO (≈972 cm⁻¹) 的透明区域有干涉条纹
    - 小角度入射(10°, 15°)时cosθ₁≈1，角度修正量小但不应忽略
    """)
