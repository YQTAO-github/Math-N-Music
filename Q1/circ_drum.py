import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import dblquad
from scipy.special import jn, jn_zeros,jv
import collections
from mpl_toolkits.mplot3d import Axes3D

def circular_coefficients(phi, psi, R, c, m, n):
    """计算圆形域波动方程展开系数"""
    # 获取贝塞尔函数根
    # m阶Bessel函数的第n个正根
    alpha = jn_zeros(m, n)[-1]
    omega = alpha * c / R
    
    # 正交归一化系数计算
    def norm_factor(m, alpha):
        integral = (R**2/2)*(jv(m+1, alpha)**2)
        return np.pi * integral * (2 if m==0 else 1)
    
    # 模式函数模板
    def mode_func_cos_cos(r, theta, t=0):
        spatial = jv(m, alpha*r/R) * np.cos(m*theta)
        temporal = np.cos(omega*t) 
        return spatial * temporal
    
    def mode_func_cos_sin(r, theta, t=0):
        spatial = jv(m, alpha*r/R) * np.cos(m*theta)
        temporal = np.sin(omega*t) 
        return spatial * temporal
    
    # 系数A计算（位移项）
    def integrand_A_cos(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return phi(x,y) * mode_func_cos_cos(r, theta) * r  # 包含雅可比行列式r
    
    A_cos = dblquad(integrand_A_cos, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / norm_factor(m, alpha)
    
    # 系数B计算（速度项）
    def integrand_B_cos(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return psi(x,y) * mode_func_cos_sin(r, theta) * r
    
    B_cos = dblquad(integrand_B_cos, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / (omega * norm_factor(m, alpha))
    
    # 模式函数模板
    def mode_func_sin_cos(r, theta, t=0):
        spatial = jv(m, alpha*r/R) * np.sin(m*theta)
        temporal = np.cos(omega*t) 
        return spatial * temporal
    
    def mode_func_sin_sin(r, theta, t=0):
        spatial = jv(m, alpha*r/R) * np.sin(m*theta)
        temporal = np.cos(omega*t) 
        return spatial * temporal
    
    # 系数A计算（位移项）
    def integrand_A_sin(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return phi(x,y) * mode_func_sin_cos(r, theta) * r  # 包含雅可比行列式r
    
    A_sin = dblquad(integrand_A_sin, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / norm_factor(m, alpha)
    
    # 系数B计算（速度项）
    def integrand_B_sin(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return psi(x,y) * mode_func_sin_sin(r, theta) * r
    
    B_sin = dblquad(integrand_B_sin, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / (omega * norm_factor(m, alpha))
    
    return A_cos, B_cos, A_sin, B_sin, omega

def circular_wave_solution(phi, psi, R, c, mmax, nmax):
    """构建完整波动解"""
    # 初始化系数矩阵
    coeffs = []
    for m in range(mmax+1):
        for n in range(1, nmax+1):
            A_cos, B_cos, A_sin, B_sin, omega = circular_coefficients(phi, psi, R, c, m, n)
            # print(A_cos)
            coeffs.append( (m, n, A_cos, B_cos, A_sin, B_sin, omega) )
    return coeffs

def animate_circular_vibration(coeffs, R, duration, fps=24):
    """生成振动动画"""
    # 创建极坐标网格
    grid_size = 80
    r = np.linspace(0, R, grid_size)
    theta = np.linspace(0, 2*np.pi, grid_size)
    R_grid, Theta_grid = np.meshgrid(r, theta)
    
    # 转换为笛卡尔坐标用于显示
    X = R_grid * np.cos(Theta_grid)
    Y = R_grid * np.sin(Theta_grid)
    
    # 时间序列
    t = np.linspace(0, duration, int(fps*duration))
    
    # 预计算各模式空间分布
    spatial_modes = []
    for (m, n, A_cos, B_cos, A_sin, B_sin, omega) in coeffs:
        alpha = jn_zeros(m, n)[-1]
        spatial_cos = jv(m, alpha*R_grid/R) * np.cos(m*Theta_grid)
        spatial_sin = jv(m, alpha*R_grid/R) * np.sin(m*Theta_grid)
        spatial_modes.append( (A_cos, B_cos, A_sin, B_sin, omega, spatial_cos, spatial_sin) )
    
    # 初始化图形
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 0.1])
    ax.set_zlim(-hmax,hmax)
    
    def update(frame):
        ax.cla()
        current_time = t[frame]
        Z = np.zeros_like(X)
        
        # 叠加各模式贡献
        for (A_cos, B_cos, A_sin, B_sin, omega, spatial_cos, spatial_sin) in spatial_modes:
            temporal_A = A_cos*np.cos(omega*current_time) + B_cos*np.sin(omega*current_time)
            temporal_B = A_sin*np.cos(omega*current_time) + B_sin*np.sin(omega*current_time)
            Z += temporal_A * spatial_cos + temporal_B * spatial_sin
        
        # 应用圆形边界
        Z[R_grid > R] = 0
        
        # # 动态归一化
        # max_amp = np.max(np.abs(Z))
        # if max_amp > 0: Z /= max_amp
        
        # 绘制表面
        ax.plot_surface(X, Y, Z, cmap='coolwarm', rstride=2, cstride=2)
        ax.set_zlim(-100,100)
        ax.set_title(f"Circular Membrane (t={current_time:.2f}s)")
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(t), blit=False)
    ani.save('circular_vibration.gif', writer='pillow', fps=fps)
    plt.show()
    return ani

def Draw_freq_domain(phi, psi,R, c, m, n):
    freq_strength = collections.defaultdict(float)
    omega_dict = {}

    for m in range(0, m + 1):
        for n in range(1, n + 1):
            A_cos, B_cos, A_sin, B_sin, omega = circular_coefficients(phi, psi, R, c, m, n)
            key = np.round(omega, 8)  # avoid float precision issues
            strength = (A_cos+B_cos)**2+(A_sin+B_sin)**2
            freq_strength[key] += strength
            omega_dict[(m, n)] = omega

    # 排序频率
    freqs = np.array(list(freq_strength.keys()))
    strengths = np.array([freq_strength[f] for f in freqs])
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    strengths = strengths[idx]

    plt.figure(figsize=(8, 4))
    plt.stem(freqs, strengths)
    plt.xlabel('Frequency ω')
    plt.ylabel('Strength $A^2+B^2$')
    plt.title('Frequency Domain Strength')
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 参数设置
    R = 5.0       # 半径
    c = 1.25       # 波速
    mmax = 5      # 最大角向模式数
    nmax = 5      # 最大径向模式数
    hmax = 100
    # 初始条件：中心扰动
    alpha = jn_zeros(1,1)[-1]
    def phi(x,y): 
        r = np.sqrt(x**2 + y**2)
        return hmax*jv(1,alpha*r/R)*(x+0.0000001)/(np.sqrt(x**2+y**2)+0.0000001)  # 环形高斯分布
    psi = lambda x,y: 0.0  # 初始静止
    
    
    Draw_freq_domain(phi, psi, R ,c, mmax, nmax)
    # 计算解系数
    coeffs = circular_wave_solution(phi, psi, R, c, mmax, nmax)
    
    # 生成动画（10秒时长）
    animate_circular_vibration(coeffs, R, duration=10, fps=24)