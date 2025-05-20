import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import dblquad
from scipy.special import jn, jn_zeros
from mpl_toolkits.mplot3d import Axes3D

def circular_coefficients(phi, psi, R, c, m, n):
    """计算圆形域波动方程展开系数"""
    # 获取贝塞尔函数根
    alpha = jn_zeros(m, n)[-1]
    omega = alpha * c / R
    
    # 正交归一化系数计算
    def norm_factor(m, alpha):
        integral = (R**2/2)*(jn(m+1, alpha)**2)
        return np.pi * integral * (2 if m==0 else 1)
    
    # 模式函数模板
    def mode_func(r, theta, t=0):
        spatial = jn(m, alpha*r/R) * np.cos(m*theta)
        temporal = np.cos(omega*t) 
        return spatial * temporal
    
    # 系数A计算（位移项）
    def integrand_A(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return phi(x,y) * mode_func(r, theta) * r  # 包含雅可比行列式r
    
    A = dblquad(integrand_A, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / norm_factor(m, alpha)
    
    # 系数B计算（速度项）
    def integrand_B(theta, r):
        x, y = r*np.cos(theta), r*np.sin(theta)
        return psi(x,y) * mode_func(r, theta) * r
    
    B = dblquad(integrand_B, 0, R, 
               lambda r: 0, lambda r: 2*np.pi, 
               epsabs=1e-6)[0] / (omega * norm_factor(m, alpha))
    
    return A, B, omega

def circular_wave_solution(phi, psi, R, c, mmax, nmax):
    """构建完整波动解"""
    # 初始化系数矩阵
    coeffs = []
    for m in range(mmax+1):
        for n in range(1, nmax+1):
            A, B, omega = circular_coefficients(phi, psi, R, c, m, n)
            coeffs.append( (m, n, A, B, omega) )
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
    for (m, n, A, B, omega) in coeffs:
        alpha = jn_zeros(m, n)[-1]
        spatial = jn(m, alpha*R_grid/R) * np.cos(m*Theta_grid)
        spatial_modes.append( (A, B, omega, spatial) )
    
    # 初始化图形
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1.5, 1.5)
    
    def update(frame):
        ax.cla()
        current_time = t[frame]
        Z = np.zeros_like(X)
        
        # 叠加各模式贡献
        for (A, B, omega, spatial) in spatial_modes:
            temporal = A*np.cos(omega*current_time) + B*np.sin(omega*current_time)
            Z += temporal * spatial
        
        # 应用圆形边界
        Z[R_grid > R] = 0
        
        # 动态归一化
        max_amp = np.max(np.abs(Z))
        if max_amp > 0: Z /= max_amp
        
        # 绘制表面
        ax.plot_surface(X, Y, Z, cmap='coolwarm', rstride=2, cstride=2)
        ax.set_title(f"Circular Membrane (t={current_time:.2f}s)")
        ax.set_zlim(-1.5, 1.5)
        return ax,
    
    ani = FuncAnimation(fig, update, frames=len(t), blit=False)
    ani.save('./Math-N-Music/Q1/circular_vibration.gif', writer='pillow', fps=fps)
    plt.show()
    return ani

# 使用示例
if __name__ == "__main__":
    # 参数设置
    R = 1.0       # 半径
    c = 0.5       # 波速
    mmax = 2      # 最大角向模式数
    nmax = 2      # 最大径向模式数
    
    # 初始条件：中心扰动
    def phi(x,y): 
        r = np.sqrt(x**2 + y**2)
        return np.exp(-20*(r-0.5)**2)  # 环形高斯分布
    
    psi = lambda x,y: 0.0  # 初始静止
    
    # 计算解系数
    coeffs = circular_wave_solution(phi, psi, R, c, mmax, nmax)
    
    # 生成动画（10秒时长）
    animate_circular_vibration(coeffs, R, duration=10, fps=24)