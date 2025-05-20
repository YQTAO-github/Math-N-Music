# 计算矩形边界条件初值问题的解
# phi(x,y),为T=0时的初始条件,psi(x,y),为T=0时的一阶导数初始条件
# 用u(x,y,t)=\Sum_{m,n}(A_{mn}cos(\omega_{mn}t)+B_{mn}sin(\omega_{mn}t))sin(m\pi x/Lx)sin(n\pi y/Ly)
# 其中\omega_{mn}=\sqrt{(m\pi/Lx)^2+(n\pi/Ly)^2}*c，Lx, Ly为矩形的长和宽

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy import integrate
from mpl_toolkits.mplot3d import Axes3D


def SolveRect_coefficient(phi, psi, Lx, Ly, c, m, n):
    """
    计算矩形边界条件初值问题的解
    :param phi: 初始条件函数
    :param psi: 一阶导数初始条件函数
    :param Lx: 矩形的长
    :param Ly: 矩形的宽
    :param c: 波速
    :param m: x方向的分量
    :param n: y方向的分量
    :return: 系数A和B
    """
    # 计算omega
    omega = np.sqrt((m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2) * c

    # 计算A和B
    # dblquad的参数依次为：被积函数, x的下界, x的上界, y的下界函数, y的上界函数
    # 注意被积函数的参数顺序为(x, y)
    A = integrate.dblquad(
        lambda y, x: phi(x, y) * np.sin(m * np.pi * x / Lx) * np.sin(n * np.pi * y / Ly),
        0, Lx,
        lambda x: 0,
        lambda x: Ly,
        epsabs=1e-10, epsrel=1e-10
    )[0] * 4 / (Lx * Ly)
        
    B = integrate.dblquad(
        lambda y, x: psi(x, y) * np.sin(m * np.pi * x / Lx) * np.sin(n * np.pi * y / Ly),
        0, Lx,
        lambda x: 0,
        lambda x: Ly,
        epsabs=1e-10, epsrel=1e-10
    )[0] * 4 / (Lx * Ly) / omega

    return A, B
    
def SolveRect(phi, psi, Lx, Ly, c, m_max, n_max):
    """
    计算矩形边界条件初值问题的解
    :param phi: 初始条件函数
    :param psi: 一阶导数初始条件函数
    :param Lx: 矩形的长
    :param Ly: 矩形的宽
    :param c: 波速
    :param m_max: x方向的分量最大值
    :param n_max: y方向的分量最大值
    :return: 系数A和B
    """
    A = np.zeros((m_max, n_max))
    B = np.zeros((m_max, n_max))

    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            A[m - 1, n - 1], B[m - 1, n - 1] = SolveRect_coefficient(phi, psi, Lx, Ly, c, m, n)

    return A, B
def SolveRect_3D_plot_gif(phi, psi, Lx, Ly, c, m_max, n_max, T_max, fps=30):
    """
    计算矩形边界条件初值问题的解，并绘制gif图像
    :param phi: 初始条件函数
    :param psi: 一阶导数初始条件函数
    :param Lx: 矩形的长
    :param Ly: 矩形的宽
    :param c: 波速
    :param m_max: x方向的分量最大值
    :param n_max: y方向的分量最大值
    :param T_max: 时间最大值
    :param fps: 帧率
    """

    A, B = SolveRect(phi, psi, Lx, Ly, c, m_max, n_max)
    # 计算时间步长
    dt = 1 / fps
    # 计算时间序列
    t = np.arange(0, T_max, dt)
    # 计算空间网格
    # x和y的网格
    x = np.linspace(0, Lx, 100)
    y = np.linspace(0, Ly, 100)
    X, Y = np.meshgrid(x, y)
    # 计算z的网格
    Z = np.zeros((len(x), len(y), len(t)))
    # 计算z的值
    for i in range(len(t)):
        for m in range(1, m_max + 1):
            for n in range(1, n_max + 1):
                omega = np.sqrt((m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2) * c
                Z[:, :, i] += A[m - 1, n - 1] * np.cos(omega * t[i]) * np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
                Z[:, :, i] += B[m - 1, n - 1] * np.sin(omega * t[i]) * np.sin(m * np.pi * X / Lx) * np.sin(n * np.pi * Y / Ly)
    # 绘制gif图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(-1, 1)
    ax.set_title('Rectangular Drum')
    ax.view_init(30, 30)
    # 绘制每一帧
    # 注意这里的t是时间序列

    def update(i):
        ax.cla()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_zlim(-1, 1)
        ax.set_title('Rectangular Drum')
        ax.view_init(30, 30)
        # 绘制当前帧
        ax.plot_surface(X, Y, Z[:, :, i], cmap='viridis')
        ax.set_zlim(-1, 1)
        ax.set_title(f'Time: {t[i]:.2f}s')
        return ax,
    ani = FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=False)
    ani.save('./Math-N-Music/Q1/rectangular_drum.gif', writer='pillow', fps=fps)
    plt.show()

    return ani  

    
# 测试
if __name__ == '__main__':

    # 矩形的长和宽
    Lx = 3.0
    Ly = 1.0

    def psi0(x, y):
        if abs(x-Lx/2) < 0.1 and abs(y-Ly/2) < 0.1:
            return 1.0
        else:
            return 0.0

    # 初始条件函数
    phi = lambda x, y: 0.0
    psi = psi0#lambda x, y: psi0

    # 波速
    c = 1.0

    # x方向的分量最大值
    m_max = 10

    # y方向的分量最大值
    n_max = 10

    # 时间最大值
    T_max = 2.0

    # 绘制gif图像
    SolveRect_3D_plot_gif(phi, psi, Lx, Ly, c, m_max, n_max, T_max)