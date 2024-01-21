import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 设置模型参数
r = np.array([0.8, 0.85, 0.95, 1.09, 0.8])      # 每个物种的内禀增长率
K = np.array([500, 800, 900, 1000, 600])    # 每个物种的环境容量

# 定义竞争模型方程
def compModel(x, t):
    dxdt = np.zeros_like(x)
    dxdt[0] = r[0]*x[0]*(1 - (x[0] + x[0]/(x[0]+x[1])*x[1] + x[0]/(x[0]+x[2])*x[2] + x[0]/(x[0]+x[3])*x[3] + x[0]/(x[0]+x[4])*x[4])/K[0])
    dxdt[1] = r[1]*x[1]*(1 - (x[1] + x[1]/(x[0]+x[1])*x[0] + x[1]/(x[2]+x[1])*x[2] + x[1]/(x[1]+x[3])*x[3] + x[1]/(x[4]+x[1])*x[4])/K[1])
    dxdt[2] = r[2]*x[2]*(1 - (x[2] + x[2]/(x[0]+x[2])*x[0] + x[2]/(x[2]+x[1])*x[1] + x[2]/(x[2]+x[3])*x[3] + x[2]/(x[2]+x[4])*x[4])/K[2])
    dxdt[3] = r[3]*x[3]*(1 - (x[3] + x[3]/(x[0]+x[3])*x[0] + x[3]/(x[3]+x[1])*x[1] + x[3]/(x[3]+x[2])*x[2] + x[3]/(x[3]+x[4])*x[4])/K[3])
    dxdt[4] = r[4]*x[4]*(1 - (x[4] + x[4]/(x[0]+x[4])*x[0] + x[4]/(x[4]+x[1])*x[1] + x[4]/(x[4]+x[2])*x[2] + x[4]/(x[4]+x[3])*x[3])/K[4])
    return dxdt

# 初始种群数量
x0 = np.array([145, 425, 515, 605, 233])

# 时间区间
t = np.linspace(0, 30, 100)

# 求解微分方程
x = odeint(compModel, x0, t)

# 绘制种群数量随时间的变化曲线
plt.plot(t, x[:, 0], 'b', label='植物1')
# plt.plot(t, x[:, 1], 'g', label='
