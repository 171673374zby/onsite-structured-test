import numpy as np
import math
from scipy.interpolate import splprep, splev
from math import sin, cos, pi

# qp_s= [-4.98751951e-07, 4.55401560e-01, 9.08226893e-01, 1.35666580e+00,
#                 1.79967858e+00, 2.23684759e+00, 2.66824636e+00, 3.09432322e+00,
#                 3.51579737e+00, 3.93356484e+00, 4.34861233e+00, 4.76193571e+00,
#                 5.17446121e+00, 5.58696664e+00, 6.00000051e+00]
# qp_t = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_qp(s0, s1, t0, t1, qp_s_init, qp_t_init):
    plt.scatter([t0, t1], [s0, s1], color='red')  # 添加额外数据点
    plt.plot(qp_t_init, qp_s_init, linestyle='--')  # 绘制新的数据
    plt.xlabel('qp_t')
    plt.ylabel('qp_s')
    plt.title('Plot of qp_s and qp_t')
    plt.grid(True)
    plt.legend(['Original Data', 'Extra Data'])
    plt.show()

# 示例数组

s0 = 4.278
s1 = 6.130
t0 = 0.0
t1 = 1.5

qp_s_init = [-2.10627621e-05, 1.99815239e-01, 3.89794380e-01, 5.64788660e-01,
             7.23687247e-01, 8.67846373e-01, 9.99998721e-01, 1.12344427e+00,
             1.24151271e+00, 1.35739411e+00, 1.47410654e+00, 1.59445422e+00,
             1.72094665e+00, 1.85566150e+00, 2.00002330e+00, 2.15451132e+00,
             2.31862541e+00]

qp_t_init = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
             1.4, 1.5, 1.6]

# 绘制图像
plot_qp(s0, s1, t0, t1, qp_s_init, qp_t_init)

