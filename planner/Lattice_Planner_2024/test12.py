import math
# def is_target_in_front(x_self, y_self, theta_self, x_target, y_target):
#     # 计算目标车辆相对于自车的位置
#     delta_x = x_target - x_self
#     delta_y = y_target - y_self
#
#     # 计算目标车辆相对于自车的方向角度
#     relative_angle = math.atan2(delta_y, delta_x)
#
#     # 计算相对角度差
#     angle_diff = relative_angle - theta_self
#
#     # 将角度差调整到 -pi 到 pi 的范围内
#     while angle_diff > math.pi:
#         angle_diff -= 2 * math.pi
#     while angle_diff < -math.pi:
#         angle_diff += 2 * math.pi
#
#     # 判断目标车辆是否在自车的前方（角度差在 -pi/2 到 pi/2 之间）
#     if -math.pi / 2 < angle_diff < math.pi / 2:
#         return True
#     else:
#         return False
#
#
# # 示例数据
# x_self = 0
# y_self = 0
# theta_self = 4.468
# x_target = 1
# y_target = 1
#
# # 判断目标车辆是否在自车的前方
# if is_target_in_front(x_self, y_self, theta_self, x_target, y_target):
#     print("目标车辆在自车的前方")
# else:
#     print("目标车辆在自车的后方")

#####
# import matplotlib.pyplot as plt
#
# # 示例数据
# obs_s_in = [0]
# obs_s_out = [0.21180074494525858]
# obs_t_in = [1.456120932916504]
# obs_t_out = [1.5]
#
# # 绘制图像
# plt.figure(figsize=(8, 6))
#
# # 绘制obs_s_in
# plt.plot(obs_t_in, obs_s_in, 'ro', label='obs_s_in')
# plt.text(obs_t_in[0], obs_s_in[0], ' obs_s_in', fontsize=12, va='bottom')
#
# # 绘制obs_s_out
# plt.plot(obs_t_out, obs_s_out, 'bo', label='obs_s_out')
# plt.text(obs_t_out[0], obs_s_out[0], ' obs_s_out', fontsize=12, va='bottom')
#
# # 设置横坐标和纵坐标范围
# plt.xlim(0, 1.7)
# plt.ylim(0, 19.5)
#
# # 添加标题和标签
# plt.title('obs_s_in and obs_s_out')
# plt.xlabel('t')
# plt.ylabel('s')
# plt.legend()
#
# # 显示图像
# plt.grid(True)
# plt.show()
#####
import numpy as np
from scipy.interpolate import interp1d

# 示例数据
dp_speed_t = [0.,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6]  # 时间
dp_speed_s = [0.6, 1.,  1.2, 1.4, 1.6, 1.8, 2.,  2.2, 2.4, 2.6, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8, 2.8]  # 速度

# 观测时间
obs_t = 0.75
dp_speed_end_index = 17
# 创建线性插值函数
dp_s = interp1d([0] + dp_speed_t[:dp_speed_end_index + 1], [0] + dp_speed_s[:dp_speed_end_index + 1])(obs_t)

dp_speed_t = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
print(dp_speed_t)
