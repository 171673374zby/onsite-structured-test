import numpy as np
# from qpsolvers import solve_qp
# import cvxopt

import math
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from math import sin, cos, pi
from test03 import cal_refer_path_info
# 计算速度规划开始时s方向上的速度和加速度信息
def cal_plan_start_s_dotanddot2(controller,path):
    plan_start_v = controller.observation.ego_info.v
    plan_start_a = controller.observation.ego_info.a
    plan_start_yaw = controller.observation.ego_info.yaw
    plan_start_vx = plan_start_v * cos(math.radians(plan_start_yaw))
    plan_start_vy = plan_start_v * sin(math.radians(plan_start_yaw))
    x_new, y_new, curvatures, plan_start_heading = cal_refer_path_info(controller, path)
    tor = np.array([cos(math.radians(plan_start_heading)), sin(math.radians(plan_start_heading))])
    nor = np.array([-sin(math.radians(plan_start_heading)), cos(math.radians(plan_start_heading))])
    a_tor = plan_start_a * tor
    a_nor = (plan_start_v ** 2) * curvatures[0] * nor
    plan_start_ax = a_tor[0] + a_nor[0]
    plan_start_ay = a_tor[1] + a_nor[1]
    plan_start_s_dot = np.dot(tor, np.array([plan_start_vx, plan_start_vy]))
    plan_start_s_dot2 = np.dot(tor, np.array([plan_start_ax, plan_start_ay]))
    return plan_start_s_dot, plan_start_s_dot2
#
def SpeedPlanningwithQuadraticPlanning(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t, s_lb, s_ub, s_dot_lb, s_dot_ub):
    # w_cost_s_dot2 = 10
    w_cost_s_dot2 = 10
    w_cost_v_ref = 50
    # w_cost_jerk = 500
    w_cost_jerk = 500
    speed_reference = 15
    # 由于dp的结果未必是16，该算法将计算dp_speed_end到底是多少
    # dp_speed_end = 16
    dp_speed_end = 16
    for i in range(len(dp_speed_s)):
        if np.isnan(dp_speed_s[i]):
            dp_speed_end = i - 1
            break

    # 初始化输出
    # n = 17
    n = 17
    qp_s_init = np.ones(n) * np.nan
    qp_s_dot_init = np.ones(n) * np.nan
    qp_s_dot2_init = np.ones(n) * np.nan
    relative_time_init = np.ones(n) * np.nan
    s_end = dp_speed_s[dp_speed_end - 1]
    recommend_T = dp_speed_t[dp_speed_end - 1]
    qp_size = dp_speed_end + 1

    # 连续性约束矩阵初始化
    Aeq = np.zeros((3 * qp_size, 2 * qp_size - 2))
    beq = np.zeros(2 * qp_size - 2)
    lb = np.ones(3 * qp_size)
    ub = np.ones(3 * qp_size)

    # 计算采样间隔时间dt
    # dt = recommend_T / dp_speed_end
    dt = 0.1
    A_sub = np.array([[1, 0],
                      [dt, 1],
                      [(1 / 3) * dt ** 2, (1 / 2) * dt],
                      [-1, 0],
                      [0, -1],
                      [(1 / 6) * dt ** 2, dt / 2]])

    for i in range(1, qp_size):
        Aeq[3 * i - 2 - 1:3 * i + 3, 2 * i - 1 - 1:2 * i] = A_sub

    A = np.array(np.zeros((qp_size - 1, 3 * qp_size)))
    b = np.zeros((qp_size - 1,1))

    for i in range(1, qp_size):
        A[i - 1, 3 * i - 3] = 1
        A[i - 1, 3 * i + 1 - 1] = -1

    for i in range(2, qp_size + 1):
        lb[3 * i - 2 - 1] = s_lb[i - 1 - 1]
        lb[3 * i - 1 - 1] = s_dot_lb[i - 1 - 1]
        lb[3 * i - 1] = -6
        ub[3 * i - 2 - 1] = s_ub[i - 1 - 1]
        ub[3 * i - 1 - 1] = s_dot_ub[i - 1 - 1]
        ub[3 * i - 1] = 4

    lb[0] = 0
    lb[1] = plan_start_s_dot
    lb[2] = plan_start_s_dot2
    ub[0] = lb[0]
    ub[1] = lb[1]
    ub[2] = lb[2]

    # 加速度代价 jerk代价 以及推荐速度代价
    A_s_dot2 = np.zeros((3 * qp_size, 3 * qp_size))
    A_jerk = np.zeros((3 * qp_size, qp_size - 1))
    A_ref = np.zeros((3 * qp_size, 3 * qp_size))
    A4_sub = np.array([[0], [0], [1], [0], [0], [-1]])

    for i in range(1, qp_size + 1):
        # A_s_dot2[3 * i - 1, 3 * i - 1] = 1
        A_s_dot2[3 * i - 1, 3 * i - 1] = 0.3
        A_ref[3 * i - 1 - 1, 3 * i - 1 - 1] = 1

    for i in range(1, qp_size):
        A_jerk[3 * i - 2 - 1:3 * i + 3, i - 1:i] = A4_sub

    # 生成H f
    # H = np.array(np.zeros(51,51))
    H = w_cost_s_dot2 * (np.dot(A_s_dot2, A_s_dot2.T)) + w_cost_jerk * (np.dot(A_jerk, A_jerk.T)) + w_cost_v_ref * (
        np.dot(A_ref, A_ref.T))

    H = 2 * H
    H = np.array(H)
    f = np.zeros((3 * qp_size, 1))
    #
    for i in range(1, qp_size + 1):
        f[3 * i - 1 - 1] = -2 * w_cost_v_ref * speed_reference
    # 设置初始猜测值
    x0 = np.zeros((3 * qp_size, 1))

    A_eq = Aeq.T
    b_eq = beq.T
    l_b = lb.T
    u_b = ub.T
    # 定义约束条件
    constraints = ({'type': 'eq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[0]},
                   {'type': 'ineq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[1]},
                   {'type': 'ineq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[2]})

    # 最小化目标函数
    result = minimize(objective, x0, args=(H, f), constraints = constraints)
    # result = minimize(objective,  args=(H, f), constraints = constraints)
    # 输出结果
    X = result.x

    for i in range(1, qp_size + 1):
        qp_s_init[i - 1] = X[3 * i - 2 - 1]
        qp_s_dot_init[i - 1] = X[3 * i - 1 - 1]
        qp_s_dot2_init[i - 1] = X[3 * i - 1]
        relative_time_init[i - 1] = (i - 1) * dt

    return qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init
# 定义目标函数
def objective(X, H, f):
    return 0.5 * np.dot(X.T, np.dot(H, X)) + np.dot(f.T, X)

# 定义约束条件
def constraint(X, A_eq, b_eq, l_b, u_b):
    return np.dot(A_eq, X) - b_eq, X - l_b, u_b - X

# # 设置初始猜测值
# x0 = np.zeros((3 * qp_size, 1))
#
# A_eq = Aeq.T
# b_eq = beq.T
# l_b = lb.T
# u_b = ub.T
# # 定义约束条件
# constraints = ({'type': 'eq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[0]},
#                {'type': 'ineq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[1]},
#                {'type': 'ineq', 'fun': lambda x: constraint(x, A_eq, b_eq, l_b, u_b)[2]})
#
# # 最小化目标函数
# result = minimize(objective, x0, args=(H, f), constraints=constraints)
#
# # 输出结果
# X = result.x
#
# for i in range(1, qp_size + 1):
#     qp_s_init[i - 1] = X[3 * i - 2 - 1]
#     qp_s_dot_init[i - 1] = X[3 * i - 1 - 1]
#     qp_s_dot2_init[i - 1] = X[3 * i - 1]
#     relative_time_init[i - 1] = (i - 1) * dt


# 输出结果
# print("qp_s_init:", qp_s_init)
# print("qp_s_dot_init:", qp_s_dot_init)
# print("qp_s_dot2_init:", qp_s_dot2_init)
# print("relative_time_init:", relative_time_init)
# plt.plot(relative_time_init, qp_s_init, marker='o', linestyle='-')
# plt.xlabel('relative_time_init')
# plt.ylabel('qp_s_init')
# plt.title('qp_s_init vs Time')
# plt.grid(True)
# plt.show()
#
# plt.plot(relative_time_init, qp_s_dot_init, marker='o', linestyle='-')
# plt.xlabel('relative_time_init')
# plt.ylabel('qp_s_dot_init')
# plt.title('qp_s_dot_init vs Time')
# plt.grid(True)
# plt.show()

# # 定义目标函数
# def objective(X):
#     return 0.5 * np.dot(X, np.dot(H, X)) + np.dot(f, X)
#
# # 定义约束条件
# constraints = ({'type': 'eq', 'fun': lambda x: np.dot(Aeq.T, x) - beq},
#                {'type': 'ineq', 'fun': lambda x: lb - x},
#                {'type': 'ineq', 'fun': lambda x: x - ub})
#
# # 初始猜测值
# x0 = np.ones(3 * qp_size)
# B = np.array(np.ones((16, 51)))
# # 最小化目标函数
# result = minimize(objective, x0, constraints=constraints)
# # 输出结果
# X = result.x
# plan_start_s_dot = 25
# plan_start_s_dot2 = 1.5
# dp_speed_s = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 16,  17.5, 19]
# dp_speed_t = [0.5, 1, 1.5, 2, 2.5, 3,  3.5, 4, 4.5, 5, 5.5, 6,  6.5, 7, 7.5, 8]
# # s_lb = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
# # s_ub = [np.inf, np.inf, np.inf, np.inf, 10.0, 15.0, 20.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
# s_lb = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]
# s_ub = [100, 100, 100, 100, 10.0, 15.0, 20.0, 100, 100, 100, 100, 100, 100, 100, 100, 100]
# s_dot_lb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# s_dot_ub = [6.42941527, 5.86257287, 5.47956303, 5.21217466, 5.02086594, 4.88428515, 4.78993765, 4.73022833, 4.70110072, 4.69784359, 4.71881191, 4.76347107, 4.83307134, 4.98717047, 5.20922827, 5.518631]