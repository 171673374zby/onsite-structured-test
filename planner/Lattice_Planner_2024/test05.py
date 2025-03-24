import numpy as np
import math
from scipy.interpolate import interp1d
from math import sin, cos, pi

def GenerateConvexSpace(dp_speed_s, dp_speed_t, path_index2s, obs_st_s_in_set, obs_st_s_out_set,
        obs_st_t_in_set, obs_st_t_out_set, trajectory_kappa_init, max_lateral_accel):
    # 初始化上下界
    n = 17
    # s_lb = np.full((n,), -np.inf)
    # s_ub = np.full((n,), np.inf)
    # s_dot_lb = np.full((n,), -np.inf)
    # s_dot_ub = np.full((n,), np.inf)
    s_lb = np.full((n,), -100)
    s_ub = np.full((n,), 100)
    s_dot_lb = np.full((n,), -100)
    s_dot_ub = np.full((n,), 100)

    path_index2s_end_index = len(path_index2s)
    dp_speed_end_index = len(dp_speed_s)

    # 找到有效的path_index2s的末尾的位置
    for k in range(1, len(path_index2s)):
        if path_index2s[k] == 0 and path_index2s[k - 1] != 0:
            path_index2s_end_index = k - 1
            break
        path_index2s_end_index = k

    # 找到有效的dp_speed_s的位置
    # for k in range(len(dp_speed_s)):
    #     if np.isnan(dp_speed_s[k]):
    #         dp_speed_end_index = k - 1
    #         break
    # 2024.5.14改
    for k in range(len(dp_speed_s)):
        if np.isnan(dp_speed_s[k]):
            dp_speed_s[k] = dp_speed_s[k-1]
            break
    # print(path_index2s[-1])
    for i in range(n):
        # 施加车辆动力学约束
        if np.isnan(dp_speed_s[i]):
            break
        cur_s = dp_speed_s[i]
        # 通过插值找到对应的曲率
        # 此处cur_s超出了path_index2s中的s的范围
        if cur_s > path_index2s[-1]:
            break
        cur_kappa = interp1d(path_index2s[:path_index2s_end_index + 1], trajectory_kappa_init[:path_index2s_end_index + 1])(cur_s)
        # 计算最大速度
        max_speed = np.sqrt(max_lateral_accel / (abs(cur_kappa) + 1e-10))
        min_speed = 0
        s_dot_lb[i] = min_speed
        s_dot_ub[i] = max_speed
    # print(obs_st_t_in_set)
    for i in range(len(obs_st_s_in_set)):
        if np.isnan(obs_st_s_in_set[i]):
            continue
        obs_t = (obs_st_t_in_set[i] + obs_st_t_out_set[i]) / 2
        # print(obs_t)
        # print(dp_speed_t)
        # print(dp_speed_s)
        obs_s = (obs_st_s_in_set[i] + obs_st_s_out_set[i]) / 2
        obs_speed = (obs_st_s_out_set[i] - obs_st_s_in_set[i]) / (obs_st_t_out_set[i] - obs_st_t_in_set[i])
        # obs_t = (obs_st_t_in_set + obs_st_t_out_set) / 2
        # obs_s = (obs_st_s_in_set + obs_st_s_out_set) / 2
        # obs_speed = (obs_st_s_out_set - obs_st_s_in_set) / (obs_st_t_out_set - obs_st_t_in_set)
        dp_s = interp1d([0] + dp_speed_t[:dp_speed_end_index + 1], [0] + dp_speed_s[:dp_speed_end_index + 1])(obs_t)

        # t_lb_index = next(
        #     j for j in range(len(dp_speed_t)) if dp_speed_t[j] <= obs_st_t_in_set[i] < dp_speed_t[j + 1])
        # t_ub_index = next(
        #     j for j in range(len(dp_speed_t)) if dp_speed_t[j] <= obs_st_t_out_set[i] < dp_speed_t[j + 1])
        # t_lb_index = max(t_lb_index - 2, 3)
        # t_ub_index = min(t_ub_index + 2, dp_speed_end_index)
        t_lb_index = 0
        t_ub_index = 0
        for j in range(len(dp_speed_t) - 1):
            # for i in range(len(obs_st_t_in_set)):
            if dp_speed_t[j] <= obs_st_t_in_set[i] < dp_speed_t[j + 1]:
                t_lb_index = j
        for k in range(len(dp_speed_t) - 1):
            if dp_speed_t[k] <= obs_st_t_out_set[i] < dp_speed_t[k + 1]:
                t_ub_index = k
    # t_lb_index = 0
    # t_ub_index = 0
    # for j in range(len(dp_speed_t) - 1):
    #     if dp_speed_t[j] <= obs_st_t_in_set < dp_speed_t[j + 1]:
    #         t_lb_index = j
    # for k in range(len(dp_speed_t) - 1):
    #     if dp_speed_t[k] <= obs_st_t_out_set < dp_speed_t[k + 1]:
    #         t_ub_index = k

    # if obs_s > dp_s:
    #     for m in range(t_lb_index, t_ub_index + 1):
    #         dp_t = dp_speed_t[m]
    #         s_ub[m] = min(s_ub[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
    # else:
    #     for m in range(t_lb_index, t_ub_index + 1):
    #         dp_t = dp_speed_t[m]
    #         s_lb[m] = max(s_lb[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
        if obs_s > dp_s:
            for m in range(t_lb_index, t_ub_index + 1):
                dp_t = dp_speed_t[m]
                s_ub[m] = min(s_ub[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
        else:
            for m in range(t_lb_index, t_ub_index + 1):
                dp_t = dp_speed_t[m]
                s_lb[m] = max(s_lb[m], obs_st_s_in_set[i] + obs_speed * (dp_t - obs_st_t_in_set[i]))
        # 降低上限的数值
    for i in range(len(s_ub)):
        s_ub[i] = s_ub[i] - 7
    for i in range(len(s_lb)):
        s_lb[i] = s_lb[i] + 4
    return s_lb, s_ub, s_dot_lb, s_dot_ub

def path_index2s(path):
    # 计算投影信息
    pathpoints_index2s = np.zeros(len(path[0]))
    for i in range(1, len(path[0])):
        pathpoints_index2s[i] = math.sqrt(
            (path[0][i] - path[0][i-1]) ** 2 + (path[1][i] - path[1][i-1]) ** 2) + pathpoints_index2s[i - 1]
    return pathpoints_index2s





