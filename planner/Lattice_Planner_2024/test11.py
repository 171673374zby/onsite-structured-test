import numpy as np
def dynamic_programming(prev_timestamp, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed,
                        reference_speed_unlimit, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list):
    # 声明 st 代价矩阵，表示从起点开始到(i, j)点的最小代价为 dp_st_cost(i, j)

    s_list = np.concatenate([
        np.arange(0, 3, 0.2),  # [0:0.2:3]
        np.arange(4, 8, 1),  # [4:1:8]
        np.arange(9, 19.5, 1.5)  # [9:1.5:16.5]
    ])

    t_list = np.arange(0, 1.7, 0.1)
    # s_list = np.concatenate([np.arange(0, 5, 0.5), np.arange(5.5, 15, 1), np.arange(16, 30, 1.5), np.arange(32, 55, 2.5)])
    # t_list = np.arange(0.5, 8.5, 0.5)
    m = len(s_list)
    n = len(t_list)

    dp_st_cost = np.ones((m, n)) * np.inf  # 首先初始化为无穷大
    dp_st_s_dot = np.zeros((m, n))  # 表示从起点开始到(i, j)点的最优路径的末速度
    dp_st_node = np.zeros((m, n))  # 需要一个矩阵保持最优路径的前一个节点方便回溯
    for i in range(len(s_list)):
        dp_st_cost[i, 0] = calc_dp_cost(0, 0, i, 1, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                                         w_cost_ref_speed, reference_speed_unlimit, w_cost_accel, w_cost_obs,
                                         plan_start_s_dot, s_list, t_list, dp_st_s_dot, prev_timestamp)
        s_end, t_end = calc_st_coordinate(i, 1, s_list, t_list)
        dp_st_s_dot[i, 0] = s_end / t_end
    for i in range(1, len(t_list)):
        for j in range(len(s_list)):
            cur_row = j
            cur_col = i
            if 5 < cur_row < 19:
                for k in range(cur_row - 5, cur_row + 6):
                    pre_row = k
                    pre_col = i - 1
                    cost_temp = calc_dp_cost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                              obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed, reference_speed_unlimit,
                                              w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot,prev_timestamp)
                    if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                        dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                        s_start, t_start = calc_st_coordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = calc_st_coordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                        dp_st_node[cur_row, cur_col] = pre_row
            if cur_row <= 5:
                for k in range(0, cur_row + 6):
                    pre_row = k
                    pre_col = i - 1
                    cost_temp = calc_dp_cost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                             obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed,
                                             reference_speed_unlimit,
                                             w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot,prev_timestamp)
                    if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                        dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                        s_start, t_start = calc_st_coordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = calc_st_coordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                        dp_st_node[cur_row, cur_col] = pre_row
            if cur_row >= 19:
                for k in range(cur_row - 5, len(s_list)):
                    pre_row = k
                    pre_col = i - 1
                    cost_temp = calc_dp_cost(pre_row, pre_col, cur_row, cur_col, obs_st_s_in_set, obs_st_s_out_set,
                                             obs_st_t_in_set, obs_st_t_out_set, w_cost_ref_speed,
                                             reference_speed_unlimit,
                                             w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot,prev_timestamp)
                    if cost_temp + dp_st_cost[pre_row, pre_col] < dp_st_cost[cur_row, cur_col]:
                        dp_st_cost[cur_row, cur_col] = cost_temp + dp_st_cost[pre_row, pre_col]
                        s_start, t_start = calc_st_coordinate(pre_row, pre_col, s_list, t_list)
                        s_end, t_end = calc_st_coordinate(cur_row, cur_col, s_list, t_list)
                        dp_st_s_dot[cur_row, cur_col] = (s_end - s_start) / (t_end - t_start)
                        dp_st_node[cur_row, cur_col] = pre_row
    dp_speed_s = np.ones(len(t_list)) * np.nan
    dp_speed_t = np.copy(dp_speed_s)
    min_cost = np.inf
    min_row = np.inf
    min_col = np.inf
    for i in range(len(s_list)):
        if dp_st_cost[i, -1] <= min_cost:
            min_cost = dp_st_cost[i, -1]
            min_row = i
            min_col = len(t_list) - 1
    for j in range(len(t_list)):
        if dp_st_cost[0, j] <= min_cost:
            min_cost = dp_st_cost[0, j]
            min_row = 0
            min_col = j
    dp_speed_s[min_col], dp_speed_t[min_col] = calc_st_coordinate(min_row, min_col, s_list, t_list)
    while min_col != 0:
        pre_row = dp_st_node[int(min_row), int(min_col)]
        pre_col = min_col - 1
        dp_speed_s[pre_col], dp_speed_t[pre_col] = calc_st_coordinate(pre_row, pre_col, s_list, t_list)
        min_row = pre_row
        min_col = pre_col
        # 2024.5.14改
        dp_speed_t = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.,  1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    return dp_speed_s, dp_speed_t

def calc_st_coordinate(row, col, s_list, t_list):
# 计算矩阵节点的 s t 坐标
    m = len(s_list)
    s_value = s_list[int(m - row - 1)]
    t_value = t_list[col]
    return s_value, t_value

def calc_collision_cost(w_cost_obs, min_dis):
    if abs(min_dis) < 0.5:
        collision_cost = w_cost_obs
    # elif 0.5 < abs(min_dis) < 1.5:
    elif 0.5 < abs(min_dis) < 1.5:
        collision_cost = w_cost_obs ** ((0.5 - abs(min_dis)) + 1)
    else:
        collision_cost = 0
    return collision_cost

def calc_obs_cost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set, w_cost_obs):
    # 计算边的障碍物代价
    obs_cost = 0
    n = 5  # 采样点的个数
    dt = (t_end - t_start) / (n - 1)  # 采样时间间隔
    k = (s_end - s_start) / (t_end - t_start)  # 边的斜率
    for i in range(1, n + 1):
        t = t_start + (i - 1) * dt  # 计算采样点的坐标
        s = s_start + k * (i - 1) * dt
        for j in range(len(obs_st_s_in_set)):
            if np.isnan(obs_st_s_in_set[j]).any():
                continue
            vector1 = np.array([obs_st_s_in_set[j], obs_st_t_in_set[j]]) - np.array([s, t])
            vector2 = np.array([obs_st_s_out_set[j], obs_st_t_out_set[j]]) - np.array([s, t])
        # vector1 = np.array([obs_st_s_in_set, obs_st_t_in_set]) - np.array([s, t])
        # vector2 = np.array([obs_st_s_out_set, obs_st_t_out_set]) - np.array([s, t])
            vector3 = vector2 - vector1
            dis1 = np.sqrt(np.dot(vector1, vector1))
            dis2 = np.sqrt(np.dot(vector2, vector2))
            dis3 = abs(vector1[0] * vector3[1] - vector1[1] * vector3[0]) / np.sqrt(np.dot(vector3, vector3))
            if (np.dot(vector1, vector3) > 0 and np.dot(vector2, vector3) > 0) or (np.dot(vector1, vector3) < 0 and np.dot(vector2, vector3) < 0):
                min_dis = min(dis1, dis2)
            else:
                min_dis = dis3
            # print("min_dis:", min_dis)
            obs_cost += calc_collision_cost(w_cost_obs, min_dis)
    return obs_cost

def calc_dp_cost(row_start, col_start, row_end, col_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set, obs_st_t_out_set,
                 w_cost_ref_speed, reference_speed, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, dp_st_s_dot,prev_timestamp):
    # 计算链接两个节点之间边的代价
    s_end, t_end = calc_st_coordinate(row_end, col_end, s_list, t_list)
    # 规定起点的行列号为0
    if row_start == 0:
        s_start, t_start = 0, 0
        s_dot_start = plan_start_s_dot
    else:
        s_start, t_start = calc_st_coordinate(row_start, col_start, s_list, t_list)
        s_dot_start = dp_st_s_dot[row_start][col_start]
    cur_s_dot = (s_end - s_start) / (t_end - t_start)
    cur_s_dot2 = (cur_s_dot - s_dot_start) / (t_end - t_start)
    # 计算推荐速度代价
    cost_ref_speed = w_cost_ref_speed * (cur_s_dot - reference_speed) ** 2
    # 计算加速度代价，这里注意，加速度不能超过车辆动力学上下限
    if 4 < cur_s_dot2 < -6:
        cost_accel = w_cost_accel * cur_s_dot2 ** 2
    else:
        # 超过车辆动力学限制，代价会增大很多倍
        # cost_accel = 100000 * w_cost_accel * cur_s_dot2 ** 2
        cost_accel = 10 * w_cost_accel * cur_s_dot2 ** 2
    cost_obs = calc_obs_cost(s_start, t_start, s_end, t_end, obs_st_s_in_set, obs_st_s_out_set, obs_st_t_in_set,
                              obs_st_t_out_set, w_cost_obs)
    cost = cost_obs + cost_accel + cost_ref_speed
    # print("prev_timestamp:", prev_timestamp)
    # print("cost_obs:", cost_obs)
    # print("cost_accel:", cost_accel)
    # print("cost_ref_speed:", cost_ref_speed)
    # print("1")
    return cost
