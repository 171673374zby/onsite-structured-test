import numpy as np
import math
from scipy.interpolate import splprep, splev
from math import sin, cos, pi


# obs_list = [[[973,1002,1.1],[974,1001,1.2],[975,1000,1.3]],[[985,1035,2.3],[986,1036,2.4],[987,1037,2.5]]]
# frenet_path_x = [973.1237260494466, 973.6800946163513, 974.2375460461392, 974.7962186146368, 975.3559109282381, 975.9160828060287, 976.4758863938275, 977.0342200291102, 977.5897979143815, 978.1412291215818, 978.6870999687667, 979.2260544613995, 979.7568683002731, 980.278512905942, 980.7902069412085, 981.2914538559089, 981.7820649561947, 982.2621683479271, 982.7322047739615, 983.1929118347491, 983.6452983514662, 984.0906107221654, 984.5302930709735, 984.9659428439651, 985.3992633114495, 985.8320142403712, 986.2659618400523, 986.7028289863279, 987.1442467069396, 987.5917079647028, 988.0465248905414, 988.5097907696966, 988.9823482348323, 989.4647652262751, 989.9573202969494, 990.4599987255745, 990.9725006231752, 991.4942617562484, 992.0244871650682, 992.5621968486532, 993.1062818597165, 993.6555681593654, 994.2088845847385, 994.7651303401411, 995.3233365722458, 995.8827158408573, 996.4426926189705, 997.0029072822953, 997.563185287887]
# frenet_path_y = [1001.8206435902564, 1001.7545864333791, 1001.698596761935, 1001.6568294756702, 1001.6328465796026, 1001.6296285400172, 1001.6495808737774, 1001.6945405746633, 1001.7657864887227, 1001.8640567607738, 1001.9895752225448, 1002.1420872774169, 1002.320904617061, 1002.5249571001987, 1002.7528494087095, 1003.0029197040238, 1003.2732974312313, 1003.5619576226158, 1003.8667694781578, 1004.1855375788916, 1004.5160347496412, 1004.8560262667428, 1005.2032857512814, 1005.5556036599869, 1005.9107897581538, 1006.2666713168032, 1006.621089012968, 1006.971892625704, 1007.3169386120844, 1007.6540915190873, 1007.98123094209, 1008.2962653840673, 1008.5971539113553, 1008.8819359589916, 1009.1487690377235, 1009.3959734735979, 1009.6220827185365, 1009.8258972641527, 1010.0065398329656, 1010.1635093694641, 1010.296731454365, 1010.4066031432162, 1010.4940308787278, 1010.5604610004067, 1010.6079033882656, 1010.6389497983265, 1010.6567893004291, 1010.6652236921982, 1010.6686855661045]
# match_point_index_set = []
# match_point_index = []
# increase_count = 0
# obs_proj_x_list = []
# obs_all_proj_x_list = []
def cal_refer_path_info(path):
    x = []
    y = []
    for i in range(len(path[0])):
        x.append(path[0][i])
        y.append(path[1][i])
    x = np.array(path[0])
    y = np.array(path[1])
    tck, u = splprep([x, y], s = 0)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck)

    # 计算导数
    x_prime = np.gradient(x_new)
    y_prime = np.gradient(y_new)

    # 计算二阶导数
    x_double_prime = np.gradient(x_prime)
    y_double_prime = np.gradient(y_prime)

    # 计算曲率和航向角
    numerator = np.abs(x_prime * y_double_prime - y_prime * x_double_prime)
    denominator = (x_prime ** 2 + y_prime ** 2) ** (3 / 2)
    curvatures = numerator / denominator

    # 将航向角从弧度转换为角度
    headings = np.arctan2(y_prime, x_prime)
    headings_degrees = np.degrees(headings)
    # 角度值转弧度值
    headings_degrees_rad = np.radians(headings_degrees)
    return x_new, y_new, curvatures, headings_degrees_rad

# 从这里开始定义一个新的函数
def obs_process(path_index2s, obs_list, frenet_path_x, frenet_path_y, headings_degrees):
    match_point_index_set = []
    match_point_index = []
    obs_all_proj_list = []
    increase_count = 0
    # 第一个循环为障碍物的数量
    # for i in range(len(obs_list)):
    # 第二个循环为每个障碍物的所有时刻的
    for i in range(len(obs_list)):
        for j in range(len(obs_list[i])):
            min_distance = float('inf')
            for k in range(len(frenet_path_x)):
                distance = ((obs_list[i][j][0]) - frenet_path_x[k])**2 + ((obs_list[i][j][1]) - frenet_path_y[k])**2
                if distance < min_distance:
                    min_distance = distance
                    match_point_index_set.append(k)
                    increase_count = 0
                else:
                    increase_count += 1
                # if increase_count >= 50:
                #     break
            match_point_index.append(match_point_index_set[-1])
    # obs_proj_x_list.append(match_point_index)
    # obs_all_proj_x_list.append([obs_proj_x_list])
    # n的值为每个障碍物有多少个时刻的信息
    n = 4
    obs_all_proj_list = [match_point_index[i:i+n] for i in range(0, len(match_point_index), n)]
    points_index2s = np.zeros(len(frenet_path_x))
    for i in range(1,len(frenet_path_x)):
        points_index2s[i] = math.sqrt((frenet_path_x[i] - frenet_path_x[i - 1]) ** 2 + (frenet_path_y[i] - frenet_path_y[i - 1]) ** 2) + points_index2s[i - 1]

    s_set = []
    l_set = []
    # 计算[[0, 35, 40], [0, 39, 36]]中[0,35,40]中每个点的s和l
    for i in range(len(obs_all_proj_list)):
        for j in range(len(obs_all_proj_list[i])):
            s_set.append(points_index2s[obs_all_proj_list[i][j]])
            n_r = np.array([-sin(headings_degrees[obs_all_proj_list[i][j]]), cos(headings_degrees[obs_all_proj_list[i][j]])])
            r_h = np.array([obs_list[i][j][0], obs_list[i][j][1]])
            r_r = np.array([frenet_path_x[obs_all_proj_list[i][j]], frenet_path_y[obs_all_proj_list[i][j]]])
            # matrix = [list(row) for row in zip(*(r_h - r_r))]
            # l_set.append(np.matmul(matrix, n_r))
            l_set.append(np.dot((r_h - r_r), n_r))

    # n = 3
    s_set = [s_set[i:i+n] for i in range(0, len(s_set), n)]
    l_set = [l_set[i:i+n] for i in range(0, len(l_set), n)]
    # print(s_set)
    # print(l_set)

    #[x y t] obs_list = [[[973,1002,1.1],[974,1001,1.2],[975,1000,1.3]],[[985,1006,2.3],[986,1007,2.4],[987,1008,2.5]]]
    SLTofsingle_obs_list = []
    list = []
    for j in range(len(s_set)):
        for i in range(len(s_set[0])):
            SLTofsingle_obs_list.append([s_set[j][i], l_set[j][i], obs_list[j][i][2]])

    # 将障碍物以['障碍物1'：[[s0,l0,t0],[s1,l1,t1]],‘障碍物2’：[[s0,l0,t0],[s1,l1,t1]] ]形式表示
    SLTofsingle_obs_list = [SLTofsingle_obs_list[i:i+n] for i in range(0, len(SLTofsingle_obs_list), n)]

    message_of_stl = []
    for i in range(len(SLTofsingle_obs_list)):
        sub_list = SLTofsingle_obs_list[i]
        sl_fit = []
        for j in range(len(sub_list)):
            # sl_fit = []
            if abs(sub_list[j][1]) < 1.9:
                sl_fit.append([sub_list[j][0], sub_list[j][1], sub_list[j][2]])
            else:
                continue
        if not sl_fit:
            continue
        else:
            message_of_stl.append(sl_fit)
    obs_s_in_set = []
    obs_s_out_set = []
    obs_t_in_set = []
    obs_t_out_set = []
    # print("message_of_stl:",message_of_stl)
    # obs_to_consider = True
    # for sub_list in message_of_stl:
    #     obs_s_in_set.append(sub_list[0][0])
    #     obs_s_out_set.append(sub_list[-1][0])
    #     obs_t_in_set.append(sub_list[0][2])
    #     obs_t_out_set.append(sub_list[-1][2])
    # # 用于收集需要移除的索引
    # indexes_to_remove = []
    #
    # # 第一个循环：针对 obs_t_in_set 和 obs_t_out_set
    # for i in range(len(obs_t_in_set)):
    #     if obs_t_in_set[i] == obs_t_out_set[i]:
    #         if obs_t_in_set[i] == 0:
    #             indexes_to_remove.append(i)
    #         else:
    #             obs_t_in_set[i] -= 0.1
    #
    # # 移除需要删除的元素
    # for index in sorted(indexes_to_remove, reverse=True):
    #     obs_s_in_set.pop(index)
    #     obs_s_out_set.pop(index)
    #     obs_t_in_set.pop(index)
    #     obs_t_out_set.pop(index)
    #
    # # 第二个循环：针对 obs_s_in_set 和 obs_s_out_set
    # index = 0
    # while index < len(obs_s_in_set):
    #     if obs_s_in_set[index] == obs_s_out_set[index] and obs_s_in_set[index] > (path_index2s[-1] - 1):
    #         obs_s_in_set.pop(index)
    #         obs_s_out_set.pop(index)
    #         obs_t_in_set.pop(index)
    #         obs_t_out_set.pop(index)
    #     else:
    #         index += 1
    #
    # index01 = 0
    # while index01 < len(obs_s_in_set):
    #     if obs_s_in_set[index01] == 0:
    #         obs_s_in_set.pop(index01)
    #         obs_s_out_set.pop(index01)
    #         obs_t_in_set.pop(index01)
    #         obs_t_out_set.pop(index01)
    #     else:
    #         index01 += 1
    # if obs_s_in_set:
    #     obs_to_consider = True
    # else:
    #     obs_to_consider = False
    if message_of_stl:
        obs_to_consider = True
        for sub_list in message_of_stl:
            obs_s_in_set.append(sub_list[0][0])
            obs_s_out_set.append(sub_list[-1][0])
            obs_t_in_set.append(sub_list[0][2])
            obs_t_out_set.append(sub_list[-1][2])
        # 用于收集需要移除的索引
        indexes_to_remove = []

        # 第一个循环：针对 obs_t_in_set 和 obs_t_out_set
        for i in range(len(obs_t_in_set)):
            if obs_t_in_set[i] == obs_t_out_set[i]:
                if obs_t_in_set[i] == 0:
                    indexes_to_remove.append(i)
                else:
                    obs_t_in_set[i] -= 0.1

        # 移除需要删除的元素
        for index in sorted(indexes_to_remove, reverse=True):
            obs_s_in_set.pop(index)
            obs_s_out_set.pop(index)
            obs_t_in_set.pop(index)
            obs_t_out_set.pop(index)

        # 第二个循环：针对 obs_s_in_set 和 obs_s_out_set
        index = 0
        while index < len(obs_s_in_set):
            if obs_s_in_set[index] == obs_s_out_set[index] and obs_s_in_set[index] > (path_index2s[-1] - 1):
                obs_s_in_set.pop(index)
                obs_s_out_set.pop(index)
                obs_t_in_set.pop(index)
                obs_t_out_set.pop(index)
            else:
                index += 1

        index01 = 0
        while index01 < len(obs_s_in_set):
            if obs_s_in_set[index01] == 0:
                obs_s_in_set.pop(index01)
                obs_s_out_set.pop(index01)
                obs_t_in_set.pop(index01)
                obs_t_out_set.pop(index01)
            else:
                index01 += 1
        # 2024.5.14改
        # if obs_s_in_set:
        #     obs_to_consider = True
        # else:
        #     obs_to_consider = False
    else:
        obs_to_consider = False
    return obs_s_in_set, obs_s_out_set, obs_t_in_set, obs_t_out_set, obs_to_consider, SLTofsingle_obs_list
    # values = [sublist[0][0] for sublist in SLTofsingle_obs_list]
    #
    # # 对提取的值进行排序
    # sorted_values = sorted(values)
    #
    # increase_count = 0
    #
    # # 找到排序后的第一个值
    # min_value = sorted_values[0]
    # # 遍历obs列表，找到包含最小值的子列表，并取出第二个元素
    # selected_obs_list = []
    # obs_to_consider = True
    #
    # for sublist in SLTofsingle_obs_list:
    #     if sublist[0][0] == min_value:
    #         second_element = sublist[0][1]
    #         # if abs(second_element) < 2.5:
    #         if abs(second_element) < 2:
    #             selected_obs_list = sublist
    #             break
    #         else:
    #             obs_to_consider = False
    #             increase_count += 1
    #             min_value = sorted_values[increase_count]
    #             continue
    #     else:
    #         continue
    # if obs_to_consider:
    #     # print(selected_obs_list)
    #     obs_st_s_in = selected_obs_list[0][0]
    #     obs_st_s_out = selected_obs_list[-1][0]
    #     obs_st_t_in = selected_obs_list[0][2]
    #     obs_st_t_out = selected_obs_list[n-1][2]
    # else:
    #     obs_st_s_in = []
    #     obs_st_s_out = []
    #     obs_st_t_in = []
    #     obs_st_t_out = []
    # return obs_s_in_set, obs_s_out_set, obs_t_in_set, obs_t_out_set, obs_to_consider


# 从这里开始定义一个新的函数
def cal_egoinfo(observation, headings_degrees, curvatures):
    reference_speed = 15
    w_cost_ref_speed = 4000
    w_cost_accel = 100
    # w_cost_obs = 10000000
    w_cost_obs = 10000000
    plan_start_v = observation.ego_info.v
    # plan_start_v = 25
    plan_start_a = observation.ego_info.a
    # plan_start_a = 13
    plan_start_yaw = observation.ego_info.yaw
    # plan_start_yaw = 25
    plan_start_vx = plan_start_v*cos(math.radians(plan_start_yaw))
    plan_start_vy = plan_start_v*sin(math.radians(plan_start_yaw))
    plan_start_heading = headings_degrees[0]
    tor = np.array([cos(math.radians(plan_start_heading)), sin(math.radians(plan_start_heading))])
    nor = np.array([-sin(math.radians(plan_start_heading)), cos(math.radians(plan_start_heading))])
    # a_tor = plan_start_a*tor
    # a_nor = (plan_start_v**2)*curvatures[0]*nor
    # plan_start_ax = a_tor[0] + a_nor[0]
    # plan_start_ay = a_tor[1] + a_nor[1]
    plan_start_s_dot = np.dot(tor, np.array([plan_start_vx, plan_start_vy]))
    # # plan_start_s_dot2 = np.dot(tor, np.array([plan_start_ax, plan_start_ay]))
    # plan_start_s_dot2 = np.dot(tor, np.array([plan_start_ax, plan_start_ay]))
    plan_start_s_dot2 = plan_start_a
    # s_list = np.concatenate([
    # np.arange(0, 5, 0.5),          # [0:0.5:4.5]
    # np.arange(5.5, 15, 1),         # [5.5:1:14.5]
    # np.arange(16, 30, 1.5),        # [16:1.5:29.5]
    # np.arange(32, 55, 2.5)         # [32:2.5:54.5]
    # ])
    #
    # t_list = np.arange(0.5, 8.5, 0.5)
    s_list = np.concatenate([
    np.arange(0, 3, 0.2),  # [0:0.2:3]
    np.arange(4, 8, 1),  # [4:1:8]
    np.arange(9, 19.5, 1.5)  # [9:1.5:16.5]
    ])

    t_list = np.arange(0, 1.6, 0.1)
    return reference_speed, w_cost_ref_speed, plan_start_v, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, plan_start_s_dot2

