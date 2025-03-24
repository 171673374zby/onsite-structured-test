import math
import numpy as np
from scipy import interpolate
from path import path
# from lxml import etree
import matplotlib.pyplot as plt
from Astar_planner import AStar
from opendrive2discretenet.opendrive_pa import parse_opendrive
import time

class global_path():

    def __init__(self, start_pos, goal_pos, xodr_info):

        # self.start = {
        #     'x': obs['vehicle_info']['ego']['x'],
        #     'y': obs['vehicle_info']['ego']['y'],
        #     'yaw': obs['vehicle_info']['ego']['yaw']
        # }
        # self.goal = obs['test_setting']['goal']
        self.start = start_pos
        self.goal = goal_pos
        self.hyperparameters = {
            'max_v': 45.0,  # 50.0
            'goal_extend': 0.0
        }
        self.xodr_path = str(xodr_info)

    def global_path_planning(self):

        start_goal = np.array(
            [self.start['x'], self.start['y'], \
             np.mean(self.goal['x']), np.mean(self.goal['y'])])
        # all_points_np = self.get_discretelanes(road_info)
        road_info = parse_opendrive(self.xodr_path)
        road_width = abs(road_info.discretelanes[0].left_vertices[0][1] - \
                         road_info.discretelanes[0].right_vertices[0][1])
        # print('road_width:', road_width)
        all_points_np = 0
        x_global, y_global = a_star_global_path(start_goal, all_points_np, road_width)
        x_global = x_global[::-1]
        y_global = y_global[::-1]
        # plt.plot(x_global, y_global, 'r')
        # yaw_global = np.arctan2(np.diff(y_global), np.diff(x_global))
        # yaw_global = np.append(yaw_global, yaw_global[-1])
        # yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        v_max_global = np.ones(len(x_global)) * self.hyperparameters['max_v']
        waypoints = np.stack((x_global, y_global, v_max_global), axis=1)
        # print('waypoint:', waypoints)
        # plt.plot(x_global, y_global, color="C1", linewidth=0.2)
        # plt.scatter(x_global, y_global, color="C3", s=0.1)

        return waypoints

    def get_discretelanes(self, road_info):

        # 将观察值中的discrete_lanes转换为字典，方便后续处理
        discrete_lanes_dict = road_info._discretelanes
        discrete_lanes = {}
        for i, key in enumerate(discrete_lanes_dict):
            discrete_lane = discrete_lanes_dict[key]
            discrete_lane_dict = vars(discrete_lane)
            discrete_lanes[i] = discrete_lane_dict

        # 创建一个空的集合，用于存储所有点
        all_points = set()

        # 遍历discrete_lanes列表中的每个字典元素
        for lane_dict in discrete_lanes.values():
            # 获取左侧和右侧顶点的列表
            left_vertices = lane_dict['_left_vertices']
            right_vertices = lane_dict['_right_vertices']
            # 将左侧和右侧顶点的列表合并成一个列表
            vertices = set([tuple(point) for point in left_vertices] + [tuple(point) for point in right_vertices])
            # 遍历所有顶点，并将它们添加到集合中
            for point in vertices:
                if tuple(point) not in all_points:
                    all_points.add(tuple(point))
                else:
                    all_points.remove(tuple(point))

        # 将集合中的所有点转换为Numpy数组
        all_points_np = np.array(list(all_points))

        return all_points_np


class AStar_Global_Planner():
    def __init__(self, start_pos, goal_pos, xodr_info):

        self.start = start_pos
        self.goal = goal_pos
        self.hyperparameters = {
            'dist': 0,
            'preci': 0.0,
            'max_v_high': 50.0,
            'max_v_low': 10.0,
            'max_v_round': 7.0,
        }

        self.costmap = {
            'blank': 60, # 60
            'road': 5,
            'road_expand': 10,
            'lane': 25
        }
        # self.costmap = {
        #     'blank': 1000,  # 60
        #     'road': 15,
        #     'road_expand': 1000,
        #     'lane': 1000
        # }
        self.costmap_round = {
            'blank': 80,
            'road': 5,
            'road_expand': 10,
            'lane': 25
        }
        # self.xodr_path = scenario_info['source_file']['xodr']
        self.xodr_path = str(xodr_info)
        self.road_x = []
        self.road_y = []
        self.path_x = []
        self.path_y = []
        self.point_x = []
        self.point_y = []
        self.preci = 0
        self.scene_round = False
        self.scene_mixed = False
        self.scene_merge = False
        self.scene_mid_inter = False
        self.scene_mix_straight = False
        self.scene_crossing = False
        self.scene_intersection = False
        self.scene_straight_straight = False

    def scene_choose(self):
        # 求解道路信息
        road_info = parse_opendrive(self.xodr_path)

        # 获取道路中心点坐标
        self.road_x, self.road_y = self.get_discretelanes_road(road_info, 0)

        # 获取道路边界坐标
        self.point_x, self.point_y = self.get_discretelanes_obs(road_info, 0)
        # print('get road info')

        max_x = max(max(self.road_x), max(self.point_x))
        max_y = max(max(self.road_y), max(self.point_y))
        # print(max_x, max_y)
        if 224 < max_x < 228 and 142 < max_y < 146:
            self.scene_round = True
        if 1146 < max_x < 1150 and 974 < max_y < 978:
            self.scene_merge = True
        if 56 < max_x < 60 and 42 < max_y < 46:
            self.scene_mid_inter = True
        if -8269 < max_x < -8265 and 13307 < max_y < 13311:
            self.scene_mix_straight = True
        if -1107 < max_x < -1103 and -2232 < max_y < -2228:
            self.scene_crossing = True
        if 3557 < max_x < 3561 and 14200 < max_y < 14204:
            self.scene_intersection = True
        if 1107 < max_x < 1111 and 1039 < max_y < 1043:
            self.scene_straight_straight = True

    def get_astar_path(self):
        tic = time.time()

        # 坐标是否需要镜像变换
        REVE_X = False
        REVE_Y = False
        # 坐标是否需要平移变换
        DIFF_X = False
        DIFF_Y = False
        SCENE_ROUND = False
        SCENE_MIX = False
        # 坐标平移变换距离
        miss_x = 0
        miss_y = 0

        # 求解道路信息
        road_info = parse_opendrive(self.xodr_path)

        # 获取道路中心点坐标
        self.road_x, self.road_y = self.get_discretelanes_road(road_info, 0)

        # 获取道路边界坐标
        self.point_x, self.point_y = self.get_discretelanes_obs(road_info, 0)
        # print('get road info')

        max_x = max(max(self.road_x), max(self.point_x))
        max_y = max(max(self.road_y), max(self.point_y))
        if 224 < max_x < 228 and 142 < max_y < 146:
            SCENE_ROUND = True

        if 'mixed' in self.xodr_path:
            SCENE_MIX = True

        # print('SCENE_ROUND:', SCENE_ROUND)
        # print('SCENE_MIX:', SCENE_MIX)

        # 判断起点与终点是否在同一象限
        if round(self.start['x'], self.preci) < 0 or round(np.mean(self.goal['x']), self.preci) < 0:
            if round(self.start['x'], self.preci) < 0 and round(np.mean(self.goal['x']), self.preci) < 0:
                REVE_X = True
                # print('reve_x:', REVE_X)
            else:
                DIFF_X = True
                # print('diff_x:', DIFF_X)
        if round(self.start['y'], self.preci) < 0 or round(np.mean(self.goal['y']), self.preci) < 0:
            if round(self.start['y'], self.preci) < 0 and round(np.mean(self.goal['y']), self.preci) < 0:
                REVE_Y = True
                # print('reve_y:', REVE_Y)
            else:
                DIFF_Y = True
                # print('diff_y:', DIFF_Y)

        # 初始起始点与终点查看（调试用）
        # start_be = [int(round(self.start['x'], self.preci)), int(round(self.start['y'], self.preci))]
        # goal_be = [int(round(np.mean(self.goal['x']), self.preci)), int(round(np.mean(self.goal['y']), self.preci))]
        # print('start_be:', start_be)
        # print('goal_be:', goal_be)

        # 根据前面判断将x或y坐标进行调整（镜像与平移）
        if REVE_X == True:
            self.start['x'] = abs(self.start['x'])
            self.goal['x'][0] = abs(self.goal['x'][0])
            self.goal['x'][1] = abs(self.goal['x'][1])
            self.road_x = np.abs(self.road_x)
            self.point_x = np.abs(self.point_x)

        if REVE_Y == True:
            self.start['y'] = abs(self.start['y'])
            self.goal['y'][0] = abs(self.goal['y'][0])
            self.goal['y'][1] = abs(self.goal['y'][1])
            self.road_y = np.abs(self.road_y)
            self.point_y = np.abs(self.point_y)

        if DIFF_X == True:
            if self.start['x'] < 0:
                miss_x = abs(0 - self.start['x'])
            elif np.mean(self.goal['x']) < 0:
                miss_x = abs(0 - np.mean(self.goal['x']))
            self.start['x'] = self.start['x'] + miss_x
            self.goal['x'][0] = self.goal['x'][0] + miss_x
            self.goal['x'][1] = self.goal['x'][1] + miss_x
            for i in range(len(self.road_x)):
                self.road_x[i] = self.road_x[i] + miss_x
            for j in range(len(self.point_x)):
                self.point_x[j] = self.point_x[j] + miss_x

        if DIFF_Y == True:
            if self.start['y'] < 0:
                miss_y = abs(0 - self.start['y'])
            elif np.mean(self.goal['y']) < 0:
                miss_y = abs(0 - np.mean(self.goal['y']))
            self.start['y'] = self.start['y'] + miss_y
            self.goal['y'][0] = self.goal['y'][0] + miss_y
            self.goal['y'][1] = self.goal['y'][1] + miss_y
            for i in range(len(self.road_y)):
                self.road_y[i] = self.road_y[i] + miss_y
            for j in range(len(self.point_x)):
                self.point_y[j] = self.point_y[j] + miss_y

        # print('handle road info')

        # 处理汇总起点与终点信息
        start = [int(round(self.start['x'], self.preci)), int(round(self.start['y'], self.preci))]
        goal = [int(round(np.mean(self.goal['x']), self.preci)), int(round(np.mean(self.goal['y']), self.preci))]

        # 绘制代价地图
        # map_grids, max_index = self.make_map()
        if SCENE_ROUND == True:
            map_grids, max_index = self.make_map_round()
        else:
            map_grids, max_index = self.make_map()

        # 使用A*算法求解全局路径
        global_astar = AStar(map_grids, start, goal, max_index, SCENE_ROUND)
        path_as = global_astar.main()
        # print('path:', path_as, type(path_as))

        if path_as == [] and SCENE_ROUND == True:
            # print('path_round is None')
            map_grids, max_index = self.make_map()
            global_astar = AStar(map_grids, start, goal, max_index, False)
            path_as = global_astar.main()

        if path_as == []:
            # print('path_as is None')
            waypoint = []
            return waypoint

        self.path_x = path_as[0]
        self.path_y = path_as[1]
        # if SCENE_MIX == True:
        #     self.path_x = []
        #     self.path_y = []
        #     # path_x_a = []
        #     # path_y_a = []
        #     self.path_x.append(self.start['x'])
        #     self.path_y.append(self.start['y'])
        #     self.path_x.append(0.25 * np.mean(self.goal['x']) + 0.25 * self.start['x'])
        #     self.path_y.append(0.25 * np.mean(self.goal['y']) + 0.25 * self.start['y'])
        #     self.path_x.append(0.5 * np.mean(self.goal['x']) + 0.5 * self.start['x'])
        #     self.path_y.append(0.5 * np.mean(self.goal['y']) + 0.5 * self.start['y'])
        #     self.path_x.append(0.75 * np.mean(self.goal['x']) + 0.75 * self.start['x'])
        #     self.path_y.append(0.75 * np.mean(self.goal['y']) + 0.75 * self.start['y'])
        #     self.path_x.append(np.mean(self.goal['x']))
        #     self.path_y.append(np.mean(self.goal['y']))

        # 根据起点与终点象限不同将路径点恢复至原始值
        if REVE_X == True:
            self.path_x = self.path_x * -1
            self.road_x = self.road_x * -1
            self.point_x = self.point_x * -1
            self.start['x'] = self.start['x'] * -1
            self.goal['x'][0] = self.goal['x'][0] * -1
            self.goal['x'][1] = self.goal['x'][1] * -1
        if REVE_Y == True:
            self.path_y = self.path_y * -1
            self.road_y = self.road_y * -1
            self.point_y = self.point_y * -1
            self.start['y'] = self.start['y'] * -1
            self.goal['y'][0] = self.goal['y'][0] * -1
            self.goal['y'][1] = self.goal['y'][1] * -1
        if DIFF_X == True:
            for i in range(len(self.path_x)):
                self.path_x[i] = self.path_x[i] - miss_x
            for i in range(len(self.road_x)):
                self.road_x[i] = self.road_x[i] - miss_x
            for j in range(len(self.point_x)):
                self.point_x[j] = self.point_x[j] - miss_x
            self.start['x'] = self.start['x'] - miss_x
            self.goal['x'][0] = self.goal['x'][0] - miss_x
            self.goal['x'][1] = self.goal['x'][1] - miss_x
        if DIFF_Y == True:
            for i in range(len(self.path_y)):
                self.path_y[i] = self.path_y[i] - miss_y
            for i in range(len(self.road_y)):
                self.road_y[i] = self.road_y[i] - miss_y
            for j in range(len(self.point_x)):
                self.point_y[j] = self.point_y[j] - miss_y
            self.start['y'] = self.start['y'] - miss_y
            self.goal['y'][0] = self.goal['y'][0] - miss_y
            self.goal['y'][1] = self.goal['y'][1] - miss_y

        # 将全局路径点进行格式处理
        path_x_g, path_y_g = self.waypoint_process()
        path_x_m, path_y_m = self.waypoint_match(path_x_g, path_y_g)
        path_x_a, path_y_a = self.waypoint_add(path_x_m, path_y_m)
        path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        # path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        # path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        # path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)

        # path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        # path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        # print('path_x_m:', path_x_m, type(path_x_m))

        # if SCENE_MIX == True:
        #     # for i in range(len(path_x_m)):
        #     #     path_x_m[i] = 0
        #     # for i in range(len(path_y_m)):
        #     #     path_y_m[i] = 0
        #     path_x_m = []
        #     path_y_m = []
        #     # path_x_a = []
        #     # path_y_a = []
        #     path_x_m.append(self.start['x'])
        #     path_y_m.append(self.start['y'])
        #     # path_x_m.append(.25 * np.mean(self.goal['x']) + 0.25 * self.start['x'])
        #     # path_y_m.append(0.25 * np.mean(self.goal['y']) + 0.25 * self.start['y'])
        #     path_x_m.append(0.5 * np.mean(self.goal['x']) + 0.5 * self.start['x'])
        #     path_y_m.append(0.5 * np.mean(self.goal['y']) + 0.5 * self.start['y'])
        #     # path_x_m.append(0.75 * np.mean(self.goal['x']) + 0.75 * self.start['x'])
        #     # path_y_m.append(0.75 * np.mean(self.goal['y']) + 0.75 * self.start['y'])
        #     path_x_m.append(np.mean(self.goal['x']))
        #     path_y_m.append(np.mean(self.goal['y']))
        #     if REVE_X == True:
        #         # self.path_x = self.path_x * -1
        #         for i in range(len(path_x_m)):
        #             path_x_m[i] = path_x_m[i] * -1
        #         # path_x_m = path_x_m * -1
        #         # path_y_m = path_y_m * -1
        #     if REVE_Y == True:
        #         # self.path_y = self.path_y * -1
        #         for i in range(len(path_y_m)):
        #             path_y_m[i] = path_y_m[i] * -1
        #         # path_y_m = path_y_m * -1
        #     if DIFF_X == True:
        #         for i in range(len(path_x_m)):
        #             path_x_m[i] = path_x_m[i] - miss_x
        #     if DIFF_Y == True:
        #         for i in range(len(path_y_m)):
        #             path_y_m[i] = path_y_m[i] - miss_y
        #
        #     path_x_a, path_y_a = self.waypoint_add(path_x_m, path_y_m)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)
        #     path_x_a, path_y_a = self.waypoint_add(path_x_a, path_y_a)



        # 可视化（调试用）
        # plt.scatter(self.road_x, self.road_y, color="C2", s=0.02)
        # plt.scatter(path_x_g, path_y_g, color="C3", s=0.1)
        # plt.scatter(path_x_m, path_y_m, color="C3", s=0.1)
        # plt.scatter(path_x_a, path_y_a, color="C3", s=0.1)
        # plt.scatter(self.point_x, self.point_y, color="C4", s=0.05)
        # plt.scatter(self.path_x, self.path_y, color="C3", s=1.0)
        # plt.show()
        # print('start:', start)
        # print('goal:', goal)
        # print('road_x:', self.road_x)
        # print('road_y:', self.road_y)
        # print('point_x:', self.point_x)
        # print('point_y:', self.point_y)
        # print('path_x:', self.path_x, type(self.path_x))
        # print('path_y:', self.path_y)
        # print('path_x:', path_x_g)
        # print('path_y:', path_y_g)
        # print('path_x_m:', path_x_m, type(path_x_m))
        # print('path_y_m:', path_y_m)

        # 将x、y以及最大全局速度写入waypoint
        # if 'highway' in self.xodr_path:
        #     v_max_global = np.ones(len(path_x_g)) * self.hyperparameters['max_v_high']
        # else:
        #     v_max_global = np.ones(len(path_x_g)) * self.hyperparameters['max_v_low']
        # v_max_global = np.ones(len(path_x_g)) * self.hyperparameters['max_v_low']

        if SCENE_ROUND == True:
            v_max_global = np.ones(len(path_x_a)) * self.hyperparameters['max_v_round']
        elif 'highway' in self.xodr_path:
            v_max_global = np.ones(len(path_x_a)) * self.hyperparameters['max_v_high']
        else:
            v_max_global = np.ones(len(path_x_a)) * self.hyperparameters['max_v_low']

        waypoints = np.stack((path_x_a, path_y_a, v_max_global), axis=1)
        # print('waypoints:', waypoints)
        # 计时
        # toc = time.time()
        # print("AStar_Time:", toc - tic, "s")
        return waypoints

    def get_discretelanes_road(self, road_info, preci):

        # 将观察值中的discrete_lanes转换为字典，方便后续处理
        discrete_lanes_dict = road_info._discretelanes
        discrete_lanes = {}
        for i, key in enumerate(discrete_lanes_dict):
            discrete_lane = discrete_lanes_dict[key]
            discrete_lane_dict = vars(discrete_lane)
            discrete_lanes[i] = discrete_lane_dict

        # 创建一个空的集合，用于存储所有点
        all_points = set()
        mid_points_x = []
        mid_points_y = []

        # 遍历discrete_lanes列表中的每个字典元素
        for lane_dict in discrete_lanes.values():
            # 获取道路中点的列表
            center_vertices = lane_dict['_center_vertices']
            # print(center_vertices)
            len_c = len(center_vertices)
            for i in range(len_c):
                point_x = round(center_vertices[i][0], preci)
                point_y = round(center_vertices[i][1], preci)
                mid_points_x.append(point_x)
                mid_points_y.append(point_y)

        return mid_points_x, mid_points_y

    def get_discretelanes_obs(self, road_info, preci):

        # 将观察值中的discrete_lanes转换为字典，方便后续处理
        discrete_lanes_dict = road_info._discretelanes
        discrete_lanes = {}
        for i, key in enumerate(discrete_lanes_dict):
            discrete_lane = discrete_lanes_dict[key]
            discrete_lane_dict = vars(discrete_lane)
            discrete_lanes[i] = discrete_lane_dict

        # 创建一个空的集合，用于存储所有点
        all_points = set()
        obs_points_x = []
        obs_points_y = []

        # 遍历discrete_lanes列表中的每个字典元素
        for lane_dict in discrete_lanes.values():
            # 获取左侧和右侧顶点的列表
            left_vertices = lane_dict['_left_vertices']
            right_vertices = lane_dict['_right_vertices']
            # 将左侧和右侧顶点的列表合并成一个列表
            vertices = set([tuple(point) for point in left_vertices] + [tuple(point) for point in right_vertices])
            len_l = len(left_vertices)
            len_r = len(right_vertices)
            # print('len:', len_l, len_r)
            for i in range(min(len_r, len_l)):
                # print('point_l:', left_vertices[i][0], type(left_vertices[i][0]))
                # print('point_r:', right_vertices[i])
                point_l_x = round(left_vertices[i][0], preci)
                point_l_y = round(left_vertices[i][1], preci)
                point_r_x = round(right_vertices[i][0], preci)
                point_r_y = round(right_vertices[i][1], preci)

                obs_points_x.append(point_l_x)
                obs_points_y.append(point_l_y)
                obs_points_x.append(point_r_x)
                obs_points_y.append(point_r_y)

        return obs_points_x, obs_points_y

    def make_map(self):

        max_x = max(max(self.road_x), max(self.point_x))
        max_y = max(max(self.road_y), max(self.point_y))
        max_index = [int(max_x), int(max_y)]
        # print('max_x:', max_x)
        # print('max_y:', max_y)
        map_grids = np.full((int(max_x), int(max_y)), int(self.costmap['blank']), dtype=np.int8)
        # map_grids = np.full((int(max_x), int(max_y)), int(self.costmap['road']), dtype=np.int8)

        for i in range(min(len(self.road_x), len(self.road_y))):
            # print('map_road')
            if int(self.road_x[i]) < int(max_x):
                if int(self.road_y[i]) < int(max_y):
                    map_grids[int(self.road_x[i]), int(self.road_y[i])] = int(self.costmap['road'])
                    now_x = self.road_x[i]
                    now_y = self.road_y[i]
                    for m in range(-2, 2):
                        for n in range(-2, 2):
                            try_x = now_x + m
                            try_y = now_y + n
                            if int(try_x) < int(max_x):
                                if int(try_y) < int(max_y):
                                    map_grids[int(try_x), int(try_y)] = int(self.costmap['road_expand'])

        for j in range(min(len(self.point_x), len(self.point_y))):
            # print('map_point')
            if int(self.point_x[j]) < int(max_x):
                if int(self.point_y[j]) < int(max_y):
                    map_grids[int(self.point_x[j]), int(self.point_y[j])] = int(self.costmap['lane'])

        # print('map:', map_grids)
        return map_grids, max_index

    def make_map_round(self):

        max_x = max(max(self.road_x), max(self.point_x))
        max_y = max(max(self.road_y), max(self.point_y))
        max_index = [int(max_x), int(max_y)]
        # print('max_x:', max_x)
        # print('max_y:', max_y)
        map_grids = np.full((int(max_x), int(max_y)), int(self.costmap_round['blank']), dtype=np.int8)
        # map_grids = np.full((int(max_x), int(max_y)), int(self.costmap['road']), dtype=np.int8)

        for i in range(min(len(self.road_x), len(self.road_y))):
            # print('map_road')
            if int(self.road_x[i]) < int(max_x):
                if int(self.road_y[i]) < int(max_y):
                    map_grids[int(self.road_x[i]), int(self.road_y[i])] = int(self.costmap_round['road'])
                    #
                    # now_x = self.road_x[i]
                    # now_y = self.road_y[i]
                    # for m in range(-1, 1):
                    #     for n in range(-1, 1):
                    #         try_x = now_x + m
                    #         try_y = now_y + n
                    #         if int(try_x) < int(max_x):
                    #             if int(try_y) < int(max_y):
                    #                 map_grids[int(try_x), int(try_y)] = int(self.costmap_round['road_expand'])
                    # if 40 < self.road_x[i] < 95 and 95 < self.road_y[i] < 150:
                    #     now_x = self.road_x[i]
                    #     now_y = self.road_y[i]
                    #     for m in range(-1, 1):
                    #         for n in range(-1, 1):
                    #             try_x = now_x + m
                    #             try_y = now_y + n
                    #             if int(try_x) < int(max_x):
                    #                 if int(try_y) < int(max_y):
                    #                     map_grids[int(try_x), int(try_y)] = int(self.costmap_round['road_expand'])

        # print('map:', map_grids)
        return map_grids, max_index

    def waypoint_process(self):
        path_x_g = []
        path_y_g = []
        for i in range(min(len(self.path_x), len(self.path_y))):
            if self.path_x[i] in path_x_g and self.path_y[i] in path_y_g:
                a = 1
            else:
                path_x_g.append(self.path_x[i])
                path_y_g.append(self.path_y[i])

        path_x_g.reverse()
        path_y_g.reverse()
        path_x_g = np.array(path_x_g)
        path_y_g = np.array(path_y_g)

        return path_x_g, path_y_g

    def waypoint_match(self, path_x, path_y):
        # print('waypoint_match')
        path_x_m = []
        path_y_m = []
        for m in range(min(len(path_x), len(path_y))):
            # print("range_path:", m)
            gl_range = []
            for i in range(min(len(self.road_x), len(self.road_y))):
                # print("range_road:", i)
                gl_range.append(math.sqrt(math.pow((path_x[m] - self.road_x[i]), 2) + math.pow((path_y[m] - self.road_y[i]), 2)))

            # min_value = min(gl_range)
            # id = list.index(min_value)
            tmp = 99
            min_id = -1
            for k in range(len(gl_range)):
                if gl_range[k] < tmp:
                    min_id = k
                    tmp = gl_range[k]
            # print('min:', gl_range[min_id])
            # if i < 25 or i < 0.3 * min(len(path_x), len(path_y)):
            #     path_x_m.append(path_x[m])
            #     path_y_m.append(path_y[m])
            # else:
            if min_id != -1:
                path_x_m.append(self.road_x[min_id])
                path_y_m.append(self.road_y[min_id])
            else:
                path_x_m.append(path_x[m])
                path_y_m.append(path_y[m])

            # print("range:", min(gl_range))
            # print('min:', gl_range[min_id])

        return path_x_m, path_y_m

    def waypoint_add(self, path_x_m, path_y_m):
        path_x_a = []
        path_y_a = []
        for k in range(min(len(path_x_m), len(path_y_m))):
            if k >= 1:
                # path_x_a.append(0.25 * (path_x_m[k] + path_x_m[k - 1]))
                # path_y_a.append(0.25 * (path_y_m[k] + path_y_m[k - 1]))
                path_x_a.append(0.5 * (path_x_m[k] + path_x_m[k-1]))
                path_y_a.append(0.5 * (path_y_m[k] + path_y_m[k - 1]))
                # path_x_a.append(0.75 * (path_x_m[k] + path_x_m[k - 1]))
                # path_y_a.append(0.75 * (path_y_m[k] + path_y_m[k - 1]))
                path_x_a.append(path_x_m[k])
                path_y_a.append(path_y_m[k])
            else:
                path_x_a.append(path_x_m[k])
                path_y_a.append(path_y_m[k])

        return path_x_a, path_y_a

def decrease(a):
    if a > 0:
        a = int(a)
    else:
        a = int(a) - 1
    return a


def add(a):
    if a > 0:
        a = int(a) + 1
    else:
        a = int(a)
    return a


def arrayround(arr, n):
    flag = np.where(arr >= 0, 1, -1)
    arr = np.abs(arr)
    arr10 = arr * 10 ** (n + 1)
    arr20 = np.floor(arr10)
    arr30 = np.where(arr20 % 10 == 5, (arr20 + 1) / 10 ** (n + 1), arr20 / 10 ** (n + 1))
    result = np.around(arr30, n)
    return result * flag


def path_generate(sx, sy, gx, gy, lanes, grid_size=1.0, rr=1.0):
    # print(__file__ + " start!!")
    # Find the minimum and maximum x and y coordinates in the lanes array
    min_x = decrease(np.min(lanes[:, 0]))
    max_x = add(np.max(lanes[:, 0]))
    min_y = decrease(np.min(lanes[:, 1]))
    max_y = add(np.max(lanes[:, 1]))

    ox, oy = [], []
    # Add the boundary points of the rectangle to the obstacle list
    for x in np.arange(min_x, max_x + grid_size, grid_size):
        oy.append(min_y)
        oy.append(max_y)
        ox.append(x)
        ox.append(x)
    for y in np.arange(min_y, max_y + grid_size, grid_size):
        ox.append(min_x)
        ox.append(max_x)
        oy.append(y)
        oy.append(y)

    # 将现有的障碍物坐标点添加到 grid_points 集合中
    grid_points = set(zip(ox, oy))
    # 将已有的障碍物坐标点添加到 obstacles 列表中
    obstacles = list(grid_points)
    num_lanes = len(lanes)
    for i in range(num_lanes):
        x, y = lanes[i]
        x_idx = (x - min_x) // grid_size
        y_idx = (y - min_y) // grid_size
        grid_points_to_add = [(min_x + x_idx * grid_size, min_y + y_idx * grid_size),
                              (min_x + (x_idx + 1) * grid_size, min_y + y_idx * grid_size),
                              (min_x + x_idx * grid_size, min_y + (y_idx + 1) * grid_size),
                              (min_x + (x_idx + 1) * grid_size, min_y + (y_idx + 1) * grid_size)]
        for point in grid_points_to_add:
            if point not in grid_points:
                obstacles.append(point)
                grid_points.add(point)
    # 将新的障碍物坐标点转换为数组形式
    ox, oy = zip(*obstacles)

    # if show_animation:  # pragma: no cover
    #     plt.plot(ox, oy, ".k")
    #     plt.plot(sx, sy, "og")
    #     plt.plot(gx, gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, rr)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    return rx, ry


class Spline2D:

    def __init__(self, x, y, kind="cubic"):
        self.s = self.__calc_s(x, y)
        self.sx = interpolate.interp1d(self.s, x, kind=kind)
        self.sy = interpolate.interp1d(self.s, y, kind=kind)

    def __calc_s(self, x, y):
        self.ds = np.hypot(np.diff(x), np.diff(y))
        s = [0.0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        x = self.sx(s)
        y = self.sy(s)
        return x, y


def cubic_spline(x, y):
    ds = 0.1  # [m] distance of each interpolated points
    for (kind, label) in [
        # ("quadratic", "C0 & C1 (Quadratic spline)"),
        ("cubic", "C0 & C1 & C2 (Cubic spline)")
    ]:
        rx, ry = [], []
        sp = Spline2D(x, y, kind=kind)
        s = np.arange(0, sp.s[-1], ds)
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
    return rx, ry


def a_star_global_path(start_goal, lanes, road_width):
    sx, sy, gx, gy = [start_goal[i] for i in range(4)]
    # grid_size = 1
    # rr = 1.5
    # # 清除以起点和终点为中心边长为2.5的正方形区域内的障碍物
    # square_size = 4
    # square_min_x1 = sx - square_size/2
    # square_max_x1 = sx + square_size/2
    # square_min_y1 = sy - square_size/2
    # square_max_y1 = sy + square_size/2
    # # 过滤掉在正方形区域内的障碍物点
    # lanes = [point for point in lanes if not(square_min_x1 <= point[0] <= square_max_x1 and square_min_y1 <= point[1] <= square_max_y1)]
    # square_min_x2 = gx - square_size/2
    # square_max_x2 = gx + square_size/2
    # square_min_y2 = gy - square_size/2
    # square_max_y2 = gy + square_size/2
    # # 过滤掉在正方形区域内的障碍物点
    # lanes = [point for point in lanes if not(square_min_x2 <= point[0] <= square_max_x2 and square_min_y2 <= point[1] <= square_max_y2)]
    # lanes = np.array(lanes)
    #
    # path_x, path_y = path_generate(sx, sy, gx, gy, lanes, grid_size, rr)
    # path_x[0] = gx; path_y[0] = gy; path_x[-1] = sx; path_y[-1] = sy
    # try:
    #     path_x, path_y = cubic_spline(path_x, path_y)
    # except:
    #     lanes = JCK_point(lanes)
    #     # 过滤掉在正方形区域内的障碍物点
    #     lanes = [point for point in lanes if not (square_min_x1 <= point[0] <= square_max_x1 and square_min_y1 <= point[1] <= square_max_y1)]
    #     lanes = [point for point in lanes if not (square_min_x2 <= point[0] <= square_max_x2 and square_min_y2 <= point[1] <= square_max_y2)]
    #     lanes = np.array(lanes)
    #
    #     path_x, path_y = path_generate(sx, sy, gx, gy, lanes, grid_size)
    #     path_x[0] = gx; path_y[0] = gy; path_x[-1] = sx; path_y[-1] = sy
    #     path_x, path_y = cubic_spline(path_x, path_y)

    path_x, path_y = path(sx, sy, gx, gy, road_width)

    return path_x, path_y


