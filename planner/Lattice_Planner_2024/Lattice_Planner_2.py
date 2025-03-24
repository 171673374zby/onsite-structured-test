from __future__ import print_function
from __future__ import division

# System level imports
import os
import sys
script_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_directory)
import math
import numpy as np
import controller2d
import local_planner
import behavioural_planner
import velocity_planner
import matplotlib.pyplot as plt
import random

from scipy.interpolate import interp1d
from scipy.optimize import minimize

from global_path_planner import global_path, AStar_Global_Planner
from opendrive2discretenet.opendrive_pa import parse_opendrive

from test03 import cal_refer_path_info, obs_process, cal_egoinfo
from test11 import dynamic_programming
from test05 import GenerateConvexSpace, path_index2s
from test06 import cal_plan_start_s_dotanddot2, SpeedPlanningwithQuadraticPlanning
from test09 import project_curvature_and_heading, cal_localwaypoints
from collision_checker_1 import collision_check
from obstacle_prediction import extract_nearby_vehicles
from obstacle_prediction import obstacle_prediction

from position_pid import PositionPID
from PID_controll import PIDAngleController

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'

class Lattice_Planner_2():

    def __init__(self):
        self.start_pos = {}
        self.goal_pos = {}
        self.scenario_info = {}
        self.frame = 0
        self.dt = 0
        self.scene_t = 0
        self.hyperparameters = {
            'max_accel': 9.8,
            'max_steering': 0.68,
            'overtaking_speed': 15.0
        }
        self.con_ite = 1
        self.high_state = 0
        self.scene_round = False
        self.scene_mixed = False
        self.scene_merge = False
        self.scene_mid_inter = False
        self.scene_mix_straight = False
        self.scene_crossing = False
        self.scene_intersection = False
        self.scene_straight_straight = False

        self.None_later_control = False
        self.None_later_control_highway = False
        self.None_later_control_city = True

        self.path_all_false = True
        self.middle_follow_distance = -1
        self.init_yaw = 0
        self.Initial_v = 10

        # 动力学约束
        self.ACC_LIMIT = 9.78  # m/s^2 9.8
        self.JERK_LIMIT = 8  # m/s^3  49
        self.ROT_LIMIT = 0.68  # rad 0.7
        self.ROT_RATE_LIMIT = 0.2  # rad/s 1.4
        self.GLOBAL_MAX_SPEED = 50.0  # m/s 55.0

        self.waypoints = []
        self.pre_local_waypoints = None
        self.pre_u0 = np.array([0.0, 0.0])
        self.controller = []
        self.pre_control = []
        """
        Configurable params
        """
        # Planning Constants
        self.NUM_PATHS = 11
        self.BP_LOOKAHEAD_BASE = 15.0  # m 15
        self.BP_LOOKAHEAD_TIME = 1.5  # s 1.5
        self.BP_LOOKAHEAD_BASE_CITY = 18  # m 25
        self.BP_LOOKAHEAD_TIME_CITY = 1.8  # s 2.5
        self.PATH_OFFSET = 1.0  # m 1.0
        self.CIRCLE_OFFSETS = [-1.0, 1.0, 3.0]  # m
        self.CIRCLE_RADII = [1.5, 1.5, 1.5]  # m
        self.TIME_GAP = 1.0  # s 1.0
        self.PATH_SELECT_WEIGHT = 10
        self.A_MAX = self.hyperparameters['max_accel']  # m/s^2
        self.SLOW_SPEED = 0.0  # m/s
        self.HIGH_SPEED = 40.0  # m/s
        self.STOP_LINE_BUFFER = 3.5  # m 3.5
        self.LP_FREQUENCY_DIVISOR = 1  # Frequency divisor to make the
                                    # local planner operate at a lower
                                    # frequency than the controller
                                    # (which operates at the simulation
                                    # frequency). Must be a natural
                                    # number.
        self.LP_FREQUENCY_DIVISOR_HIGHWAY = 1
        self.LP_FREQUENCY_DIVISOR_CITY = 1

        # Path interpolation parameters
        self.INTERP_DISTANCE_RES = 0.001  # distance between interpolated points
        self.lp = []
        self.bp = []
        self.scene_info = {}

    def init(self, scene_info):
        self.scenario_info = scene_info
        print('scene:', self.scenario_info)
        if self.scenario_info['type'] == 'REPLAY':
            self.init_replay()
        elif self.scenario_info['type'] == 'FRAGMENT':
            # print('fragement')
            self.init_replay()
            # return
        else:
            # print('not replay')
            self.init_replay()
            # return


# init_replay作用：
#1、读取仿真场景信息
#2、计算全局路径，利用get_waypoints获取
#3、初始化控制器controller2d
#4、设置停车目标
#5、初始化局部路径规划器
#6、初始化行为规划器
    def init_replay(self):
        self.start_pos = {
            'x': self.scenario_info['task_info']['startPos'][0],
            'y': self.scenario_info['task_info']['startPos'][1],
        }
        self.goal_pos = {
            'x': [self.scenario_info['task_info']['targetPos'][0][0], self.scenario_info['task_info']['targetPos'][1][0]],
            'y': [self.scenario_info['task_info']['targetPos'][0][1], self.scenario_info['task_info']['targetPos'][1][1]],
        }
        self.dt = self.scenario_info['task_info']['dt']
        self.waypoints = self.get_waypoints()

        if 'highway' in self.scenario_info['name'] and self.scenario_info['type'] == 'REPLAY':
            # print('hi_merge_replay')
            self.A_MAX = abs(self.pre_u0[0]) + abs(40 * self.dt)
        else:
            # print('no')
            self.A_MAX = abs(self.pre_u0[0]) + abs(self.JERK_LIMIT * self.dt)

        self.TIME_GAP = self.dt

        # print("waypoint:", self.waypoints)
        # # global path
        # if len(self.waypoints) < 3:
        #     self.waypoints = self.generate_global_path(self.start_pos['x'], self.start_pos['y'], np.mean(self.goal_pos['x']), np.mean(self.goal_pos['y']))

        self.controller = controller2d.Controller2D(self.waypoints)

        # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
        stopsign_fences = np.array([
            [self.goal_pos['x'][0], self.goal_pos['y'][0], self.goal_pos['x'][1], self.goal_pos['y'][1]]
        ])
        self.lp = local_planner.LocalPlanner(self.NUM_PATHS,
                                             self.PATH_OFFSET,
                                             self.CIRCLE_OFFSETS,
                                             self.CIRCLE_RADII,
                                             self.PATH_SELECT_WEIGHT,
                                             self.TIME_GAP,
                                             self.A_MAX,
                                             self.SLOW_SPEED,
                                             self.STOP_LINE_BUFFER)
        self.bp = behavioural_planner.BehaviouralPlanner(self.BP_LOOKAHEAD_BASE, stopsign_fences)

# act作用：基于当前环境信息计算车辆的控制动作

    def act(self, observation):
        #如果是简单串行场景，执行简单制动策略
        if self.scenario_info['type'] == 'SERIAL':
            # print('serial')
            # action = self.act_city_fragment(observation)
            # return action
            u2 = np.array([0.0, 0.0])
            if abs(observation.ego_info.v) > 0:
                u2[0] = -9.8
            else:
                u2[0] = 0
            return u2

        # action = self.act_city(observation)
        #如果是高速场景，则调用act_highway()
        if 'follow' in self.scenario_info['name'] or \
                'lanechanging' in self.scenario_info['name'] or \
                'highway' in self.scenario_info['name'] or \
                'cutin' in self.scenario_info['name']:
            # print('highway!')
            action = self.act_highway(observation)
        #下面是城市道路场景，分为Fragment和非Fragment模式，act_city（城市复杂场景）和act_city_fragment(处理局部复杂场景)
        else:
            # print('not highway')
            # action = self.act_city(observation)
            if self.scenario_info['type'] == 'FRAGMENT':
                # print('act')
                if self.scene_mid_inter:
                    if abs(self.start_pos['x'] - np.mean(self.goal_pos['x'])) < 5 or abs(
                            self.start_pos['y'] - np.mean(self.goal_pos['y'])) < 5:
                        action = self.act_city(observation)
                    else:
                        action = self.act_city_fragment(observation)
                else:
                    action = self.act_city_fragment(observation)
                # u2 = np.array([0.0, 0.0])
                # if abs(observation.ego_info.v) > 0:
                #     u2[0] = -9.8
                # else:
                #     u2[0] = 0
                # return u2
            else:
                action = self.act_city(observation)
        # print('ation:', action)
        return action


# 该函数的主要功能是 计算全局路径，并根据不同场景 选择合适的路径规划方法。
# 它首先 分析当前场景（如高速公路、交叉口、汇入场景等），然后 选择不同的路径规划策略（A* 或 global_path），
# 最后 返回路径点 waypoint

#1、调用AStar_Global_Planner解析场景类型
#2、根据场景标志
#3、判断是否在特殊场景（交叉口）调整控制模式
#4、选择不同的路径规划方法
#5、如果路径规划失败，则使用折线路径
#6、返回waypoint
    def get_waypoints(self):

        #解析场景类型，并且更新场景标志
        scene_judge = AStar_Global_Planner(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
        scene_judge.scene_choose()
        self.scene_merge = scene_judge.scene_merge
        self.scene_round = scene_judge.scene_round
        self.scene_mid_inter = scene_judge.scene_mid_inter
        self.scene_mix_straight = scene_judge.scene_mix_straight
        self.scene_crossing = scene_judge.scene_crossing
        self.scene_intersection = scene_judge.scene_intersection
        self.scene_straight_straight = scene_judge.scene_straight_straight
        # print('round:', self.scene_round)
        # print('merge:', self.scene_merge)

        #交叉口特殊处理
        if self.scene_mid_inter:
        # if self.scene_mid_inter:
            if abs(self.start_pos['x'] - np.mean(self.goal_pos['x'])) < 5 or abs(
                        self.start_pos['y'] - np.mean(self.goal_pos['y'])) < 5:
                self.None_later_control_city = False
        # # print('city_none_control:', self.None_later_control_city)
        # path_ang = math.atan2((self.waypoints[k][1] - obs['vehicle_info']['ego']['y']), (self.waypoints[k][0] - obs['vehicle_info']['ego']['x']))

        #选择路径规划策略
        if self.None_later_control == True:#是否使用默认的全局路径规划模式
            gp = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
            waypoint = gp.global_path_planning()
        else:
            #涉及跟车、变道、加塞
            if 'follow' in self.scenario_info['name'] or \
                    'lanechanging' in self.scenario_info['name'] or \
                    'cutin' in self.scenario_info['name']:  # or \
                    # 'highway' in self.scenario_info['name']:
                # print('highway!')
                gp_high = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                waypoint = gp_high.global_path_planning()

            #高速公路模式
            elif 'highway' in self.scenario_info['name']:
                # astar_path = AStar_Global_Planner(self.start_pos, self.goal_pos,
                #                                   self.scenario_info['source_file']['xodr'])
                # waypoint = astar_path.get_astar_path()
                gp_high = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                waypoint = gp_high.global_path_planning()

            # 否则采用A*方式规划全局路径
            else:
                # print('not highway!')
                if self.None_later_control_city == True:
                    gp = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                    waypoint = gp.global_path_planning()
                elif self.scenario_info['type'] == 'SERIAL':
                    gp = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                    waypoint = gp.global_path_planning()
                else:#否则则使用A*算法规划
                    astar_path = AStar_Global_Planner(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                    waypoint = astar_path.get_astar_path()

            # 若A*规划失败，使用折线规划
            if waypoint == [] or len(waypoint) < 5:
                # print('waypoint is empty!')
                gp_none = global_path(self.start_pos, self.goal_pos, self.scenario_info['source_file']['xodr'])
                waypoint = gp_none.global_path_planning()

        # plt.scatter(waypoint[:][0], waypoint[:][1], color="C3", s=0.1)
        # print("waypoint:", waypoint)
        # print("get waypoint")

        return waypoint

#1、提取自车的状态信息
#2、提取测试信息
    def obs_change(self, observation):
        # observation = controller.observation
        # scenario_info = controller.scenario_info
        obs = {
            'vehicle_info': {
                'ego': {
                    'a': observation.ego_info.a,
                    'rot': observation.ego_info.rot,
                    'x': observation.ego_info.x,
                    'y': observation.ego_info.y,
                    'v': observation.ego_info.v,
                    'yaw': observation.ego_info.yaw,
                    'width': observation.ego_info.width,
                    'length': observation.ego_info.length,
                },

            },
            'test_setting': {
                'goal': {
                    'x': [self.scenario_info['task_info']['targetPos'][0][0], self.scenario_info['task_info']['targetPos'][1][0]],
                    'y': [self.scenario_info['task_info']['targetPos'][0][1], self.scenario_info['task_info']['targetPos'][1][1]],
                },
                't': observation.test_info['t'],
                'dt': observation.test_info['dt'],
                'end': observation.test_info['end'],
            },
            'test_info:': {
                't': observation.test_info['t'],
                'dt': observation.test_info['dt'],
                'end': observation.test_info['end'],

            }
        }

        # print('veh_info:', obs)
        return obs

#将计算出的控制指令转换为加速 (throttle)、转向 (steer)、刹车 (brake) 命令，并映射到最终的控制变量 u0
    # 控制输出接口
    def send_control_command(self, throttle=0.0, steer=0.0, brake=0.0,
                            hand_brake=False, reverse=False):

        # control = VehicleControl()

        # Clamp all values within their limits
        steer = np.fmax(np.fmin(steer, 1.0), -1.0)
        throttle = np.fmax(np.fmin(throttle, 1.0), 0)
        brake = np.fmax(np.fmin(brake, 1.0), 0)

        if throttle > 0:
            accelerate = throttle
        else:
            accelerate = - brake

        # 映射到加减速度范围为[-3, 3]
        accelerate *= self.hyperparameters['max_accel']

        u0 = np.array([accelerate, steer])

        return u0


#1、获取传感器数据
#2、障碍物检测和预测
#3、计算局部路径
#4、检测前车、后车信息
#5、评估可行路径
#6、计算速度
#7、选择最佳路径
#8、输出控制指令
    def act_highway(self, observation):
        print('observation:', observation)
        # print('observation:', observation.test_info['t'])
        self.scene_t = observation.test_info['t']
        obs = self.obs_change(observation)
        # print('obs:', obs)
        middle_lead_car_found = False
        left_lead_car_found = False
        right_lead_car_found = False
        middle_follow_car_found = False
        left_follow_car_found = False
        right_follow_car_found = False
        # 障碍物预测
        nearby_vehicles_ago = extract_nearby_vehicles(observation, True)
        predict_array = obstacle_prediction(nearby_vehicles_ago, self.dt)

        if self.None_later_control_highway == True:
            small_path_x = []
            small_path_y = []
            for m in range(0, 250, 1):
                path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                small_path_x.append(path_x)
                small_path_y.append(path_y)
            v_max_global = np.ones(len(small_path_x)) * 10
            self.waypoints = np.stack((small_path_x, small_path_y, v_max_global), axis=1)

        lead_car_info = {
            'middle':{
                'pre_long': float('inf')
            },
            'left':{
                'pre_long': float('inf')
            },
            'right':{
                'pre_long': float('inf')
            },
        }

        follow_car_info = {
            'middle':{
                'pre_long': -float('inf')
            },
            'left':{
                'pre_long': -float('inf')
            },
            'right':{
                'pre_long': -float('inf')
            },
        }

        self_x = obs['vehicle_info']['ego']['x']
        self_y = obs['vehicle_info']['ego']['y']
        self_yaw = obs['vehicle_info']['ego']['yaw']
        self_width = obs['vehicle_info']['ego']['width']
        self_length = obs['vehicle_info']['ego']['length']
        self_v = obs['vehicle_info']['ego']['v']

        for vehi in observation.object_info['vehicle'].values():
            # print('vehi:', vehi)
            self_x = obs['vehicle_info']['ego']['x']
            self_y = obs['vehicle_info']['ego']['y']
            self_yaw = obs['vehicle_info']['ego']['yaw']
            self_width = obs['vehicle_info']['ego']['width']
            # road_info = parse_opendrive(str(self.scenario_info['name']))
            # road_width = abs(road_info.discretelanes[0].left_vertices[0][1] - \
            #                  road_info.discretelanes[0].right_vertices[0][1])
            road_width = 3.5
            vec_x = vehi.x - self_x
            vec_y = vehi.y - self_y
            # print('vec_x:', vec_x)

            long = vec_x * math.cos(self_yaw) + vec_y * math.sin(self_yaw)
            lat = - vec_x * math.sin(self_yaw) + vec_y * math.cos(self_yaw)
            err = road_width + (vehi.width + self_width) / 2
            if long > 0:
                if abs(lat) < (vehi.width + self_width) / 2 + 0.8:
                    if long < lead_car_info['middle']['pre_long']:
                        lead_car_info['middle']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        lead_car_info['middle']['x'] = vehi.x
                        lead_car_info['middle']['y'] = vehi.y
                        lead_car_info['middle']['v'] = vehi.v
                        lead_car_info['middle']['a'] = vehi.a
                        lead_car_info['middle']['yaw'] = vehi.yaw
                        lead_car_info['middle']['width'] = vehi.width
                        lead_car_info['middle']['length'] = vehi.length
                        middle_lead_car_found = True
                elif lat > 0:
                    if long < lead_car_info['left']['pre_long'] and \
                        lat < err:
                        lead_car_info['left']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                            # lead_car_info['left'][key] = vehi[1].key
                        lead_car_info['left']['x'] = vehi.x
                        lead_car_info['left']['y'] = vehi.y
                        lead_car_info['left']['v'] = vehi.v
                        lead_car_info['left']['a'] = vehi.a
                        lead_car_info['left']['yaw'] = vehi.yaw
                        lead_car_info['left']['width'] = vehi.width
                        lead_car_info['left']['length'] = vehi.length
                        left_lead_car_found = True
                else:
                    if long < lead_car_info['right']['pre_long'] and \
                        lat > -err:
                        lead_car_info['right']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     lead_car_info['right'][key] = vehi[1].key
                        lead_car_info['right']['x'] = vehi.x
                        lead_car_info['right']['y'] = vehi.y
                        lead_car_info['right']['v'] = vehi.v
                        lead_car_info['right']['a'] = vehi.a
                        lead_car_info['right']['yaw'] = vehi.yaw
                        lead_car_info['right']['width'] = vehi.width
                        lead_car_info['right']['length'] = vehi.length
                        right_lead_car_found = True
            else:
                if abs(lat) < (vehi.width + self_width) / 2 + 0.8:
                    if long > follow_car_info['middle']['pre_long']:
                        follow_car_info['middle']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['middle'][key] = vehi[1].key
                        follow_car_info['middle']['x'] = vehi.x
                        follow_car_info['middle']['y'] = vehi.y
                        follow_car_info['middle']['v'] = vehi.v
                        follow_car_info['middle']['a'] = vehi.a
                        follow_car_info['middle']['yaw'] = vehi.yaw
                        follow_car_info['middle']['width'] = vehi.width
                        follow_car_info['middle']['length'] = vehi.length
                        middle_follow_car_found = True
                elif lat > 0:
                    if long > follow_car_info['left']['pre_long'] and \
                        lat < err:
                        follow_car_info['left']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['left'][key] = vehi[1].key
                        follow_car_info['left']['x'] = vehi.x
                        follow_car_info['left']['y'] = vehi.y
                        follow_car_info['left']['v'] = vehi.v
                        follow_car_info['left']['a'] = vehi.a
                        follow_car_info['left']['yaw'] = vehi.yaw
                        follow_car_info['left']['width'] = vehi.width
                        follow_car_info['left']['length'] = vehi.length
                        left_follow_car_found = True
                else:
                    if long > follow_car_info['right']['pre_long'] and \
                        lat > -err:
                        follow_car_info['right']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['right'][key] = vehi[1].key
                        follow_car_info['right']['x'] = vehi.x
                        follow_car_info['right']['y'] = vehi.y
                        follow_car_info['right']['v'] = vehi.v
                        follow_car_info['right']['a'] = vehi.a
                        follow_car_info['right']['yaw'] = vehi.yaw
                        follow_car_info['right']['width'] = vehi.width
                        follow_car_info['right']['length'] = vehi.length
                        right_follow_car_found = True

        # Obtain Lead Vehicle information.
        if middle_lead_car_found:
            lead_car_pos = np.array([
                [lead_car_info['middle']['x'], lead_car_info['middle']['y'], lead_car_info['middle']['yaw']]])
            lead_car_length = np.array([
                [lead_car_info['middle']['length']]])
            lead_car_speed = np.array([
                [lead_car_info['middle']['v']]])
        else:
            lead_car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            lead_car_length = np.array([
                [99999.0]])
            lead_car_speed = np.array([
                [99999.0]])
            
        # Obtain Follow Vehicle information.    
        if middle_follow_car_found:
            follow_car_pos = np.array([
                [follow_car_info['middle']['x'], follow_car_info['middle']['y'], follow_car_info['middle']['yaw']]])
            follow_car_length = np.array([
                [follow_car_info['middle']['length']]])
            follow_car_speed = np.array([
                [follow_car_info['middle']['v']]])
        else:
            follow_car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            follow_car_length = np.array([
                [99999.0]])
            follow_car_speed = np.array([
                [99999.0]])

        # 当前参数获取
        current_x, current_y, current_yaw, current_speed, ego_length = \
            [obs['vehicle_info']['ego'][key] for key in ['x', 'y', 'yaw', 'v', 'length']]
        current_timestamp = obs['test_setting']['t'] + \
                            obs['test_setting']['dt']
        self_Car = [current_speed, current_yaw]

        # Obtain parkedcar_box_pts
        parkedcar_box_pts = []  # [x,y]
        parkedcar_box_pts_2 = []
        parkedcar_box_pts_3 = []

        # 与当前车道的前车进行速度判断，若小于当前车速按照障碍车绕行
        if middle_lead_car_found and \
            lead_car_info['middle']['v'] < self.hyperparameters['overtaking_speed'] and \
                lead_car_info['middle']['v'] < obs['vehicle_info']['ego']['v']:#obs.v < self.v: #与onsite接口对接
            parkedcar_num = 0
            # 只需要当前和左右车道中最近的前车的信息
            for vehi in lead_car_info.items():
                id = vehi[0]
                if 'v' in lead_car_info[id].keys():
                          
                    x, y, yaw, xrad, yrad, obs_v = [vehi[1][key] for key in ['x', 'y', 'yaw', 'length', 'width', 'v']]
                    
                    xrad = 0.5 * xrad # 0.5 * 车长
                    yrad = 0.5 * yrad # 0.5 * 车宽
                    ox = x + obs_v * np.cos(yaw) * 0.25
                    oy = y + obs_v * np.sin(yaw) * 0.25
                    ox_2 = x + obs_v * np.cos(yaw) * 0.8
                    oy_2 = y + obs_v * np.sin(yaw) * 0.8
                    ox_3 = x + obs_v * np.cos(yaw) * 1.2
                    oy_3 = y + obs_v * np.sin(yaw) * 1.2

                    cpos = np.array([
                        [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0,
                         -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                        [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad,
                         -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
                    rotyaw = np.array([
                        [np.cos(yaw), np.sin(yaw)],
                        [-np.sin(yaw), np.cos(yaw)]])
                    cpos_shift = np.array([
                        [x, x, x, x, x, x, x, x, ox, ox, ox, ox, ox, ox, ox, ox, ox_2, ox_2, ox_2, ox_2, ox_2, ox_2,
                         ox_2, ox_2, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3],
                        [y, y, y, y, y, y, y, y, oy, oy, oy, oy, oy, oy, oy, oy, oy_2, oy_2, oy_2, oy_2, oy_2, oy_2,
                         oy_2, oy_2, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3]])
                    cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
                    
                    if len(parkedcar_box_pts) <= parkedcar_num:
                            parkedcar_box_pts.append([])

                    for j in range(cpos.shape[1]):
                        parkedcar_box_pts[parkedcar_num].append([cpos[0, j], cpos[1, j]])
                    # print("lead")
                    parkedcar_num += 1

        # '''
        # follow
        parkedcar_num = 0
        for vehi in follow_car_info.items():
            id = vehi[0]
            if 'v' in follow_car_info[id].keys():

                x, y, yaw, xrad, yrad, obs_v = [vehi[1][key] for key in ['x', 'y', 'yaw', 'length', 'width', 'v']]

                xrad = 0.6 * xrad  # 0.5 * 车长
                yrad = 0.5 * yrad  # 0.5 * 车宽

                ox = x + obs_v * np.cos(yaw) * 0.25
                oy = y + obs_v * np.sin(yaw) * 0.25
                ox_2 = x + obs_v * np.cos(yaw) * 0.5
                oy_2 = y + obs_v * np.sin(yaw) * 0.5
                ox_3 = x + obs_v * np.cos(yaw) * 1.2
                oy_3 = y + obs_v * np.sin(yaw) * 1.2


                cpos = np.array([
                    [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad,
                     -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                    [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0,
                     yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
                rotyaw = np.array([
                    [np.cos(yaw), np.sin(yaw)],
                    [-np.sin(yaw), np.cos(yaw)]])
                cpos_shift = np.array([
                    [x, x, x, x, x, x, x, x, ox, ox, ox, ox, ox, ox, ox, ox, ox_2, ox_2, ox_2, ox_2, ox_2, ox_2, ox_2,
                     ox_2, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3],
                    [y, y, y, y, y, y, y, y, oy, oy, oy, oy, oy, oy, oy, oy, oy_2, oy_2, oy_2, oy_2, oy_2, oy_2, oy_2,
                     oy_2, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3]])

                cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)


                if len(parkedcar_box_pts_2) <= parkedcar_num:
                    parkedcar_box_pts_2.append([])

                for j in range(cpos.shape[1]):
                    parkedcar_box_pts_2[parkedcar_num].append([cpos[0, j], cpos[1, j]])

                # print("follow")

                parkedcar_num += 1

        # lead
        parkedcar_num = 0
        for vehi in lead_car_info.items():
            id = vehi[0]
            if 'v' in lead_car_info[id].keys():

                x, y, yaw, xrad, yrad, obs_v = [vehi[1][key] for key in ['x', 'y', 'yaw', 'length', 'width', 'v']]
                dx = x - current_x
                dy = y - current_y
                lead_dis = np.sqrt(dx**2 + dy**2)
                lead_dy = np.sqrt(dy ** 2)
                if lead_dy > 0.8:
                    if lead_dis < 1.5*ego_length:
                        xrad = 0.6 * xrad  # 0.5 * 车长
                        yrad = 0.5 * yrad  # 0.5 * 车宽

                        ox = x - obs_v * np.cos(yaw) * 0.1
                        oy = y - obs_v * np.sin(yaw) * 0.1
                        ox_2 = x + obs_v * np.cos(yaw) * 0.5
                        oy_2 = y + obs_v * np.sin(yaw) * 0.5
                        ox_3 = x + obs_v * np.cos(yaw) * 1.0
                        oy_3 = y + obs_v * np.sin(yaw) * 1.0

                        cpos = np.array([
                            [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0,
                             -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0, -xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                            [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad,
                             -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad, -yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
                        rotyaw = np.array([
                            [np.cos(yaw), np.sin(yaw)],
                            [-np.sin(yaw), np.cos(yaw)]])
                        cpos_shift = np.array([
                            [x, x, x, x, x, x, x, x, ox, ox, ox, ox, ox, ox, ox, ox, ox_2, ox_2, ox_2, ox_2, ox_2, ox_2,
                             ox_2, ox_2, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3, ox_3],
                            [y, y, y, y, y, y, y, y, oy, oy, oy, oy, oy, oy, oy, oy, oy_2, oy_2, oy_2, oy_2, oy_2, oy_2,
                             oy_2, oy_2, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3, oy_3]])

                        cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)

                        if len(parkedcar_box_pts_3) <= parkedcar_num:
                            parkedcar_box_pts_3.append([])

                        for j in range(cpos.shape[1]):
                            parkedcar_box_pts_3[parkedcar_num].append([cpos[0, j], cpos[1, j]])

                        # print("side")

                        parkedcar_num += 1





        local_waypoints = None
        path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)
        
        reached_the_end = False

        # Update pose and timestamp
        prev_timestamp = obs['test_setting']['t']




        if self.frame % self.LP_FREQUENCY_DIVISOR_HIGHWAY == 0:

            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp, obs['vehicle_info']['ego']['v'])

            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0], lead_car_length[0][0]]
            follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2], follow_car_speed[0][0], follow_car_length[0][0]]
            
            # Set lookahead based on current speed.
            self.bp.set_lookahead(self.BP_LOOKAHEAD_BASE + self.BP_LOOKAHEAD_TIME * open_loop_speed)

            # Perform a state transition in the behavioural planner.
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            # Check to see if we need to follow the lead vehicle.
            veh_lon_index = 0
            if obs['vehicle_info']['ego']['length'] - 5 > 0:
                veh_lon_index = 1.8*(abs(obs['vehicle_info']['ego']['length'] - 5))
            if self.scenario_info['type'] == 'FRAGMENT':
                dist_gap = 8.0 + veh_lon_index  # 2.0
                min_foll_index = 6
                foll_index = 4
                # if 'merge' in self.scenario_info['name']:
                #     dist_gap = 13.0 + veh_lon_index  # 2.0
                #     min_foll_index = 6
                #     foll_index = 4
            else:
                if 'merge' in self.scenario_info['name']:
                    dist_gap = 15.0 + veh_lon_index * 0.5  # 2.0
                    min_foll_index = 0
                    foll_index = 0
                else:
                    dist_gap = 3.0 + veh_lon_index  # 2.0
                    min_foll_index = 0
                    foll_index = 0
                # dist_gap = 3.0 + veh_lon_index  # 2.0
                # min_foll_index = 0
                # foll_index = 0
                # if 'merge' in self.scenario_info['name']:
                #     dist_gap = 8.0 + veh_lon_index  # 2.0
                #     min_foll_index = 0
                #     foll_index = 0
            ego_rear_dist = self.calc_car_dist(ego_state, follow_car_state) - dist_gap
            ego_rear_dist = ego_rear_dist if ego_rear_dist > 0 else 1e-8
            ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) - dist_gap
            # ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) + dist_gap
            ego_front_dist = ego_front_dist if ego_front_dist > 0 else 1e-8
            a = (2 * follow_car_speed[0][0] - current_speed) * current_speed / (2 * ego_rear_dist)
            a_x = max(1e-8, min(self.hyperparameters['max_accel'], a))
            min_follow_dist = (current_speed ** 2 - lead_car_speed[0][0] ** 2) / (2 * self.hyperparameters['max_accel']) + dist_gap + min_foll_index
            follow_dist = current_speed ** 2 / (2 * a_x) - lead_car_speed[0][0] ** 2 / (2 * self.hyperparameters['max_accel']) + foll_index
            if follow_dist > ego_front_dist:
                follow_dist = min_follow_dist
            LEAD_VEHICLE_LOOKAHEAD = follow_dist if follow_dist > 0 else 0
            # LEAD_VEHICLE_LOOKAHEAD = 15
            self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0], LEAD_VEHICLE_LOOKAHEAD)

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints, ego_state)

            # print('pre path')
            # Calculate planned paths in the local frame.
            paths, path_validity = self.lp.plan_paths(goal_state_set)

            # print('path done')
            # speed_ac = velocity_planner.speed_gen(paths, current_speed, current_speed + 10, self.SLOW_SPEED, self.HIGH_SPEED)
            # speed_de = velocity_planner.speed_gen(paths, current_speed, current_speed - 10, self.SLOW_SPEED, self.HIGH_SPEED)
            # speed_ho = velocity_planner.speed_gen(paths, current_speed, current_speed, self.SLOW_SPEED, self.HIGH_SPEED)
            # print(speed_ac)
            # print(speed_de)
            # print(speed_ho)
            # print(paths)

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)

            # Perform collision checking.
            # collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts, parkedcar_box_pts_2, parkedcar_box_pts_3, self_Car)
            # # print('coll0428')
            # self_car = [self_v, self_yaw, self_width, self_length]
            # # print(self_car)
            # collision_check_array = collision_check(paths, predict_array, self_car, self.dt)
            if 'highway' in self.scenario_info['name']:
                # coll 0428
                # print('coll0428')
                self_car = [self_v, self_yaw, self_width, self_length]
                # print(self_car)
                collision_check_array = collision_check(paths, predict_array, self_car, self.dt)
                # print(collision_check_array)
            else:
                # print('coll_old')
                collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts, parkedcar_box_pts_2, parkedcar_box_pts_3, self_Car)

            # for i in range(min(len(path_validity), len(collision_check_array))):
            #     if path_validity[i] == False or collision_check_array[i] == False:
            #         collision_check_array[i] == False
            # print(collision_check_array)

            # # Compute the best local path.
            # if self.lp._prev_best_path == None:
            #     best_index = int(len(paths) / 2)
            # else:
            #     best_index = self.lp._collision_checker.select_best_path_index(paths, collision_check_array, self.bp._goal_state)
            # # If no path was feasible, continue to follow the previous best path.
            # if best_index == None or paths == []:
            #     best_path = self.lp._prev_best_path
            # else:
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path
            best_index = self.lp._collision_checker.select_best_path_index(paths, collision_check_array)

            self.path_all_false = True
            for kt in range(len(paths)):
                # print(kt)
                if collision_check_array[kt] == False:
                    self.path_all_false = False


            # if best_index == None or paths == []:
            #     best_path = self.lp._prev_best_path
            # else:
            #     print('path_num:', len(paths))
            #     print('best_index:', best_index)
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path

            # best_index = int(len(paths) / 2)
            # while collision_check_array[best_index] == False:
            #     best_index = best_index + 1
            #     if best_index == len(paths):
            #         best_index = 0

            # if  paths != []:
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path
            # else:
            #     best_path = self.lp._prev_best_path
            if self.None_later_control_highway == True:
                best_path = paths[int(len(paths) / 2)]
            else:
                if paths != []:
                    best_path = paths[best_index]
                    self.lp._prev_best_path = best_path
                else:
                    best_path = self.lp._prev_best_path

            goal_dist = np.sqrt((obs['vehicle_info']['ego']['x'] - np.mean(self.goal_pos['x'])) ** 2 + (obs['vehicle_info']['ego']['y'] - np.mean(self.goal_pos['y'])) ** 2)
            goal_y_dist = abs(obs['vehicle_info']['ego']['y'] - np.mean(self.goal_pos['y']))
            if goal_y_dist < 3.5:
                best_path = paths[int(len(paths) / 2)]

            # all_false_index = True
            # print(collision_check_array)
            # print(len(collision_check_array))

            # for kt in range(len(paths)):
            #     # print(kt)
            #     if  collision_check_array[kt] == True:
            #         all_false_index = False

            over_goal = False
            if obs['vehicle_info']['ego']['x'] > np.mean(self.goal_pos['x']) and obs['vehicle_info']['ego']['x'] > np.mean(self.start_pos['x']):
                over_goal = True
            elif obs['vehicle_info']['ego']['x'] < np.mean(self.goal_pos['x']) and obs['vehicle_info']['ego']['x'] < np.mean(self.start_pos['x']):
                over_goal = True

            if over_goal == True:
                # print('over goal!')
                small_path_x = []
                small_path_y = []
                small_path_v = []
                for m in range(0, 250, 1):
                    path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                    path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                    small_path_x.append(path_x)
                    small_path_y.append(path_y)
                    small_path_v.append(25)
                best_path = [[0, 0], [0, 0], [0, 0]]
                best_path[0] = small_path_x
                best_path[1] = small_path_y
                best_path[2] = small_path_v

            # if best_index > len(paths) - 1:
            #     print('best>len')
            #     best_path = self.lp._prev_best_path
            # else:
            #     # best_path = self.lp._prev_best_path
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path

            # if best_index < len(paths) / 2:
            #     lead_car_pos, lead_car_length, lead_car_speed = \
            #         self.get_lead_follow_car_info(left_lead_car_found, lead_car_info, 'left')
            #     follow_car_pos, follow_car_length, follow_car_speed = \
            #         self.get_lead_follow_car_info(left_follow_car_found, follow_car_info, 'left')
            # elif best_index > len(paths) / 2:
            #     lead_car_pos, lead_car_length, lead_car_speed = \
            #         self.get_lead_follow_car_info(right_lead_car_found, lead_car_info, 'right')
            #     follow_car_pos, follow_car_length, follow_car_speed = \
            #         self.get_lead_follow_car_info(right_follow_car_found, follow_car_info, 'right')
            # # car_state
            # lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0], lead_car_length[0][0]]
            # follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2], follow_car_speed[0][0], follow_car_length[0][0]]
            # # 重新计算是否跟车
            # # Check to see if we need to follow the lead vehicle.
            # dist_gap = 2.0
            # ego_rear_dist = self.calc_car_dist(ego_state, follow_car_state) - dist_gap
            # ego_rear_dist = ego_rear_dist if ego_rear_dist > 0 else 1e-8
            # ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) - dist_gap
            # ego_front_dist = ego_front_dist if ego_front_dist > 0 else 1e-8
            # a = (2 * follow_car_speed[0][0] - current_speed) * current_speed / (2 * ego_rear_dist)
            # a_x = max(1e-8, min(self.hyperparameters['max_accel'], a))
            # min_follow_dist = (current_speed ** 2 - lead_car_speed[0][0] ** 2) / (2 * self.hyperparameters['max_accel']) + dist_gap
            # follow_dist = current_speed ** 2 / (2 * a_x) - lead_car_speed[0][0] ** 2 / (2 * self.hyperparameters['max_accel'])
            # if follow_dist > ego_front_dist:
            #     follow_dist = min_follow_dist
            # LEAD_VEHICLE_LOOKAHEAD = follow_dist if follow_dist > 0 else 0
            # # LEAD_VEHICLE_LOOKAHEAD = 15
            # self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0], LEAD_VEHICLE_LOOKAHEAD)

            # Compute the velocity profile for the path, and compute the waypoints.
            # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
            # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
            # if middle_lead_car_found or best_index != len(paths) / 2:
            if middle_lead_car_found:
                desired_speed = np.sqrt(lead_car_speed[0][0] ** 2 + 2 * ego_front_dist * self.hyperparameters['max_accel']) * 0.95
            else:
                desired_speed = self.bp._goal_state[2]
            decelerate_to_stop = self.bp._state == behavioural_planner.DECELERATE_TO_STOP
            local_waypoints = self.lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
                                                                            current_speed, decelerate_to_stop,
                                                                            lead_car_state, self.bp._follow_lead_vehicle)
            # --------------------------------------------------------------

            if local_waypoints != None:
                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.  Linear interpolation computation on the waypoints
                # is also used to ensure a fine resolution between points.
                wp_distance = []  # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                        np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                                (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp = []  # interpolated values
                # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))

                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                    float(self.INTERP_DISTANCE_RES)) - 1)
                    wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = self.INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))

                # Update the other controller values and controls
                self.controller.update_waypoints(wp_interp)

        ###
        # Controller Update
        ###
        # if local_waypoints != None and local_waypoints != []:
        #     self.controller.update_values(current_x, current_y, current_yaw,
        #                             current_speed,
        #                             current_timestamp, frame)
        #     self.controller.update_controls()
        #     cmd_throttle, cmd_steer, cmd_brake = self.controller.get_commands()
        #     # 尝试输出控制值
        #     u0 = self.send_control_command(throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
        # else:
        #     cmd_throttle = 0.0
        #     cmd_steer = 0.0
        #     cmd_brake = 0.0
        #     u0 = self.send_control_command(throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
        
        if local_waypoints != None:
            self.pre_local_waypoints = local_waypoints

        if local_waypoints != None and local_waypoints != []:
            u0 = self.generate_control(local_waypoints, obs)
            self.pre_u0 = u0
        elif self.pre_local_waypoints != None:
            u0 = self.generate_control(self.pre_local_waypoints, obs)
            self.pre_u0 = u0
        else:
            u0 = self.pre_u0

        self.frame += 1
        # print('action:', u0)

        return u0
#主要功能和act_highway函数一样，只是适用场景不同
    def act_city(self, observation):
        # 数据格式处理
        obs = self.obs_change(observation)
        self.scene_t = observation.test_info['t']
        # 障碍物预测
        nearby_vehicles_ago = extract_nearby_vehicles(observation, False)
        predict_array = obstacle_prediction(nearby_vehicles_ago, self.dt)
        # print('obs:', obs)
        print('observation:', observation)
        # print('observation:', observation.light_info, type(observation.light_info))
        # print('pre_veh:', predict_array )
        # # 坐标是否需要镜像变换
        # REVE_X = False
        # REVE_Y = False
        # # 坐标是否需要平移变换
        # DIFF_X = False
        # DIFF_Y = False
        # # 判断起点与终点是否在同一象限
        # if round(self.start_pos['x'], 1) < 0 or round(np.mean(self.goal_pos['x']), 1) < 0:
        #     if round(self.start_pos['x'], 1) < 0 and round(np.mean(self.goal_pos['x']), 1) < 0:
        #         REVE_X = True
        #         # print('reve_x:', REVE_X)
        #     else:
        #         DIFF_X = True
        #         # print('diff_x:', DIFF_X)
        # if round(self.start_pos['y'], 1) < 0 or round(np.mean(self.goal_pos['y']), 1) < 0:
        #     if round(self.start_pos['y'], 1) < 0 and round(np.mean(self.goal_pos['y']), 1) < 0:
        #         REVE_Y = True
        #         # print('reve_y:', REVE_Y)
        #     else:
        #         DIFF_Y = True
        #         # print('diff_y:', DIFF_Y)
        #
        #         # 根据前面判断将x或y坐标进行调整（镜像与平移）
        #         if REVE_X == True:
        #             self.start_pos['x'] = abs(self.start_pos['x'])
        #             self.goal_pos['x'][0] = abs(self.goal_pos['x'][0])
        #             self.goal_pos['x'][1] = abs(self.goal_pos['x'][1])
        #
        #         if REVE_Y == True:
        #             self.start_pos['y'] = abs(self.start_pos['y'])
        #             self.goal_pos['y'][0] = abs(self.goal_pos['y'][0])
        #             self.goal_pos['y'][1] = abs(self.goal_pos['y'][1])
        #
        #         if DIFF_X == True:
        #             if self.start_pos['x'] < 0:
        #                 miss_x = abs(0 - self.start_pos['x'])
        #             elif np.mean(self.goal_pos['x']) < 0:
        #                 miss_x = abs(0 - np.mean(self.goal_pos['x']))
        #             self.start_pos['x'] = self.start_pos['x'] + miss_x
        #             self.goal_pos['x'][0] = self.goal_pos['x'][0] + miss_x
        #             self.goal_pos['x'][1] = self.goal_pos['x'][1] + miss_x
        #
        #         if DIFF_Y == True:
        #             if self.start_pos['y'] < 0:
        #                 miss_y = abs(0 - self.start_pos['y'])
        #             elif np.mean(self.goal_pos['y']) < 0:
        #                 miss_y = abs(0 - np.mean(self.goal_pos['y']))
        #             self.start_pos['y'] = self.start_pos['y'] + miss_y
        #             self.goal_pos['y'][0] = self.goal_pos['y'][0] + miss_y
        #             self.goal_pos['y'][1] = self.goal_pos['y'][1] + miss_y
        way_ang = [0,0,0,0]
        way_ang[0] = math.atan2((self.goal_pos['y'][0] - obs['vehicle_info']['ego']['y']),
                               (self.goal_pos['x'][0] - obs['vehicle_info']['ego']['x']))
        way_ang[1] = math.atan2((self.goal_pos['y'][0] - obs['vehicle_info']['ego']['y']),
                               (self.goal_pos['x'][1] - obs['vehicle_info']['ego']['x']))
        way_ang[2] = math.atan2((self.goal_pos['y'][1] - obs['vehicle_info']['ego']['y']),
                               (self.goal_pos['x'][0] - obs['vehicle_info']['ego']['x']))
        way_ang[3] = math.atan2((self.goal_pos['y'][1] - obs['vehicle_info']['ego']['y']),
                               (self.goal_pos['x'][1] - obs['vehicle_info']['ego']['x']))
        for m in range(len(way_ang)):
            if way_ang[m] < 0:
                way_ang[m] = way_ang[m] + 6.28
        get_to_the_goal = False
        for k in range(len(way_ang)):
            if obs['vehicle_info']['ego']['yaw'] <= way_ang[k]:
                for j in range(len(way_ang)):
                    if obs['vehicle_info']['ego']['yaw'] >= way_ang[j]:
                        get_to_the_goal = True

        if get_to_the_goal == True:
            self.None_later_control_city = True

        # print('way_ang_1:',way_ang[0])
        # print('way_ang_2:', way_ang[1])
        # print('way_ang_3:', way_ang[2])
        # print('way_ang_4:', way_ang[3])
        # print('veh_ang:', obs['vehicle_info']['ego']['yaw'])
        # print('get_goal:', get_to_the_goal)
        # print('None_later:', self.None_later_control_city)

        follow_time = 0
        # stop_dist = 10
        if self.frame == 15:
            self.Initial_v = obs['vehicle_info']['ego']['v']
        stop_dist = 5 + 1.7 * self.Initial_v
        veh_dist = pow(pow((obs['vehicle_info']['ego']['x'] - self.start_pos['x']), 2) + pow(
            (obs['vehicle_info']['ego']['y'] - self.start_pos['y']), 2), 0.5)

        self_x = obs['vehicle_info']['ego']['x']
        self_y = obs['vehicle_info']['ego']['y']
        self_yaw = obs['vehicle_info']['ego']['yaw']
        self_width = obs['vehicle_info']['ego']['width']
        self_length = obs['vehicle_info']['ego']['length']
        self_v = obs['vehicle_info']['ego']['v']
        if self.scene_t == 0:
            self.init_yaw = obs['vehicle_info']['ego']['yaw']
        # print('observation:')

        # 当前参数获取
        current_x, current_y, current_yaw, current_speed, ego_length = \
            [obs['vehicle_info']['ego'][key] for key in ['x', 'y', 'yaw', 'v', 'length']]
        current_timestamp = obs['test_setting']['t'] + \
                            obs['test_setting']['dt']
        self_Car = [current_speed, current_yaw]
        follow_car_info = {
            'middle': {
                'pre_long': -15
            },
            'left': {
                'pre_long': -float('inf')
            },
            'right': {
                'pre_long': -float('inf')
            },
        }
        middle_follow_car_found = False

        for vehi in observation.object_info['vehicle'].values():
            # print('vehi:', vehi)
            self_x = obs['vehicle_info']['ego']['x']
            self_y = obs['vehicle_info']['ego']['y']
            self_yaw = obs['vehicle_info']['ego']['yaw']
            self_width = obs['vehicle_info']['ego']['width']
            # road_width = abs(road_info.discretelanes[0].left_vertices[0][1] - \
            #                  road_info.discretelanes[0].right_vertices[0][1])
            road_width = 3.5
            vec_x = vehi.x - self_x
            vec_y = vehi.y - self_y
            # print('vec_x:', vec_x)

            long = vec_x * math.cos(self_yaw) + vec_y * math.sin(self_yaw)
            lat = - vec_x * math.sin(self_yaw) + vec_y * math.cos(self_yaw)
            err = road_width + (vehi.width + self_width) / 2
            if long > 0:
                pass
            else:
                if abs(lat) < (vehi.width + self_width) / 2 + 0.8:
                    if long > follow_car_info['middle']['pre_long']:
                        follow_car_info['middle']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['middle'][key] = vehi[1].key
                        follow_car_info['middle']['x'] = vehi.x
                        follow_car_info['middle']['y'] = vehi.y
                        follow_car_info['middle']['v'] = vehi.v
                        follow_car_info['middle']['a'] = vehi.a
                        follow_car_info['middle']['yaw'] = vehi.yaw
                        follow_car_info['middle']['width'] = vehi.width
                        follow_car_info['middle']['length'] = vehi.length
                        # if follow_car_info['middle']['v'] > 1.5:
                        middle_follow_car_found = True
                elif lat > 0:
                    if long > follow_car_info['left']['pre_long'] and \
                        lat < err:
                        follow_car_info['left']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['left'][key] = vehi[1].key
                        follow_car_info['left']['x'] = vehi.x
                        follow_car_info['left']['y'] = vehi.y
                        follow_car_info['left']['v'] = vehi.v
                        follow_car_info['left']['a'] = vehi.a
                        follow_car_info['left']['yaw'] = vehi.yaw
                        follow_car_info['left']['width'] = vehi.width
                        follow_car_info['left']['length'] = vehi.length
                        # middle_follow_car_found = True
                else:
                    if long > follow_car_info['right']['pre_long'] and \
                        lat > -err:
                        follow_car_info['right']['pre_long'] = long
                        # for key in vehi[1].keys():
                        #     print('keys:', key)
                        #     follow_car_info['right'][key] = vehi[1].key
                        follow_car_info['right']['x'] = vehi.x
                        follow_car_info['right']['y'] = vehi.y
                        follow_car_info['right']['v'] = vehi.v
                        follow_car_info['right']['a'] = vehi.a
                        follow_car_info['right']['yaw'] = vehi.yaw
                        follow_car_info['right']['width'] = vehi.width
                        follow_car_info['right']['length'] = vehi.length
                        # middle_follow_car_found = True

        # print('middle_follow_car_found:', middle_follow_car_found)


        local_waypoints = None
        path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)

        reached_the_end = False

        # Update pose and timestamp
        prev_timestamp = obs['test_setting']['t']
        # print("side")

        if self.frame % self.LP_FREQUENCY_DIVISOR_CITY == 0:
            if self.None_later_control_city == True:
                small_path_x = []
                small_path_y = []
                for m in range(0, 250, 1):
                    path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                    path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                    small_path_x.append(path_x)
                    small_path_y.append(path_y)
                v_max_global = np.ones(len(small_path_x)) * 10
                self.waypoints = np.stack((small_path_x, small_path_y, v_max_global), axis=1)

            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp,
                                                                            obs['vehicle_info']['ego']['v'])

            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            # lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0],
            #                   lead_car_length[0][0]]
            # follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2],
            #                     follow_car_speed[0][0], follow_car_length[0][0]]

            # Set lookahead based on current speed.
            self.bp.set_lookahead(self.BP_LOOKAHEAD_BASE_CITY + self.BP_LOOKAHEAD_TIME_CITY * open_loop_speed)

            # Perform a state transition in the behavioural planner.
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            # Check to see if we need to follow the lead vehicle.
            # dist_gap = 2.0
            # ego_rear_dist = self.calc_car_dist(ego_state, follow_car_state) - dist_gap
            # ego_rear_dist = ego_rear_dist if ego_rear_dist > 0 else 1e-8
            # ego_front_dist = self.calc_car_dist(ego_state, lead_car_state) - dist_gap
            # ego_front_dist = ego_front_dist if ego_front_dist > 0 else 1e-8
            # a = (2 * follow_car_speed[0][0] - current_speed) * current_speed / (2 * ego_rear_dist)
            # a_x = max(1e-8, min(self.hyperparameters['max_accel'], a))
            # min_follow_dist = (current_speed ** 2 - lead_car_speed[0][0] ** 2) / (
            #             2 * self.hyperparameters['max_accel']) + dist_gap
            # follow_dist = current_speed ** 2 / (2 * a_x) - lead_car_speed[0][0] ** 2 / (
            #             2 * self.hyperparameters['max_accel'])
            # if follow_dist > ego_front_dist:
            #     follow_dist = min_follow_dist
            # LEAD_VEHICLE_LOOKAHEAD = follow_dist if follow_dist > 0 else 0
            # # LEAD_VEHICLE_LOOKAHEAD = 15
            # self.bp.check_for_lead_vehicle(ego_state, lead_car_pos[0], LEAD_VEHICLE_LOOKAHEAD)

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)

            # print('pre path', goal_state_set)
            # Calculate planned paths in the local frame.
            paths, path_validity = self.lp.plan_paths(goal_state_set)
            # print('pathlen:', len(paths))

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)
            # print('pathlen:', len(paths))

            # Perform collision checking.
            # collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts,
            #                                                                    parkedcar_box_pts_2, parkedcar_box_pts_3,
            #                                                                    self_Car)

            # coll 0428
            self_car = [self_v, self_yaw, self_width, self_length]
            # print(self_car)
            collision_check_array = collision_check(paths, predict_array, self_car, self.dt)
            # print(collision_check_array)

            # print('path:', path_validity)
            for i in range(min(len(path_validity), len(collision_check_array))):
                if path_validity[i] == False or collision_check_array[i] == False:
                    collision_check_array[i] == False
            # print('coll:', collision_check_array)

            best_index = self.lp._collision_checker.select_best_path_index_city(paths, collision_check_array)
            # print(best_index)

            # Ps = np.array([
            #     [obs['vehicle_info']['ego']['x'], obs['vehicle_info']['ego']['y']],
            #     [self.waypoints[1][0], self.waypoints[3][1]],
            #     [self.waypoints[2][0], self.waypoints[2][1]],
            #     [self.waypoints[3][0], self.waypoints[3][1]],
            #     [self.waypoints[4][0], self.waypoints[4][1]],
            #     [self.waypoints[5][0], self.waypoints[5][1]],
            #     [self.waypoints[6][0], self.waypoints[6][1]],
            #     [self.waypoints[7][0], self.waypoints[7][1]],
            #     [self.waypoints[8][0], self.waypoints[8][1]],
            #     [self.waypoints[9][0], self.waypoints[9][1]],
            #     [self.waypoints[10][0], self.waypoints[10][1]],
            #     [self.waypoints[11][0], self.waypoints[11][1]],
            #     [self.waypoints[12][0], self.waypoints[12][1]],
            #
            # ])
            #
            # n = len(Ps) - 1  # 贝塞尔曲线的阶数
            # path_be = []
            # for t in np.arange(0, 1.01, 0.01):
            #     p_t = bezier(Ps, len(Ps), t)
            #     path_be.append(p_t)
            # path_bei = np.array(path_be)
            #
            # plt.scatter(path_bei[:, 0], path_bei[:, 1], color="C3", s=0.02)
            # plt.plot(path_bei[:, 0], path_bei[:, 1], 'ro')


            # if paths != []:
            #     best_path = paths[best_index]
            #     self.lp._prev_best_path = best_path
            #     # print('best_index 1')
            # else:
            #     best_path = self.lp._prev_best_path
            #     self.bp._goal_state[2] = 3
            #     # print('best_index 2')

            # if self.None_later_control == True:
            #     best_path = paths[int(len(paths) / 2)]
            # else:
            #     if paths != []:
            #         best_path = paths[best_index]
            #         self.lp._prev_best_path = best_path
            #         # print('best_index 1')
            #     else:
            #         best_path = self.lp._prev_best_path
            #         self.bp._goal_state[2] = 3
            #         # print('best_index 2')
            if self.None_later_control_city == True:
                best_path = paths[int(len(paths) / 2)]
            else:

                best_path = paths[int(len(paths) / 2)]
                # plt.scatter(best_path[0], best_path[1], color="C2", s=0.1)

                # glo_path_x = []
                # glo_path_y = []
                # best_path[0] = []
                # best_path[1] = []
                # best_path[2] = []
                # for k in range(len(self.waypoints)):
                #     path_dist = pow(pow((obs['vehicle_info']['ego']['x'] - self.waypoints[k][0]), 2) + pow(
                #         (obs['vehicle_info']['ego']['y'] - self.waypoints[k][1]), 2), 0.5)
                #     if 1 < path_dist < 25:
                #         path_ang = math.atan2((self.waypoints[k][1] - obs['vehicle_info']['ego']['y']), (self.waypoints[k][0] - obs['vehicle_info']['ego']['x']))
                #         if path_ang < 0:
                #             path_ang = path_ang + 6.28
                #         # print(path_ang)
                #         # print(obs['vehicle_info']['ego']['yaw'])
                #         if abs(path_ang - obs['vehicle_info']['ego']['yaw']) < 2:
                #             # if self.waypoints[k][0] not in glo_path_x and self.waypoints[k][1] not in glo_path_y:
                #             glo_path_x.append(self.waypoints[k][0])
                #             glo_path_y.append(self.waypoints[k][1])
                #             best_path[0].append(self.waypoints[k][0]+0.000001 * random.randint(-1000,1000))
                #             best_path[1].append(self.waypoints[k][1]+0.000001 * random.randint(-1000,1000))
                #             best_path[2].append(self.waypoints[k][2])
                # # plt.scatter(glo_path_x, glo_path_y, color="C2", s=0.1)
                # # best_path[0] = []
                # # best_path[1] = []
                # # best_path[0] = glo_path_x
                # # best_path[1] = glo_path_y
                # # print(best_path[0], type(best_path[0]))
                # plt.scatter(best_path[0], best_path[1], color="C2", s=0.1)

                # print(len(glo_path_x), glo_path_x, type(glo_path_x))
                # print(len(glo_path_y), glo_path_y, type(glo_path_y))

                # if paths != []:
                #     best_path = paths[best_index]
                #     self.lp._prev_best_path = best_path
                #     # print('best_index 1')
                # else:
                #     best_path = self.lp._prev_best_path
                #     self.bp._goal_state[2] = 3
                #     # print('best_index 2')

                # all_false_index = True
                # # print(collision_check_array)
                # # print(len(collision_check_array))
                #
                # for kt in range(len(paths)):
                #     # print(kt)
                #     if  collision_check_array[kt] == True:
                #         all_false_index = False
                #
                # if all_false_index == True:
                #     small_path_x = []
                #     small_path_y = []
                #     small_path_v = []
                #     for m in range(0, 250, 1):
                #         path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                #         path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                #         small_path_x.append(path_x)
                #         small_path_y.append(path_y)
                #         small_path_v.append(5)
                #     best_path = [[0, 0], [0, 0], [0, 0]]
                #     best_path[0] = small_path_x
                #     best_path[1] = small_path_y
                #     best_path[2] = small_path_v

            # small_path_x = []
            # small_path_y = []
            # for m in range(0, 250, 1):
            #     path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
            #     path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
            #     small_path_x.append(path_x)
            #     small_path_y.append(path_y)
            # best_path = [[0, 0], [0, 0]]
            # best_path[0] = small_path_x
            # best_path[1] = small_path_y

            # best_path = []
            # small_path_x.append(obs['vehicle_info']['ego']['x'])
            # small_path_y.append(obs['vehicle_info']['ego']['y'])

            # small_path_x = []
            # small_path_y = []
            # for m in range(0, 100, 1):
            #     path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
            #     path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
            #     small_path_x.append(path_x)
            #     small_path_y.append(path_y)
            # v_max_global = np.ones(len(path_x)) * 10
            # self.waypoints = np.stack((path_x, path_y, v_max_global), axis=1)

            # best_path.append(small_path_x)
            # best_path.append(small_path_y)
            # best_path = [small_path_x, small_path_y]

            # print(best_path)
            # print(best_path[0])
            # print(best_path[1])
            # print(best_path[2])
            # print(type(best_path))
            # plt.scatter(best_path[0], best_path[1], color="C2", s=0.1)
            # plt.scatter(small_path_x, small_path_y, color="C2", s=0.1)
            # plt.show()


            # if middle_lead_car_found:
            #     desired_speed_lead = np.sqrt(
            #         lead_car_speed[0][0] ** 2 + 2 * ego_front_dist * self.hyperparameters['max_accel'])
            #     if desired_speed_lead > self.bp._goal_state[2]:
            #         desired_speed = self.bp._goal_state[2]
            #     else:
            #         desired_speed = desired_speed_lead
            # else:
            #     desired_speed = self.bp._goal_state[2]
            # # desired_speed = self.bp._goal_state[2]
            # # print('lacall 0')
            # decelerate_to_stop = self.bp._state == behavioural_planner.DECELERATE_TO_STOP

            # decelerate_to_stop = True
            # for j in range(len(collision_check_array)):
            #     if collision_check_array[j] == True:
            #         decelerate_to_stop = False
            # print('stop:', decelerate_to_stop)
            #
            # local_waypoints = self.lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
            #                                                                      current_speed, decelerate_to_stop,
            #                                                                      lead_car_state,
            #                                                                      self.bp._follow_lead_vehicle)
            # print(self.scene_mix_straight)

            red_light_dist = -1
            stop_zone = False
            max_speed = 10
            # if self.scene_mid_inter == True:
            #     if self.start_pos['y'] < 6.5 or self.start_pos['y'] > 25.5:
            #         red_light_dist = 0
            #     else:
            #         red_light_dist = 7
            if self.scene_mid_inter == True:
                if -12 < obs['vehicle_info']['ego']['x'] < 42 and -15 < obs['vehicle_info']['ego']['y'] < 45:
                    # print('stop zone')
                    stop_zone = True
                if obs['vehicle_info']['ego']['v'] > 8:
                    if -17 < obs['vehicle_info']['ego']['x'] < 47 and -15 < obs['vehicle_info']['ego']['y'] < 45:
                        # print('stop zone')
                        stop_zone = True
                if self.scenario_info['type'] == 'REPLAY':
                    max_speed = 18

            elif self.scene_straight_straight == True:
                if 990 < obs['vehicle_info']['ego']['x'] < 1048 and 982 < obs['vehicle_info']['ego']['y'] < 1025:
                    stop_zone = True
                if self.scenario_info['type'] == 'REPLAY':
                    max_speed = 18
            else:
                red_light_dist = 8
                stop_zone = True
            # print(obs['vehicle_info']['ego']['x'])
            # print('stop:', stop_zone)
            if 'red' in observation.light_info or 'yellow' in observation.light_info:
                print('light:', observation.light_info)
                if middle_follow_car_found == True and follow_car_info['middle']['v'] > 0.5 and abs(follow_car_info['middle']['pre_long']) < 8:
                    pass
                elif abs(veh_dist) < red_light_dist:
                    pass
                elif stop_zone == True:
                    # pass
                    if abs(obs['vehicle_info']['ego']['v']) > 0:
                        u0 = [-9.8, 0]
                    else:
                        u0 = [0, 0]
                    return u0

            if self.scene_mix_straight == True or self.scene_crossing == True:
                u0 = [2.0, -0.15]
                return u0
            if middle_follow_car_found == False and self.scene_intersection == True:
                # u0 = [2.0, -0.1]
                if abs(obs['vehicle_info']['ego']['v']) > 0:
                    u0 = [-9.8, 0]
                else:
                    u0 = [0, 0]
                return u0

            # if self.scene_intersection == True or self.scene_mix_straight == True or self.scene_crossing == True:
            #     u0 = [2.0, -0.15]
            #     return u0
            # if middle_follow_car_found == False and self.scene_intersection == True:
            #     if abs(obs['vehicle_info']['ego']['v']) > 0:
            #         u0 = [-9.8, 0]
            #     else:
            #         u0 = [0, 0]
            #     return u0
            # if middle_follow_car_found == False and self.scene_mix_straight == True:
            #     u0 = [2.0, -0.15]
            #     return u0
            # if middle_follow_car_found == False and self.scene_crossing == True:
            #     u0 = [2.0, -0.15]
            #     # if self.scene_t > 1.0:
            #     #     if abs(obs['vehicle_info']['ego']['v']) > 0:
            #     #         u0 = [-9.8, 0]
            #     #     else:
            #     #         u0 = [0, 0]
            #     # else:
            #     #     u0 = [0, -0.1]
            #     return u0
            # if middle_follow_car_found == False and self.scene_round == True:
            #     if abs(obs['vehicle_info']['ego']['yaw'] - self.init_yaw) >= 3:
            #         u0 = [2.0, 0]
            #     else:
            #         u0 = [2.0, -0.15]
            #     return u0
            # if middle_follow_car_found == False and self.scene_mid_inter == True:
            #     if abs(self.start_pos['x'] - np.mean(self.goal_pos['x'])) < 5 or abs(
            #             self.start_pos['y'] - np.mean(self.goal_pos['y'])) < 5:
            #         if 'red' in observation.light_info or 'yellow' in observation.light_info:
            #             if abs(obs['vehicle_info']['ego']['v']) > 0:
            #                 u0 = [-9.8, 0]
            #             else:
            #                 u0 = [0, 0]
            #             return u0
            #         else:
            #             pass
            #     else:
            #         if abs(obs['vehicle_info']['ego']['v']) > 0:
            #             u0 = [-9.8, 0]
            #         else:
            #             u0 = [0, 0]
            #         return u0
            # if veh_dist > 8 and middle_follow_car_found == False and self.scene_straight_straight == True:
            #     # print(abs(self.start_pos['x'] - np.mean(self.goal_pos['x'])))
            #     # print(abs(self.start_pos['y'] - np.mean(self.goal_pos['y'])))
            #     if abs(self.start_pos['x'] - np.mean(self.goal_pos['x'])) < 5 or abs(
            #             self.start_pos['y'] - np.mean(self.goal_pos['y'])) < 5:
            #         if 'green' in observation.light_info:
            #             if abs(obs['vehicle_info']['ego']['v']) > 0:
            #                 u0 = [-9.8, 0]
            #             else:
            #                 u0 = [0, 0]
            #             return u0
            #         else:
            #             pass
            #     else:
            #         if abs(obs['vehicle_info']['ego']['v']) > 0:
            #             u0 = [-9.8, 0]
            #         else:
            #             u0 = [0, 0]
            #         return u0

            if self.scene_t > follow_time and veh_dist > stop_dist and middle_follow_car_found == False and self.scene_round == True:
                # for m in range(len(local_waypoints)):
                #     local_waypoints[m][2] = 0
                if abs(obs['vehicle_info']['ego']['v']) > 0:
                    u0 = [-9.8, 0]
                else:
                    u0 = [0, 0]
                return u0

            #####################
            # 检查是否存在散点图
            # for artist in plt.gca().get_children():
            #     if isinstance(artist, PathCollection):
            #         artist.remove()
            #         break  # 移除一个散点图后立即停止
            # obs['vehicle_info']['ego']['x']
            # print("当前时间：", self.dt)
            # print("predict_array:",predict_array)
            # plt.scatter(best_path[0], best_path[1], color="C2", s=0.1)
            # 需要的输入物：obs_list
            self_x = observation.ego_info.x
            self_y = observation.ego_info.y
            self_yaw = observation.ego_info.yaw
            self_width = observation.ego_info.width
            obs_backcar_list = []
            # long = 0
            for i in range(len(predict_array)):
                # obs_backcar_list = []
                for j in range(0, 4):
                    vehicle_x = predict_array[i][5 * j]
                    vehicle_y = predict_array[i][5 * j + 1]
                    delta_x = vehicle_x - self_x
                    delta_y = vehicle_y - self_y
                    # 计算目标车辆相对于自车的方向角度
                    relative_angle = math.atan2(delta_y, delta_x)
                    # 计算相对角度差
                    angle_diff = relative_angle - self_yaw
                    # 将角度差调整到 -pi 到 pi 的范围内
                    while angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    while angle_diff < -math.pi:
                        angle_diff += 2 * math.pi
                    # 判断目标车辆是否在自车的前方（角度差在 -pi/2 到 pi/2 之间）
                    if -math.pi / 2 < angle_diff < math.pi / 2:
                        continue
                    else:
                        long = - math.sqrt(delta_x ** 2 + delta_y ** 2)
                        # print("long",abs(long))
                        lat = - delta_x * math.sin(self_yaw) + delta_y * math.cos(self_yaw)
                        # print("lat",lat)
                        # 选择一个距离判断是否考虑后车会追尾
                        if abs(long) > 15:
                            continue
                        else:
                            if abs(lat) < self_width:
                                # print("long", long, "lat", lat)
                                if abs(predict_array[i][5 * j + 2] - self_yaw) < 0.524 or abs(
                                        predict_array[i][5 * j + 2] - self_yaw) > 5.76:
                                    # print("backcarfound")
                                    # print("long:", long)
                                    # print("lat:", lat)
                                    obs_backcar_list.append([long, 0.5 * j])
                                    # print("obs_backcar_list:", obs_backcar_list)
                                else:
                                    continue
                                # print("obs_backcar_list:", obs_backcar_list)
            # print("obs_backcar_list:", obs_backcar_list)
            obs_s_in = []
            obs_s_out = []
            obs_t_in = []
            obs_t_out = []
            err = 6
            # print("ego_speed:", obs['vehicle_info']['ego']['v'])
            if obs_backcar_list:
                obs_backcar_s = []
                obs_backcar_t = []
                obs_index = []
                for i in range(len(obs_backcar_list)):
                    if obs_backcar_list[i][0] + err >= 0:
                        obs_backcar_s.append(obs_backcar_list[i][0])
                        obs_backcar_t.append(obs_backcar_list[i][1])
                        obs_index.append(i)
                # print("obs_backcar_s:",obs_backcar_s)
                # print("obs_backcar_t:", obs_backcar_t)
                if obs_backcar_s:
                    m = len(obs_backcar_s)
                    if m == 1:
                        # k = (obs_backcar_s[0] - obs_backcar_list[obs_index[0] - 1][0]) / 0.5
                        # print(k)
                        # t_intersection = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                        t_intersection = 1.2
                        obs_s_in.append(0)
                        obs_s_out.append(obs_backcar_s[0] + err)
                        obs_t_in.append(t_intersection)
                        obs_t_out.append(obs_backcar_t[0])
                    elif m == 2:
                        k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                        t_intersection01 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                        if t_intersection01 < 0:
                            obs_t_in.append(0.3)
                        else:
                            obs_t_in.append(t_intersection01)
                        obs_s_in.append(0)
                        obs_s_out.append(obs_backcar_s[1] + err)
                        obs_t_out.append(obs_backcar_t[1])
                    elif m == 3:
                        k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                        t_intersection02 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                        if t_intersection02 < 0.5:
                            obs_t_in.append(0.5)
                        else:
                            obs_t_in.append(t_intersection02)
                        obs_s_in.append(0)
                        obs_s_out.append(obs_backcar_s[2] + err)
                        obs_t_out.append(obs_backcar_t[2])
                    else:
                        obs_s_in.append(0)
                        obs_s_out.append(obs_backcar_s[1] + err)
                        obs_t_in.append(0.2)
                        obs_t_out.append(0.5)
            # print("prev_timestamp:", prev_timestamp)
            # print("obs_s_in:", obs_s_in)
            # print("obs_s_out:", obs_s_out)
            # print("obs_t_in", obs_t_in)
            # print("obs_t_out", obs_t_out)

            # print("当前时间：", observation.test_info['t'])
            # ego_yaw = obs['vehicle_info']['ego']['yaw']
            selected_obs_list = []
            for i in range(len(predict_array)):
                obs_list = []
                for j in range(0, 4):
                    obs_list.append([predict_array[i][5 * j], predict_array[i][5 * j + 1], 0.5 * j])
                selected_obs_list.append(obs_list)

            # print(selected_obs_list)
            # 将路径转换为s进行表示
            pathindex2s = path_index2s(best_path)
            # 对局部路径进行增密处理，并计算曲率和航向角
            x_new, y_new, curvatures, headings_degrees = cal_refer_path_info(best_path)

            # 根据增密后的曲率，再计算出原局部路径点的曲率和航向角
            curvatures_new = project_curvature_and_heading(best_path, x_new, y_new, curvatures)
            headings_degrees_new = project_curvature_and_heading(best_path, x_new, y_new, headings_degrees)

            obs_st_s_in, obs_st_s_out, obs_st_t_in, obs_st_t_out, obs_to_consider, SLTofsingle_obs_list = obs_process(
                pathindex2s, selected_obs_list,
                x_new, y_new, headings_degrees)
            # if obs_s_in:
            #     obs_st_s_in.append(obs_s_in[0])
            #     obs_st_s_out.append(obs_s_out[0])
            #     obs_st_t_in.append(obs_t_in[0])
            #     obs_st_t_out.append(obs_t_out[0])

            # print(obs_to_consider, obs_st_s_in, obs_st_s_out, obs_st_t_in, obs_st_t_out)


            # 计算自车的信息
            if obs_to_consider:
                # if obs_s_in:
                #     obs_st_s_in.append(obs_s_in[0])
                #     obs_st_s_out.append(obs_s_out[0])
                #     obs_st_t_in.append(obs_t_in[0])
                #     obs_st_t_out.append(obs_t_out[0])

                reference_speed, w_cost_ref_speed, plan_start_v, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, plan_start_s_dot2 \
                    = cal_egoinfo(observation, headings_degrees, curvatures)
                # print('plan_start_v:',plan_start_v)
                # print('plan_start_s_dot:',plan_start_s_dot)
                # 动态规划出自车的s和t
                # print("prev_timestamp", prev_timestamp)
                # start_time01 = time.time()
                dp_speed_s, dp_speed_t = dynamic_programming(prev_timestamp, obs_st_s_in, obs_st_s_out, obs_st_t_in,
                                                             obs_st_t_out, w_cost_ref_speed,
                                                             reference_speed, w_cost_accel, w_cost_obs,
                                                             plan_start_s_dot, s_list, t_list)
                # print("prev_timestamp", prev_timestamp)
                # print('dp_speed_s', dp_speed_s)
                # print('dp_speed_t', dp_speed_t)
                # 开辟出凸空间
                # end_time01 = time.time()
                # T_dynamic_programming = end_time01 - start_time01
                # print("动态规划时间为：", T_dynamic_programming, "秒")

                # start_time02 = time.time()
                s_lb, s_ub, s_dot_lb, s_dot_ub \
                    = GenerateConvexSpace(dp_speed_s, dp_speed_t, pathindex2s, obs_st_s_in, obs_st_s_out, obs_st_t_in,
                                          obs_st_t_out,
                                          curvatures_new, max_lateral_accel=0.2 * 9.8)
                # end_time02 = time.time()
                # print("s_lb:", s_lb)
                # print("s_ub:",s_ub)
                # print("s_dot_lb:", s_dot_lb)
                # print("s_dot_ub:", s_dot_ub)
                # T_GenerateConvexSpace = end_time02 - start_time02
                # print("开辟凸空间时间为：", T_GenerateConvexSpace, "秒")
                # print('凸空间：', s_lb, s_ub, s_dot_lb, s_dot_ub)
                # 二次规划出s,s_dot,s_dot2,relative_time_init

                # start_time03 = time.time()
                qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init \
                    = SpeedPlanningwithQuadraticPlanning(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t,
                                                         s_lb, s_ub, s_dot_lb, s_dot_ub)
                # end_time03 = time.time()
                # T_QP = end_time03 - start_time03
                # print("二次规划时间为：", T_QP, "秒")
                # print('qp_s_init:', qp_s_init)
                # print('qp_s_dot_init:', qp_s_dot_init)
                # print('qp_t_init:', relative_time_init)
                # print('pannet')
                local_waypoints = cal_localwaypoints(best_path, qp_s_dot_init)
            else:
                ego_speed = obs['vehicle_info']['ego']['v']
                # local_waypoints = [[best_path[0][i], best_path[1][i], ego_speed] for i in range(len(best_path[0]))]
                for i in range(len(best_path[2])):
                    if i == 0:
                        best_path[2][i] = ego_speed
                    else:
                        best_path[2][i] = best_path[2][i - 1] + 0.4
                        if best_path[2][i] > max_speed:
                            best_path[2][i] = max_speed
                local_waypoints = [[best_path[0][i], best_path[1][i], best_path[2][i]] for i in
                                   range(len(best_path[0]))]
                accelerate = (local_waypoints[1][2] - local_waypoints[0][2]) / 0.1

                # print("accelerate:", accelerate)
            # print(best_path)
            # print('qp_s_init:', qp_s_init)
            # print('qp_s_dot_init:',qp_s_dot_init)
            # print('qp_s_dot2_init:',qp_s_dot2_init)
            # print(relative_time_init)
            # print(obs_st_s_in)
            # print(obs_st_s_out)
            #########################################


            # --------------------------------------------------------------
            # print('lacall 1')

            if local_waypoints != None:
                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.  Linear interpolation computation on the waypoints
                # is also used to ensure a fine resolution between points.
                wp_distance = []  # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                        np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                                (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp = []  # interpolated values
                # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))

                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                     float(self.INTERP_DISTANCE_RES)) - 1)
                    wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = self.INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))

                # Update the other controller values and controls
                self.controller.update_waypoints(wp_interp)

        # if self.None_later_control == True:
        #     if abs(obs['vehicle_info']['ego']['v']) > 0:
        #         u0 = [-9.8, 0]
        #     else:
        #         u0 = [0, 0]
        # else:
        # for m in range(len(local_waypoints)):
        #     if local_waypoints[m][2] < 0.5:
        #         local_waypoints[m][2] = 0.5
            # if local_waypoints[m][2] > 5:
            #     local_waypoints[m][2] = 5

        # print(local_waypoints)
        # print(len(local_waypoints))
        # print(local_waypoints[0][2])
        # print(local_waypoints[0])
        # print(local_waypoints[1])
        # print(local_waypoints[2])
        # for m in range(len(local_waypoints)):
        #     print(local_waypoints[m])
        if local_waypoints != None:
            self.pre_local_waypoints = local_waypoints

        if local_waypoints != None and local_waypoints != []:
            u0 = self.generate_control(local_waypoints, obs)
            self.pre_u0 = u0
        elif self.pre_local_waypoints != None:
            u0 = self.generate_control(self.pre_local_waypoints, obs)
            self.pre_u0 = u0
        else:
            u0 = self.pre_u0

        self.frame += 1
        # print('action:', u0)
        # self.pre_control = u0
        # if self.scene_t > follow_time and middle_follow_car_found == False:
        #     if abs(obs['vehicle_info']['ego']['v']) > 0:
        #         u0 = [-9.8, 0]
        #     else:
        #         u0 = [0, 0]
        if self.scene_mid_inter:
            if veh_dist < 18 or obs['vehicle_info']['ego']['v'] < 3 or 'red' in observation.light_info:
                u0[1] = 0

        return u0

# 主要功能和act_highway函数一样，只是适用场景不同
    def act_city_fragment(self, observation):
        # 数据格式处理
        obs = self.obs_change(observation)
        self.scene_t = observation.test_info['t']
        # 障碍物预测
        nearby_vehicles_ago = extract_nearby_vehicles(observation, False)
        predict_array = obstacle_prediction(nearby_vehicles_ago, self.dt)
        # print('obs:', obs)
        # print('observation:', observation)
        # print('pre_veh:', predict_array )

        middle_follow_car_found = False
        follow_time = 0
        stop_dist = 8
        veh_dist = pow(pow((obs['vehicle_info']['ego']['x'] - self.start_pos['x']), 2) + pow((obs['vehicle_info']['ego']['y'] - self.start_pos['y']), 2), 0.5)

        # self_x = obs['vehicle_info']['ego']['x']
        # self_y = obs['vehicle_info']['ego']['y']
        self_yaw = obs['vehicle_info']['ego']['yaw']
        self_width = obs['vehicle_info']['ego']['width']
        self_length = obs['vehicle_info']['ego']['length']
        self_v = obs['vehicle_info']['ego']['v']
        # print('observation:')

        # 当前参数获取
        current_x, current_y, current_yaw, current_speed, ego_length = \
            [obs['vehicle_info']['ego'][key] for key in ['x', 'y', 'yaw', 'v', 'length']]
        current_timestamp = obs['test_setting']['t'] + \
                            obs['test_setting']['dt']


        local_waypoints = None
        # path_validity = np.zeros((self.NUM_PATHS, 1), dtype=bool)

        # reached_the_end = False

        # Update pose and timestamp
        prev_timestamp = obs['test_setting']['t']
        # print("side")

        if self.frame % self.LP_FREQUENCY_DIVISOR_CITY == 0:
            if self.None_later_control_city == True:
                small_path_x = []
                small_path_y = []
                for m in range(0, 250, 1):
                    path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                    path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                    small_path_x.append(path_x)
                    small_path_y.append(path_y)
                v_max_global = np.ones(len(small_path_x)) * 10
                self.waypoints = np.stack((small_path_x, small_path_y, v_max_global), axis=1)

            open_loop_speed = self.lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp,
                                                                            obs['vehicle_info']['ego']['v'])

            # car_state
            ego_state = [current_x, current_y, current_yaw, open_loop_speed, ego_length]
            # lead_car_state = [lead_car_pos[0][0], lead_car_pos[0][1], lead_car_pos[0][2], lead_car_speed[0][0],
            #                   lead_car_length[0][0]]
            # follow_car_state = [follow_car_pos[0][0], follow_car_pos[0][1], follow_car_pos[0][2],
            #                     follow_car_speed[0][0], follow_car_length[0][0]]

            # Set lookahead based on current speed.
            self.bp.set_lookahead(self.BP_LOOKAHEAD_BASE_CITY + self.BP_LOOKAHEAD_TIME_CITY * open_loop_speed)

            # Perform a state transition in the behavioural planner.
            self.bp.transition_state(self.waypoints, ego_state, current_speed)

            # Compute the goal state set from the behavioural planner's computed goal state.
            goal_state_set = self.lp.get_goal_state_set(self.bp._goal_index, self.bp._goal_state, self.waypoints,
                                                        ego_state)

            # print('pre path', goal_state_set)
            # Calculate planned paths in the local frame.
            paths, path_validity = self.lp.plan_paths(goal_state_set)
            # print('pathlen:', len(paths))

            # Transform those paths back to the global frame.
            paths = local_planner.transform_paths(paths, ego_state)
            # print('pathlen:', len(paths))

            # Perform collision checking.
            # collision_check_array = self.lp._collision_checker.collision_check(paths, parkedcar_box_pts,
            #                                                                    parkedcar_box_pts_2, parkedcar_box_pts_3,
            #                                                                    self_Car)

            # coll 0428
            self_car = [self_v, self_yaw, self_width, self_length]
            # print(self_car)
            collision_check_array = collision_check(paths, predict_array, self_car, self.dt)
            # print(collision_check_array)

            # print('path:', path_validity)
            for i in range(min(len(path_validity), len(collision_check_array))):
                if path_validity[i] == False or collision_check_array[i] == False:
                    collision_check_array[i] == False
            # print('coll:', collision_check_array)

            best_index = self.lp._collision_checker.select_best_path_index_city(paths, collision_check_array)
            # print(best_index)

            if self.None_later_control_city == True:
                best_path = paths[int(len(paths) / 2)]
            else:

                best_path = paths[int(len(paths) / 2)]
                # if paths != []:
                #     best_path = paths[best_index]
                #     self.lp._prev_best_path = best_path
                #     # print('best_index 1')
                # else:
                #     best_path = self.lp._prev_best_path
                #     self.bp._goal_state[2] = 3
                    # print('best_index 2')

                all_false_index = True
                # print(collision_check_array)
                # print(len(collision_check_array))

                # for kt in range(len(paths)):
                #     # print(kt)
                #     if  collision_check_array[kt] == True:
                #         all_false_index = False
                #
                # if all_false_index == True:
                #     small_path_x = []
                #     small_path_y = []
                #     small_path_v = []
                #     for m in range(0, 250, 1):
                #         path_y = m * 0.1 * math.sin(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['y']
                #         path_x = m * 0.1 * math.cos(obs['vehicle_info']['ego']['yaw']) + obs['vehicle_info']['ego']['x']
                #         small_path_x.append(path_x)
                #         small_path_y.append(path_y)
                #         small_path_v.append(5)
                #     best_path = [[0, 0], [0, 0], [0, 0]]
                #     best_path[0] = small_path_x
                #     best_path[1] = small_path_y
                #     best_path[2] = small_path_v

            if self.scene_t > follow_time and veh_dist > stop_dist:
                # for m in range(len(local_waypoints)):
                #     local_waypoints[m][2] = 0
                # print('stop')
                if abs(obs['vehicle_info']['ego']['v']) > 0:
                    u0 = [-9.8, 0]
                else:
                    u0 = [0, 0]
                # print('return')
                return u0
                # pass
            else:
                # print('going')
                #####################
                # 检查是否存在散点图
                # for artist in plt.gca().get_children():
                #     if isinstance(artist, PathCollection):
                #         artist.remove()
                #         break  # 移除一个散点图后立即停止
                # obs['vehicle_info']['ego']['x']
                # print("当前时间：", obs['test_info']['t'])
                # print("predict_array:",predict_array)
                # plt.scatter(best_path[0], best_path[1], color="C2", s=0.1)
                # 需要的输入物：obs_list
                self_x = observation.ego_info.x
                self_y = observation.ego_info.y
                self_yaw = observation.ego_info.yaw
                self_width = observation.ego_info.width
                obs_backcar_list = []
                # long = 0
                for i in range(len(predict_array)):
                    # obs_backcar_list = []
                    for j in range(0, 4):
                        vehicle_x = predict_array[i][5 * j]
                        vehicle_y = predict_array[i][5 * j + 1]
                        delta_x = vehicle_x - self_x
                        delta_y = vehicle_y - self_y
                        # 计算目标车辆相对于自车的方向角度
                        relative_angle = math.atan2(delta_y, delta_x)
                        # 计算相对角度差
                        angle_diff = relative_angle - self_yaw
                        # 将角度差调整到 -pi 到 pi 的范围内
                        while angle_diff > math.pi:
                            angle_diff -= 2 * math.pi
                        while angle_diff < -math.pi:
                            angle_diff += 2 * math.pi
                        # 判断目标车辆是否在自车的前方（角度差在 -pi/2 到 pi/2 之间）
                        if -math.pi / 2 < angle_diff < math.pi / 2:
                            continue
                        else:
                            long = - math.sqrt(delta_x ** 2 + delta_y ** 2)
                            # print("long",abs(long))
                            lat = - delta_x * math.sin(self_yaw) + delta_y * math.cos(self_yaw)
                            # print("lat",lat)
                            # 选择一个距离判断是否考虑后车会追尾
                            if abs(long) > 15:
                                continue
                            else:
                                if abs(lat) < self_width:
                                    # print("long", long, "lat", lat)
                                    if abs(predict_array[i][5 * j + 2] - self_yaw) < 0.524 or abs(
                                            predict_array[i][5 * j + 2] - self_yaw) > 5.76:
                                        # print("backcarfound")
                                        # print("long:", long)
                                        # print("lat:", lat)
                                        obs_backcar_list.append([long, 0.5 * j])
                                        # print("obs_backcar_list:", obs_backcar_list)
                                    else:
                                        continue
                                    # print("obs_backcar_list:", obs_backcar_list)
                # print("obs_backcar_list:", obs_backcar_list)
                obs_s_in = []
                obs_s_out = []
                obs_t_in = []
                obs_t_out = []
                err = 6
                # print("ego_speed:", obs['vehicle_info']['ego']['v'])
                if obs_backcar_list:
                    obs_backcar_s = []
                    obs_backcar_t = []
                    obs_index = []
                    for i in range(len(obs_backcar_list)):
                        if obs_backcar_list[i][0] + err >= 0:
                            obs_backcar_s.append(obs_backcar_list[i][0])
                            obs_backcar_t.append(obs_backcar_list[i][1])
                            obs_index.append(i)
                    # print("obs_backcar_s:",obs_backcar_s)
                    # print("obs_backcar_t:", obs_backcar_t)
                    if obs_backcar_s:
                        m = len(obs_backcar_s)
                        if m == 1:
                            # k = (obs_backcar_s[0] - obs_backcar_list[obs_index[0] - 1][0]) / 0.5
                            # print(k)
                            # t_intersection = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                            t_intersection = 1.2
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[0] + err)
                            obs_t_in.append(t_intersection)
                            obs_t_out.append(obs_backcar_t[0])
                        elif m == 2:
                            k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                            t_intersection01 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                            if t_intersection01 < 0:
                                obs_t_in.append(0.3)
                            else:
                                obs_t_in.append(t_intersection01)
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[1] + err)
                            obs_t_out.append(obs_backcar_t[1])
                        elif m == 3:
                            k = (obs_backcar_s[1] - obs_backcar_s[0]) / 0.5
                            t_intersection02 = obs_backcar_t[0] - (1 / k) * (obs_backcar_s[0] + err)
                            if t_intersection02 < 0.5:
                                obs_t_in.append(0.5)
                            else:
                                obs_t_in.append(t_intersection02)
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[2] + err)
                            obs_t_out.append(obs_backcar_t[2])
                        else:
                            obs_s_in.append(0)
                            obs_s_out.append(obs_backcar_s[1] + err)
                            obs_t_in.append(0.2)
                            obs_t_out.append(0.5)
                # print("prev_timestamp:", prev_timestamp)
                # print("obs_s_in:", obs_s_in)
                # print("obs_s_out:", obs_s_out)
                # print("obs_t_in", obs_t_in)
                # print("obs_t_out", obs_t_out)

                # print("当前时间：", observation.test_info['t'])
                # ego_yaw = obs['vehicle_info']['ego']['yaw']
                selected_obs_list = []
                for i in range(len(predict_array)):
                    obs_list = []
                    for j in range(0, 4):
                        obs_list.append([predict_array[i][5 * j], predict_array[i][5 * j + 1], 0.5 * j])
                    selected_obs_list.append(obs_list)

                # print(selected_obs_list)
                # 将路径转换为s进行表示
                pathindex2s = path_index2s(best_path)
                # 对局部路径进行增密处理，并计算曲率和航向角
                x_new, y_new, curvatures, headings_degrees = cal_refer_path_info(best_path)
                # 根据增密后的曲率，再计算出原局部路径点的曲率和航向角
                curvatures_new = project_curvature_and_heading(best_path, x_new, y_new, curvatures)
                headings_degrees_new = project_curvature_and_heading(best_path, x_new, y_new, headings_degrees)

                obs_st_s_in, obs_st_s_out, obs_st_t_in, obs_st_t_out, obs_to_consider, SLTofsingle_obs_list = obs_process(
                    pathindex2s, selected_obs_list,
                    x_new, y_new, headings_degrees)
                # if obs_s_in:
                #     obs_st_s_in.append(obs_s_in[0])
                #     obs_st_s_out.append(obs_s_out[0])
                #     obs_st_t_in.append(obs_t_in[0])
                #     obs_st_t_out.append(obs_t_out[0])

                # print(obs_to_consider, obs_st_s_in, obs_st_s_out, obs_st_t_in, obs_st_t_out)

                # 计算自车的信息
                if obs_to_consider:
                    # if obs_s_in:
                    #     obs_st_s_in.append(obs_s_in[0])
                    #     obs_st_s_out.append(obs_s_out[0])
                    #     obs_st_t_in.append(obs_t_in[0])
                    #     obs_st_t_out.append(obs_t_out[0])

                    reference_speed, w_cost_ref_speed, plan_start_v, w_cost_accel, w_cost_obs, plan_start_s_dot, s_list, t_list, plan_start_s_dot2 \
                        = cal_egoinfo(observation, headings_degrees, curvatures)
                    # print('plan_start_v:',plan_start_v)
                    # print('plan_start_s_dot:',plan_start_s_dot)
                    # 动态规划出自车的s和t
                    # print("prev_timestamp", prev_timestamp)
                    # start_time01 = time.time()
                    dp_speed_s, dp_speed_t = dynamic_programming(prev_timestamp, obs_st_s_in, obs_st_s_out, obs_st_t_in,
                                                                 obs_st_t_out, w_cost_ref_speed,
                                                                 reference_speed, w_cost_accel, w_cost_obs,
                                                                 plan_start_s_dot, s_list, t_list)
                    # print("prev_timestamp", prev_timestamp)
                    # print('dp_speed_s', dp_speed_s)
                    # print('dp_speed_t', dp_speed_t)
                    # 开辟出凸空间
                    # end_time01 = time.time()
                    # T_dynamic_programming = end_time01 - start_time01
                    # print("动态规划时间为：", T_dynamic_programming, "秒")

                    # start_time02 = time.time()
                    s_lb, s_ub, s_dot_lb, s_dot_ub \
                        = GenerateConvexSpace(dp_speed_s, dp_speed_t, pathindex2s, obs_st_s_in, obs_st_s_out, obs_st_t_in,
                                              obs_st_t_out,
                                              curvatures_new, max_lateral_accel=0.2 * 9.8)
                    # end_time02 = time.time()
                    # print("s_lb:", s_lb)
                    # print("s_ub:",s_ub)
                    # print("s_dot_lb:", s_dot_lb)
                    # print("s_dot_ub:", s_dot_ub)
                    # T_GenerateConvexSpace = end_time02 - start_time02
                    # print("开辟凸空间时间为：", T_GenerateConvexSpace, "秒")
                    # print('凸空间：', s_lb, s_ub, s_dot_lb, s_dot_ub)
                    # 二次规划出s,s_dot,s_dot2,relative_time_init

                    # start_time03 = time.time()
                    qp_s_init, qp_s_dot_init, qp_s_dot2_init, relative_time_init \
                        = SpeedPlanningwithQuadraticPlanning(plan_start_s_dot, plan_start_s_dot2, dp_speed_s, dp_speed_t,
                                                             s_lb, s_ub, s_dot_lb, s_dot_ub)
                    # end_time03 = time.time()
                    # T_QP = end_time03 - start_time03
                    # print("二次规划时间为：", T_QP, "秒")
                    # print('qp_s_init:', qp_s_init)
                    # print('qp_s_dot_init:', qp_s_dot_init)
                    # print('qp_t_init:', relative_time_init)
                    local_waypoints = cal_localwaypoints(best_path, qp_s_dot_init)
                else:
                    ego_speed = obs['vehicle_info']['ego']['v']
                    # local_waypoints = [[best_path[0][i], best_path[1][i], ego_speed] for i in range(len(best_path[0]))]
                    for i in range(len(best_path[2])):
                        if i == 0:
                            best_path[2][i] = ego_speed
                        else:
                            best_path[2][i] = best_path[2][i - 1] + 0.4
                            if best_path[2][i] > 10:
                                best_path[2][i] = 10
                    local_waypoints = [[best_path[0][i], best_path[1][i], best_path[2][i]] for i in
                                       range(len(best_path[0]))]
                    accelerate = (local_waypoints[1][2] - local_waypoints[0][2]) / 0.1

                    # print("accelerate:", accelerate)
                # print(best_path)
                # print('qp_s_init:', qp_s_init)
                # print('qp_s_dot_init:',qp_s_dot_init)
                # print('qp_s_dot2_init:',qp_s_dot2_init)
                # print(relative_time_init)
                # print(obs_st_s_in)
                # print(obs_st_s_out)
                #########################################


            # --------------------------------------------------------------
            # print('lacall 1')

            if local_waypoints != None:
                # Update the controller waypoint path with the best local path.
                # This controller is similar to that developed in Course 1 of this
                # specialization.  Linear interpolation computation on the waypoints
                # is also used to ensure a fine resolution between points.
                wp_distance = []  # distance array
                local_waypoints_np = np.array(local_waypoints)
                for i in range(1, local_waypoints_np.shape[0]):
                    wp_distance.append(
                        np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                                (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
                wp_distance.append(0)  # last distance is 0 because it is the distance
                # from the last waypoint to the last waypoint

                # Linearly interpolate between waypoints and store in a list
                wp_interp = []  # interpolated values
                # (rows = waypoints, columns = [x, y, v])
                for i in range(local_waypoints_np.shape[0] - 1):
                    # Add original waypoint to interpolated waypoints list (and append
                    # it to the hash table)
                    wp_interp.append(list(local_waypoints_np[i]))

                    # Interpolate to the next waypoint. First compute the number of
                    # points to interpolate based on the desired resolution and
                    # incrementally add interpolated points until the next waypoint
                    # is about to be reached.
                    num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                     float(self.INTERP_DISTANCE_RES)) - 1)
                    wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                    wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                    for j in range(num_pts_to_interp):
                        next_wp_vector = self.INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                        wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                # add last waypoint at the end
                wp_interp.append(list(local_waypoints_np[-1]))

                # Update the other controller values and controls
                self.controller.update_waypoints(wp_interp)

        if local_waypoints != None:
            self.pre_local_waypoints = local_waypoints

        if local_waypoints != None and local_waypoints != []:
            u0 = self.generate_control(local_waypoints, obs)
            self.pre_u0 = u0
        elif self.pre_local_waypoints != None:
            u0 = self.generate_control(self.pre_local_waypoints, obs)
            self.pre_u0 = u0
        else:
            u0 = self.pre_u0

        self.frame += 1

        return u0
#全局路径规划
    def generate_global_path(self, x0, y0, x1, y1):
        # Create an array of x and y values for the two points
        x = np.array([x0, x1])
        y = np.array([y0, y1])
        
        # Create a function that interpolates between the two points
        f = interp1d(x, y, kind='linear')
        
        # Create an array of x values for the smooth path
        x_global = np.linspace(x0, x1, num=100)
        
        # Use the interpolation function to get the corresponding y values
        y_global = f(x_global)

        # yaw
        yaw_global = np.arctan2(np.diff(y_global), np.diff(x_global))
        yaw_global = np.append(yaw_global, yaw_global[-1])
        yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)

        waypoints = np.stack((x_global, y_global, yaw_global), axis=1)
    
        # Return the x and y values of the smooth path
        return waypoints


#该函数作用是根据局部路径和传感器数据，计算控制指令（油门、刹车）
#1、更新控制器目标路径
#2、解析当前车辆状态
#3、计算控制指令，油门刹车转向等
#4、发送控制指令，调用send_control_command
#5、返回控制信号
    def generate_control(self, local_waypoints, obs):
        # theta = self.later_control(local_waypoints, obs)
        local_waypoints = np.vstack(local_waypoints)
        result_x = local_waypoints[:, 0]
        result_y = local_waypoints[:, 1]
        yaw_global = np.arctan2(np.diff(result_y), np.diff(result_x))
        yaw_global = np.append(yaw_global, yaw_global[-1])
        yaw_global = np.where(yaw_global < 0, yaw_global + 2 * math.pi, yaw_global)
        yaw_global = yaw_global.reshape(-1, 1)
        local_waypoints = np.concatenate((local_waypoints, yaw_global), axis=1)
        
        speeds_x = local_waypoints[:, 2] * np.cos(local_waypoints[:, 3])
        speeds_y = local_waypoints[:, 2] * np.sin(local_waypoints[:, 3])

        init_state = {
            'px': obs['vehicle_info']['ego']['x'],
            'py': obs['vehicle_info']['ego']['y'],
            'v': obs['vehicle_info']['ego']['v'],
            'heading': obs['vehicle_info']['ego']['yaw'],
        }

        target_state = {
            'px': result_x[1],
            'py': result_y[1],
            'vx': speeds_x[1],
            'vy': speeds_y[1],
        }

        vel_len = obs['vehicle_info']['ego']['length'] / 1.7
        dt = obs['test_setting']['dt']

        x0_vals = np.array([init_state['px'], init_state['py'],
                           init_state['v'], init_state['heading']])
        x1_vals = np.array(
            [target_state['px'], target_state['py'], target_state['vx'], target_state['vy']])
        # print('x0:', x0_vals)
        # print('x1:', x1_vals)
        u0 = np.array([0, 0])
        bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
                (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))

        # scene_dit = self.scene_t / self.dt
        # if scene_dit % 2 == 0:
        #     bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
        #             (0, 0))
        # else:
        #     bnds = ((0, 0),
        #             (-self.hyperparameters['max_steering'], self.hyperparameters['max_steering']))

        # bnds = ((-self.hyperparameters['max_accel'], self.hyperparameters['max_accel']),
        #         (self.pre_u0[1] - (self.ROT_RATE_LIMIT * self.dt), self.pre_u0[1] + (self.ROT_RATE_LIMIT * self.dt)))

        # Minimize difference between simulated state and next state by varying input u
        u0 = minimize(self.position_orientation_objective, u0, args=(x0_vals, x1_vals, vel_len, dt),
                      options={'disp': False, 'maxiter': 100, 'ftol': 1e-9},  # 100 1e-9
                      method='SLSQP', bounds=bnds).x

        # scene_dit = self.scene_t / self.dt
        # if scene_dit % 2 == 0:
        #     u0[0] = 0
        # u0[1] = theta

        # u0[1] = round(u0[1], 4)
        # if u0[1] < 0.0001:
        #     u0[1] = 0
        # u0[1] = format(u0[1], '.3f')
        # print('init_theta:', u0[1])
        # Get simulated state using the found inputs
        # x1_sim_array = self.vehicle_dynamic(init_state, u0, vel_len, dt)
        # x1_sim = np.array([x1_sim_array['px'], x1_sim_array['py'],
        #                    x1_sim_array['v']*np.cos(x1_sim_array['heading']),
        #                    x1_sim_array['v']*np.sin(x1_sim_array['heading'])])
        #
        # x_delta = np.linalg.norm(x1_sim-x1_vals)
        # print('error:', x_delta)
        # x1_sim = vehicle_dynamics.forward_simulation(x0_vals, u0, dt, throw=False)
        # u0[1] = theta
        # if 'follow' in self.scenario_info['name'] or \
        #         'lanechanging' in self.scenario_info['name'] or \
        #         'highway' in self.scenario_info['name'] or \
        #         'cutin' in self.scenario_info['name']:
        #     u1 = self.action_dynamic_constraint_highway(obs, u0)
        # else:
        #     u1 = self.action_dynamic_constraint_city(obs, u0)
        # print('u0:', u0)
        # u1 = self.action_dynamic_constraint_highway(obs, u0)
        # u1 = self.action_dynamic_constraint_city(obs, u0)

        # u1 = [u0[0], 0]

        # print('u1:', u1)
        if self.None_later_control == True:
            # print('planner')
            if obs['vehicle_info']['ego']['v'] >= self.GLOBAL_MAX_SPEED:
                if u0[0] > 0:
                    u0[0] = 0

            # if 'follow' in self.scenario_info['name'] or \
            #         'lanechanging' in self.scenario_info['name'] or \
            #         'highway' in self.scenario_info['name'] or \
            #         'cutin' in self.scenario_info['name']:
            #     # print('highway')
            #     pass
            # else:
            #     # print('slow down')
            #     if abs(obs['vehicle_info']['ego']['v']) > 0:
            #         u0[0] = -9.8
            #     else:
            #         u0[0] = 0

            u1 = [u0[0], 0]
            return u1
        else:
            if 'follow' in self.scenario_info['name'] or \
                    'lanechanging' in self.scenario_info['name'] or \
                    'highway' in self.scenario_info['name'] or \
                    'cutin' in self.scenario_info['name']:
                # print('highway')
                if self.None_later_control_highway == True:
                    if obs['vehicle_info']['ego']['v'] >= self.GLOBAL_MAX_SPEED:
                        if u0[0] > 0:
                            u0[0] = 0
                    u1 = [u0[0], 0]
                    return u1
                else:
                    if 'highway' in self.scenario_info['name'] and self.scenario_info['type'] == 'REPLAY':
                        # print('hiway_merge_replay')
                        pid_controller = PIDAngleController(
                            K_P=0.83,  # 设置合适的比例系数
                            K_D=0,  # 设置合适的微分系数
                            K_I=0.02,  # 设置合适的积分系数
                            dt=self.dt,  # 设置合适的时间间隔
                            use_real_time=False  # 或者根据实际需要设置为True
                        )
                        # print(self.frame)
                        current_v = obs['vehicle_info']['ego']['v']
                        weight_v = 1.0 * current_v * current_v
                        # weight_v = 5.0 * current_v * current_v
                        current_a = obs['vehicle_info']['ego']['a']
                        current_angle = obs['vehicle_info']['ego']['rot']
                        target_a = u0[0]
                        # print('target_a', target_a)
                        weight_a = 20 * current_a * current_a
                        weight_rot = 40000 * current_angle * current_angle
                        weight_all = weight_v + weight_a + weight_rot + 680
                        # max_pid_output_a = 12 - weight_all/400
                        max_pid_output_a = 15 * 520 / weight_all
                        # if current_v < 30:
                        #     max_pid_output_a = 2 * max_pid_output_a
                        # print(max_pid_output_a)
                        if current_v < 20:
                            max_pid_output_a = max_pid_output_a + 0.5 * (20 - current_v)
                        # if current_a * target_a < 0:
                        max_pid_output_a = max(max_pid_output_a, 1)
                        # max_a = max_pid_output_a
                        # print('max_pid_output_a', max_pid_output_a)
                        # print('target_deta_a',target_a - current_a)
                        if self.path_all_false == False and self.middle_follow_distance > 15:
                            target_a = current_a - 10
                        min_pid_a = 8
                        if current_a * target_a < 0:
                            max_pid_output_a = 1.5 * max_pid_output_a
                            min_pid_a = 1.5 * min_pid_a
                        pid_output_a = pid_controller.get_angle(target_a, current_a, max_pid_output_a, min_pid_a)
                        u0[0] = pid_output_a + current_a
                        # print(self.path_all_false, )
                        # print(self.middle_follow_distance)
                        # u0[0] = min(u0[0], 5.6)
                        # if u0[0] < 0:
                        #     u0[0] = max(u0[0], -5.8)
                        # print('actual_a', u0[0])
                        # angle_max,error_max = dynamics_check(obs, current_a, pid_output_a)
                        ##转角##
                        # print('target_a', u0[0])
                        pid_controller = PIDAngleController(
                            K_P=0.9,  # 设置合适的比例系数
                            K_D=0,  # 设置合适的微分系数
                            K_I=0.02,  # 设置合适的积分系数
                            dt=self.dt,  # 设置合适的时间间隔
                            use_real_time=False  # 或者根据实际需要设置为True
                        )
                        max_pid_output_rot = 0.003
                        target_angle = u0[1]  # Compute your target angle here

                        pid_rot = 0.04
                        if current_angle * target_angle < 0:
                            pid_rot = 1.5 * pid_rot
                        pid_output_rot = pid_controller.get_angle(target_angle, current_angle, pid_rot, pid_rot)
                        u0[1] = pid_output_rot + current_angle
                        u0[1] = min(u0[1], max_pid_output_rot)
                        if u0[1] < 0:
                            u0[1] = max(u0[1], -0.004)
                        if current_a * target_a < 0 and abs(current_a) < 1 or current_a < -5.3:
                            u0[0] = 0.09 * abs(current_a) * current_a
                        if u0[0] > 4.5:
                            u0[0] = 0.85 * current_a

                        if current_v > 37.5 and u0[0] > 0:
                            u0[0] = 0.25 * (41.5 - current_v) * u0[0]
                        if current_v > 41.5:
                            u0[1] = min(u0[1], 0)
                        ##加速度##
                        if current_a * target_a < 0 and abs(current_a) < 0.4:
                            u0[0] = 0
                        if self.scene_t < 0.5:
                            u0[1] = 0

                        return u0

                    else:
                        # print('hiway')
                        ############0510更改############
                        pid_controller = PIDAngleController(
                            K_P=0.93,  # 设置合适的比例系数
                            K_D=0,  # 设置合适的微分系数
                            K_I=0.02,  # 设置合适的积分系数
                            dt=self.dt,  # 设置合适的时间间隔
                            use_real_time=False  # 或者根据实际需要设置为True
                        )
                        current_v = obs['vehicle_info']['ego']['v']
                        weight_v = 1.0 * current_v * current_v
                        current_a = obs['vehicle_info']['ego']['a']
                        current_angle = obs['vehicle_info']['ego']['rot']
                        target_a = u0[0]
                        weight_a = 20 * current_a * current_a
                        weight_rot = 40000 * current_angle * current_angle
                        weight_all = weight_v + weight_a + weight_rot + 680
                        # max_pid_output_a = 12 - weight_all/400
                        max_pid_output_a = 12 * 520 / weight_all
                        if current_v < 20:
                            max_pid_output_a = max_pid_output_a + 0.5 * (20 - current_v)
                        # if current_a * target_a < 0:
                        max_pid_output_a = max(max_pid_output_a, 1)
                        # max_a = max_pid_output_a
                        # print('max_pid_output_a', max_pid_output_a)
                        min_pid_a = 5
                        pid_output_a = pid_controller.get_angle(target_a, current_a, max_pid_output_a, min_pid_a)
                        u0[0] = pid_output_a + current_a
                        u0[0] = min(u0[0], 9.6, max_pid_output_a)
                        # angle_max,error_max = dynamics_check(obs, current_a, pid_output_a)
                        ##转角##
                        # print('target_a', u0[0])
                        pid_controller = PIDAngleController(
                            K_P=0.98,  # 设置合适的比例系数
                            K_D=0,  # 设置合适的微分系数
                            K_I=0.02,  # 设置合适的积分系数
                            dt=self.dt,  # 设置合适的时间间隔
                            use_real_time=False  # 或者根据实际需要设置为True
                        )
                        max_pid_output_rot = 0.035 * max_pid_output_a
                        target_angle = u0[1]  # Compute your target angle here
                        pid_rot = 0.08
                        pid_output_rot = pid_controller.get_angle(target_angle, current_angle, pid_rot, pid_rot)
                        u0[1] = pid_output_rot + current_angle
                        u0[1] = min(u0[1], max_pid_output_rot)

                        if current_a * target_a < 0 and abs(current_a) < 1 or abs(current_a) > 5.2:
                            u0[0] = 0.09 * abs(current_a) * current_a
                        ##加速度##
                        if current_a * target_a < 0 and abs(current_a) < 0.4:
                            u0[0] = 0
                        return u0

            else:
                if self.None_later_control_city == True and self.scene_merge == False:
                    if obs['vehicle_info']['ego']['v'] >= self.GLOBAL_MAX_SPEED:
                        if u0[0] > 0:
                            u0[0] = 0
                    u1 = [u0[0], 0]

                    return u1
                else:
                    # print('u0:', u0)
                    # u0 = self.action_dynamic_constraint_highway(obs, u0)
                    # print('u1:', u1)
                    ############0510更改############
                    pid_controller = PIDAngleController(
                        K_P=0.88,  # 设置合适的比例系数
                        K_D=0,  # 设置合适的微分系数
                        K_I=0.02,  # 设置合适的积分系数
                        dt=self.dt,  # 设置合适的时间间隔
                        use_real_time=False  # 或者根据实际需要设置为True
                    )
                    current_v = obs['vehicle_info']['ego']['v']
                    weight_v = 5.0 * current_v * current_v
                    current_a = obs['vehicle_info']['ego']['a']
                    current_angle = obs['vehicle_info']['ego']['rot']
                    target_a = u0[0]
                    weight_a = 20 * current_a * current_a
                    weight_rot = 40000 * current_angle * current_angle
                    weight_all = weight_a + weight_rot + 680
                    # max_pid_output_a = 12 - weight_all/400
                    max_pid_output_a = 18 * 520 / weight_all
                    min_pid_a = max_pid_output_a
                    # print('max_pid_output_a', max_pid_output_a)
                    pid_output_a = pid_controller.get_angle(target_a, current_a, max_pid_output_a, min_pid_a)
                    u0[0] = pid_output_a + current_a
                    # if current_v < 3.5 and current_a < -0.7:
                    #     u0[0] = current_a - 0.18 * current_a
                    pid_controller = PIDAngleController(
                        K_P=0.88,  # 设置合适的比例系数
                        K_D=0,  # 设置合适的微分系数
                        K_I=0.02,  # 设置合适的积分系数
                        dt=self.dt,  # 设置合适的时间间隔
                        use_real_time=False  # 或者根据实际需要设置为True
                    )
                    # max_pid_output_rot = 0.04 * max_pid_output_a
                    deta_a = abs(u0[0] - current_a)
                    target_angle = u0[1]
                    pid_rot = 0.009
                    weight_deta_rot = abs(current_angle * 100) + deta_a * 75 + current_a * 3 * current_a + 10
                    pid_rot = pid_rot + 0.15 * 18 / weight_deta_rot
                    # pid_rot = min(pid_rot, 0.18)
                    # print('pid_rot', pid_rot)
                    pid_output_rot = pid_controller.get_angle(target_angle, current_angle, pid_rot, pid_rot)
                    u0[1] = pid_output_rot + current_angle
                    if u0[1] > 0:
                        u0[1] = min(u0[1], 0.35)
                    else:
                        u0[1] = max(u0[1], -0.35)
                    # u0[1] = max(u0[1], -0.2)
                    # u0[0] = 0.01
                    # if current_a * target_a < 0 and abs(current_a) < 0.5:
                    #     u0[0] = 0.1 * abs(current_a) * current_a
                    if current_a * target_a < 0 and abs(current_a) < 1 or abs(current_a) > 5.55:
                        u0[0] = 0.09 * abs(current_a) * current_a
                    ##加速度##
                    if current_a * target_a < 0 and abs(current_a) < 0.4:
                        u0[0] = 0

                    if self.scene_merge == True and self.None_later_control_city == True and self.scenario_info['type'] == 'REPLAY':
                        if self.scene_t > 1.9:
                            u0[0] = target_a
                        # if self.scene_t * 10 % 2 == 0:
                        #     u0[1] = -0.005 * self.scene_t
                        # else:
                        #     u0[1] = current_angle
                        u0[1] = -0.01 * self.scene_t
                        u0[1] = max(-0.015, u0[1])
                        num = 2
                        if abs(u0[1]) < 0.015:
                            u0[0] = 0
                        else:
                            if current_v > 0.5:
                                if u0[0] < 0 and abs(u0[0]) > 5:
                                    u0[0] = -4
                            elif current_v < 0.5:
                                if u0[0] > num:
                                    u0[0] = num
                                if u0[0] < -num:
                                    u0[0] = -num
                            # if u0[0] > num:
                            #     u0[0] = num
                            # if u0[0] < -num:
                            #     u0[0] = -num
                        if current_v < 0.5 and u0[0] < 0:
                            u0[0] = 0
                    # print(u0)
                    return u0


            # return u0


        # return u0
        # return u1



#主要用于 限制自动驾驶车辆在高速公路上的动态控制参数，确保 加速度、转向角、侧偏角、横摆角速度等控制变量满足安全驾驶要求。
#输入是车辆当前状态数据，包括速度、位置等。以及计算得到的初步控制信号
#输出是经过动态约束修正后的控制信号
    def action_dynamic_constraint_highway(self, obs, u0):
        # 约束范围（百分比）
        dyn_rate = 0.85
        dyn_bill = 1.0

        vel_len = obs['vehicle_info']['ego']['length'] / 1.7
        wb_dis = vel_len * 0.5
        lr_rate = 0.5
        veh_mh = 1.5


        # # u0[1] = 0
        # if abs(u0[1]) < 0.01:
        #     # if u0[1] * self.pre_u0[1] < 1:
        #     if u0[1] > 0 and self.pre_u0[1] < 0:
        #         u0[1] = 0
        #     if u0[1] < 0 and self.pre_u0[1] > 0:
        #         u0[1] = 0

        if obs['vehicle_info']['ego']['v'] >= self.GLOBAL_MAX_SPEED:
            if u0[0] > 0:
                u0[0] = 0

        # # print(obs['vehicle_info']['ego']['yaw'], type(obs['vehicle_info']['ego']['yaw']))
        # if obs['vehicle_info']['ego']['yaw'] >= 6.18:
        #     if u0[1] > 0:
        #         u0[1] = 0
        # if obs['vehicle_info']['ego']['yaw'] <= 0.1:
        #     if u0[1] < 0:
        #         u0[1] = 0

        # 前轮转速约束
        max_steer_dis = self.ROT_RATE_LIMIT * self.dt
        if u0[1] - self.pre_u0[1] > max_steer_dis:
            # print('steer dis over max_steer_dis')
            u0[1] = self.pre_u0[1] + max_steer_dis
        if u0[1] - self.pre_u0[1] < -max_steer_dis:
            # print('steer dis below -max_steer_dis')
            u0[1] = self.pre_u0[1] - max_steer_dis

        # 前轮转角约束
        if u0[1] > self.ROT_LIMIT:
            u0[1] = self.ROT_LIMIT
        if u0[1] < -self.ROT_LIMIT:
            u0[1] = -self.ROT_LIMIT

        # 纵向加速度约束
        if u0[0] > self.ACC_LIMIT:
            u0[0] = self.ACC_LIMIT
        if u0[0] < -self.ACC_LIMIT:
            u0[0] = -self.ACC_LIMIT

        # 纵向加加速度约束
        max_acc_dis = self.JERK_LIMIT * self.dt
        if u0[0] - self.pre_u0[0] > max_acc_dis:
            # print('acc dis over max_acc_dis')
            u0[0] = self.pre_u0[0] + max_acc_dis
            # print('af_u0:', u0)
        if u0[0] - self.pre_u0[0] < -max_acc_dis:
            # print('acc dis below -max_acc_dis')
            u0[0] = self.pre_u0[0] - max_acc_dis
            # print('af_u0:', u0)

        l_f = 1.04  # 0.5 * vel_len
        l_r = 1.56  # 0.5 * vel_len
        k_f = 43160 * 2
        k_r = 29210 * 2
        veh_l = 2.6
        # veh_r = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
        # side_slip_angle = (l_f * k_f * u0[1] * obs['vehicle_info']['ego']['v'] - veh_r * (
        #             l_f * l_f * k_f + l_r * l_r * k_r)) / (l_f * k_f - l_r * k_r)

        # if u0[1] == 0:
        #     veh_r = 0
        #     veh_ya = 0
        # else:
        #     veh_r = vel_len / math.tan(u0[1])
        #     veh_ya = u0[0] * u0[0] / veh_r

        # veh_ya = u0[0] * u0[0] / veh_r
        # sideslip_angle = veh_mh*1*veh_ya / vel_len
        # sideslip_angle = math.atan(veh_mh * 1 * veh_ya / vel_len)
        # yaw_rate = obs['vehicle_info']['ego']['v'] / veh_r

        # dyn_rate = 1 - 0.1 * u0[0]
        # print(dyn_rate)
        # if abs(u0[0]) > 0:
        #     u0[1] = 0
        # if abs(u0[0]) > 0 and abs(u0[1]) > 0:
        #     u0[1] = 0

        # scene_dit = self.scene_t / self.dt
        # if scene_dit % 2 == 0:
        #     # print('steer = 0')
        #     if abs(u0[0]) > 0.0:
        #         u0[1] = 0
        #
        #     if u0[1] - self.pre_u0[1] > max_steer_dis:
        #         u0[0] = 0
        #     if u0[1] - self.pre_u0[1] < -max_steer_dis:
        #         u0[0] = 0
        #     if u0[0] - self.pre_u0[0] > max_acc_dis:
        #         u0[1] = 0
        #     if u0[0] - self.pre_u0[0] < -max_acc_dis:
        #         u0[1] = 0
        # else:
        #     # print('acc = 0')
        #     if abs(u0[1]) > 0.0:
        #         u0[0] = 0
        #
        #     if u0[1] - self.pre_u0[1] > max_steer_dis:
        #         u0[0] = 0
        #     if u0[1] - self.pre_u0[1] < -max_steer_dis:
        #         u0[0] = 0
        #     if u0[0] - self.pre_u0[0] > max_acc_dis:
        #         u0[1] = 0
        #     if u0[0] - self.pre_u0[0] < -max_acc_dis:
        #         u0[1] = 0

        # 质心侧偏角约束
        # sideslip_stand = math.atan(0.02 * 0.85 * 9.8)
        sideslip_stand = abs(math.atan(0.02 * 0.85 * 9.8 / dyn_bill))
        # sideslip_angle = abs(math.atan(lr_rate * math.tan(u0[1])))
        # sideslip_angle = veh_mh * 1 * veh_ya / vel_len
        # sideslip_angle = math.atan(veh_mh * 1 * veh_ya / vel_len)
        veh_r = obs['vehicle_info']['ego']['v'] * u0[1] / veh_l
        sideslip_angle = (l_f * k_f * u0[1] * obs['vehicle_info']['ego']['v'] - veh_r * (
                l_f * l_f * k_f + l_r * l_r * k_r)) / (l_f * k_f - l_r * k_r)
        # print('sideslip_stand:', sideslip_stand)
        # print('sideslip_angle:', sideslip_angle)
        if abs(sideslip_angle) > sideslip_stand * dyn_rate:
            if u0[1] > 0:  # if u0[1] - self.pre_u0[1] > 0:
                ite = 0
                side_state = 0
                # print('state1')
                while 0 < u0[1] < self.ROT_LIMIT:
                    u0[1] = u0[1] - 0.00001
                    # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                    # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                    # if u0[1] == 0:
                    #     veh_r = 0
                    #     veh_ya = 0
                    # else:
                    #     veh_r = vel_len / math.tan(u0[1])
                    #     veh_ya = u0[0] * u0[0] / veh_r
                    # sideslip_angle_test = math.atan(veh_mh * 1 * veh_ya / vel_len)
                    veh_r = obs['vehicle_info']['ego']['v'] * u0[1] / veh_l
                    sideslip_angle_test = (l_f * k_f * u0[1] * obs['vehicle_info']['ego']['v'] - veh_r * (
                            l_f * l_f * k_f + l_r * l_r * k_r)) / (l_f * k_f - l_r * k_r)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                        # print('af_u0:', u0)
                        print('sideslip_angle_test:', sideslip_angle_test)
                        side_state = 1
                        break
                    # ite = ite + 1

                # if side_state == 0:
                #     ite = 0
                #     # side_state = 0
                #     while -self.ROT_LIMIT < u0[1] < self.ROT_LIMIT:
                #         u0[1] = u0[1] + 0.00001
                #         # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                #         # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                #         if u0[1] == 0:
                #             veh_r = 0
                #             veh_ya = 0
                #         else:
                #             veh_r = vel_len / math.tan(u0[1])
                #             veh_ya = u0[0] * u0[0] / veh_r
                #         sideslip_angle_test = math.atan(veh_mh * 1 * veh_ya / vel_len)
                #         if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                #             side_state = 1
                #             print('sideslip_angle_test:', sideslip_angle_test)
                #             break
                #         ite = ite + 1

            elif u0[1] < 0:  # elif u0[1] - self.pre_u0[1] < 0:
                # print('state2')
                ite = 0
                side_state = 0
                while -self.ROT_LIMIT < u0[1] < 0:
                    u0[1] = u0[1] + 0.00001
                    # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                    # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                    # if u0[1] == 0:
                    #     veh_r = 0
                    #     veh_ya = 0
                    # else:
                    #     veh_r = vel_len / math.tan(u0[1])
                    #     veh_ya = u0[0] * u0[0] / veh_r
                    # sideslip_angle_test = math.atan(veh_mh * 1 * veh_ya / vel_len)
                    veh_r = obs['vehicle_info']['ego']['v'] * u0[1] / veh_l
                    sideslip_angle_test = (l_f * k_f * u0[1] * obs['vehicle_info']['ego']['v'] - veh_r * (
                            l_f * l_f * k_f + l_r * l_r * k_r)) / (l_f * k_f - l_r * k_r)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:

                        side_state = 1
                        print('sideslip_angle_test:', sideslip_angle_test)
                        break
                    # ite = ite + 1

                # if side_state == 0:
                #     ite = 0
                #     # side_state = 0
                #     while -self.ROT_LIMIT < u0[1] < self.ROT_LIMIT:
                #         u0[1] = u0[1] - 0.00001
                #         # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                #         # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                #         if u0[1] == 0:
                #             veh_r = 0
                #             veh_ya = 0
                #         else:
                #             veh_r = vel_len / math.tan(u0[1])
                #             veh_ya = u0[0] * u0[0] / veh_r
                #         sideslip_angle_test = math.atan(veh_mh * 1 * veh_ya / vel_len)
                #         if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                #             print('sideslip_angle_test:', sideslip_angle_test)
                #             side_state = 1
                #             break
                #         ite = ite + 1

        # 横摆角速度约束
        if obs['vehicle_info']['ego']['v'] == 0:
            yaw_rate_stand = 9.8 * 0.85 / 1
        else:
            # yaw_rate_stand = abs(9.8 * 0.85 / obs['vehicle_info']['ego']['v'])  # + 0.5 * u0[0] * self.dt)
            yaw_rate_stand = abs(9.8 * 0.85 / (obs['vehicle_info']['ego']['v'] * dyn_bill))

        vx_now = obs['vehicle_info']['ego']['v']
        vx_next = obs['vehicle_info']['ego']['v'] + u0[0] * self.dt
        if vx_next > vx_now:
            veh_vx = vx_next
        else:
            veh_vx = vx_now

        # sideslip_angle = math.atan(lr_rate * math.tan(u0[1]))
        # sideslip_angle = veh_mh * 1 * veh_ya / vel_len
        # sideslip_angle = math.atan(veh_mh * 1 * veh_ya / vel_len)
        # yaw_rate = veh_vx * math.tan(sideslip_angle) / wb_dis
        # yaw_rate = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
        if veh_r == 0:
            yaw_rate = 0
        else:
            yaw_rate = obs['vehicle_info']['ego']['v'] / veh_r
        if abs(yaw_rate) > yaw_rate_stand * dyn_rate:
            # print('yaw_rate over stand:', yaw_rate)
            if u0[1] > 0:  # if u0[1] - self.pre_u0[1] > 0:
                yaw_state = 0
                while 0 < u0[1] < self.ROT_LIMIT:
                    u0[1] = u0[1] - 0.00001
                    # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                    # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                    # yaw_rate_test = veh_vx * math.tan(sideslip_angle_test) / wb_dis
                    # yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                    # yaw_rate_test = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
                    if u0[1] == 0:
                        veh_r = 0
                        veh_ya = 0
                    else:
                        veh_r = vel_len / math.tan(u0[1])
                        veh_ya = u0[0] * u0[0] / veh_r
                    if veh_r == 0:
                        yaw_rate_test = 0
                    else:
                        yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        # print('yaw_rate current')
                        # print('current yaw_rate:', yaw_rate_test)
                        yaw_state = 1
                        break
                # if yaw_state == 0:
                #     while -self.ROT_LIMIT < u0[1] < self.ROT_LIMIT:
                #         u0[1] = u0[1] + 0.00001
                #         # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                #         # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                #         # yaw_rate_test = veh_vx * math.tan(sideslip_angle_test) / wb_dis
                #         # yaw_rate_test = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
                #         # yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                #         if u0[1] == 0:
                #             veh_r = 0
                #             veh_ya = 0
                #         else:
                #             veh_r = vel_len / math.tan(u0[1])
                #             veh_ya = u0[0] * u0[0] / veh_r
                #         if veh_r == 0:
                #             yaw_rate_test = 0
                #         else:
                #             yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                #         if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                #             # print('yaw_rate current')
                #             # print('current yaw_rate:', yaw_rate_test)
                #             yaw_state = 1
                #             break

            elif u0[1] < 0:  # elif u0[1] - self.pre_u0[1] < 0:
                yaw_state = 0
                while -self.ROT_LIMIT < u0[1] < 0:
                    u0[1] = u0[1] + 0.00001
                    # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                    # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                    # yaw_rate_test = veh_vx * math.tan(sideslip_angle_test) / wb_dis
                    # yaw_rate_test = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
                    # yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                    if u0[1] == 0:
                        veh_r = 0
                        veh_ya = 0
                    else:
                        veh_r = vel_len / math.tan(u0[1])
                        veh_ya = u0[0] * u0[0] / veh_r
                    if veh_r == 0:
                        yaw_rate_test = 0
                    else:
                        yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        # print('yaw_rate current')
                        # print('current yaw_rate:', yaw_rate_test)
                        yaw_state = 1
                        break
                # if yaw_state == 0:
                #     while -self.ROT_LIMIT < u0[1] < self.ROT_LIMIT:
                #         u0[1] = u0[1] - 0.00001
                #         # sideslip_angle_test = math.atan(lr_rate * math.tan(u0[1]))
                #         # sideslip_angle_test = veh_mh * 1 * veh_ya / vel_len
                #         # yaw_rate_test = veh_vx * math.tan(sideslip_angle_test) / wb_dis
                #         # yaw_rate_test = obs['vehicle_info']['ego']['v'] * u0[1] / vel_len
                #         # yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                #         if u0[1] == 0:
                #             veh_r = 0
                #             veh_ya = 0
                #         else:
                #             veh_r = vel_len / math.tan(u0[1])
                #             veh_ya = u0[0] * u0[0] / veh_r
                #         if veh_r == 0:
                #             yaw_rate_test = 0
                #         else:
                #             yaw_rate_test = obs['vehicle_info']['ego']['v'] / veh_r
                #         if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                #             # print('yaw_rate current')
                #             # print('current yaw_rate:', yaw_rate_test)
                #             yaw_state = 1
                #             break

        # acc_pid = PositionPID(u0[0], self.pre_u0[0], self.dt, self.pre_u0[0] + max_acc_dis, self.pre_u0[0] - max_acc_dis,
        #                     0.2, 0.1, 0.01)
        # u0[0] = acc_pid.calculate()
        #
        # steer_pid = PositionPID(u0[1], self.pre_u0[1], self.dt, self.pre_u0[1] + self.ROT_RATE_LIMIT * self.dt,
        #                         self.pre_u0[1] - self.ROT_RATE_LIMIT * self.dt,
        #                         0.65, 0.02, 0)
        # steer_pid = PositionPID(u0[1], self.pre_u0[1], self.dt, self.ROT_LIMIT, -self.ROT_LIMIT,
        #                         0.65, 0.02, 0)
        # u0[1] = steer_pid.calculate()

        # # 前轮转角约束
        # if u0[1] > self.ROT_LIMIT:
        #     u0[1] = self.ROT_LIMIT
        # if u0[1] < -self.ROT_LIMIT:
        #     u0[1] = -self.ROT_LIMIT

        # if u0[0] != 0.0 and u0[1] != 0.0:
        #     if u0[1] - self.pre_u0[1] > max_steer_dis:
        #         u0[0] = 0
        #     if u0[1] - self.pre_u0[1] < -max_steer_dis:
        #         u0[0] = 0
        #
        # if u0[0] != 0.0 and u0[1] != 0.0:
        #     if u0[0] - self.pre_u0[0] > max_acc_dis:
        #         u0[1] = 0
        #     if u0[0] - self.pre_u0[0] < -max_acc_dis:
        #         u0[1] = 0
        #
        # if u0[0] != 0.0 and u0[1] != 0.0:
        #     u0[0] = 0

        # print(u0)
        return u0


#功能同action_dynamic_constraint_highway函数一样，适用场景不同
    def action_dynamic_constraint_city(self, obs, u0):
        # 约束范围（百分比）
        dyn_rate = 0.9
        dyn_bill = 1.3

        vel_len = obs['vehicle_info']['ego']['length'] / 1.7
        wb_dis = vel_len * 0.5
        u0[1] = 0
        if abs(u0[1]) < 1e-5:
            u0[1] = 0
        if abs(u0[0]) > 5.0:
            u0[1] = 0

        # 前轮转角约束
        if u0[1] > self.ROT_LIMIT:
            u0[1] = self.ROT_LIMIT
        if u0[1] < -self.ROT_LIMIT:
            u0[1] = -self.ROT_LIMIT

        # 前轮转速约束
        max_steer_dis = self.ROT_RATE_LIMIT * self.dt
        if u0[1] - self.pre_u0[1] > max_steer_dis:
            # print('steer dis over max_steer_dis')
            u0[1] = self.pre_u0[1] + max_steer_dis
        if u0[1] - self.pre_u0[1] < -max_steer_dis:
            # print('steer dis below -max_steer_dis')
            u0[1] = self.pre_u0[1] - max_steer_dis

        # 纵向加速度约束
        if u0[0] > self.ACC_LIMIT:
            u0[0] = self.ACC_LIMIT
        if u0[0] < -self.ACC_LIMIT:
            u0[0] = -self.ACC_LIMIT

        # 纵向加加速度约束
        max_acc_dis = self.JERK_LIMIT * self.dt
        if u0[0] - self.pre_u0[0] > max_acc_dis:
            # print('acc dis over max_acc_dis')
            u0[0] = self.pre_u0[0] + max_acc_dis
            # print('af_u0:', u0)
        if u0[0] - self.pre_u0[0] < -max_acc_dis:
            # print('acc dis below -max_acc_dis')
            u0[0] = self.pre_u0[0] - max_acc_dis
            # print('af_u0:', u0)

        # 质心侧偏角约束
        # sideslip_stand = math.atan(0.02 * 0.85 * 9.8)
        sideslip_stand = math.atan(0.02 * 0.85 * 9.8 / dyn_bill)
        sideslip_angle = abs(math.atan((wb_dis * math.tan(u0[1])) / vel_len))
        # print('sideslip_stand:', sideslip_stand)
        if abs(sideslip_angle) > sideslip_stand * dyn_rate:
            if u0[1] - self.pre_u0[1] > 0:
                while True:
                    u0[1] = u0[1] - 0.00001
                    sideslip_angle_test = math.atan((wb_dis * math.tan(u0[1])) / vel_len)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                        side_state = 1
                        break

            elif u0[1] - self.pre_u0[1] < 0:
                while True:
                    u0[1] = u0[1] + 0.00001
                    sideslip_angle_test = math.atan((wb_dis * math.tan(u0[1])) / vel_len)
                    if abs(sideslip_angle_test) < sideslip_stand * dyn_rate:
                        break

        # 横摆角速度约束
        if obs['vehicle_info']['ego']['v'] == 0:
            yaw_rate_stand = 9.8 * 0.85 / 1
        else:
            # yaw_rate_stand = abs(9.8 * 0.85 / obs['vehicle_info']['ego']['v'])  # + 0.5 * u0[0] * self.dt)
            yaw_rate_stand = abs(9.8 * 0.85 / (obs['vehicle_info']['ego']['v'] * dyn_bill))

        sideslip_angle = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
        yaw_rate = (obs['vehicle_info']['ego']['v']) * math.tan(
            sideslip_angle) / wb_dis
        if abs(yaw_rate) > yaw_rate_stand * dyn_rate:
            # print('yaw_rate over stand:', yaw_rate)
            if u0[1] - self.pre_u0[1] > 0:
                while True:
                    u0[1] = u0[1] - 0.00001
                    sideslip_angle_test = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
                    yaw_rate_test = (obs['vehicle_info']['ego']['v']) * math.tan(
                        sideslip_angle_test) / wb_dis
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        break
            elif u0[1] - self.pre_u0[1] < 0:
                while True:
                    u0[1] = u0[1] + 0.00001
                    sideslip_angle_test = math.atan(wb_dis * math.tan(u0[1]) / vel_len)
                    yaw_rate_test = (obs['vehicle_info']['ego']['v']) * math.tan(
                        sideslip_angle_test) / wb_dis
                    if abs(yaw_rate_test) < yaw_rate_stand * dyn_rate:
                        break

        # print(u0)
        return u0


#基于 Pure Pursuit（纯跟踪） 轨迹跟踪方法计算期望转向角 (theta)，该方法确保自动驾驶车辆沿着局部路径 (local_waypoints) 进行平稳跟踪。
#输入是局部规划路径点序列和车辆当前状态数据
#输出是计算得到的转向角度
    def later_control(self, local_waypoints, obs):
        vel_dis = (obs['vehicle_info']['ego']['v'] + 0.5 * obs['vehicle_info']['ego']['a'] * self.dt) * self.dt
        init_dis = 10
        init_pos = 0
        for i in range(len(local_waypoints)):
            way_distance = pow(pow((local_waypoints[i][0] - obs['vehicle_info']['ego']['x']), 2) + pow((local_waypoints[i][1] - obs['vehicle_info']['ego']['y']), 2), 0.5)
            if abs(way_distance - vel_dis) < init_dis:
                init_dis = way_distance
                init_pos = i
        # print(init_dis, init_pos, len(local_waypoints), vel_dis)
        target_x = local_waypoints[init_pos][0]
        target_y = local_waypoints[init_pos][1]

        # local_waypoints = np.vstack(local_waypoints)
        # result_x = local_waypoints[:, 0]
        # result_y = local_waypoints[:, 1]
        # target_x = result_x[1]
        # target_y = result_y[1]
        init_x = obs['vehicle_info']['ego']['x']
        init_y = obs['vehicle_info']['ego']['y']
        inityaw = obs['vehicle_info']['ego']['yaw']
        vel_wb = obs['vehicle_info']['ego']['length'] / 1.7
        # vel_dis = (obs['vehicle_info']['ego']['v'] + 0.5 * obs['vehicle_info']['ego']['a'] * self.dt) * self.dt
        diff_x = target_x - init_x
        diff_y = target_y - init_y
        # print(diff_x, diff_y)
        target_l = pow(pow((target_x - init_x), 2) + pow((target_y - init_y), 2), 0.5)
        # target_a = math.atan2((result_y - init_y), (result_x - init_x))
        target_a = math.atan2(diff_y, diff_x)
        alfa = target_a - inityaw
        # alfa_sin = math.sin(alfa)
        # the = 2*vel_wb*math.sin(alfa)/target_l
        # print('the:', the)

        theta = math.atan(2*vel_wb*math.sin(alfa)/target_l)
        # print('later_theta:', theta)
        return theta
    
    def vehicle_dynamic(self, state, action, vel_len, dt):
        init_state = state
        a, rot = action

        final_state = {
            'px': 0,
            'py': 0,
            'v': 0,
            'heading': 0
        }
        # 首先根据旧速度更新本车位置
        # 更新本车转向角
        final_state['heading'] = init_state['heading'] + \
            init_state['v'] / vel_len * np.tan(rot) * dt

        # 更新本车速度
        final_state['v'] = init_state['v'] + a * dt

        # 更新X坐标
        final_state['px'] = init_state['px'] + init_state['v'] * \
            dt * np.cos(init_state['heading'])  # *np.pi/180

        # 更新Y坐标
        final_state['py'] = init_state['py'] + init_state['v'] * \
            dt * np.sin(init_state['heading'])  # *np.pi/180

        return final_state
    
    def position_orientation_objective(self, u: np.array, x0_array: np.array, x1_array: np.array, vel_l: float, dt: float, e: np.array = np.array([2e-3, 2e-3, 3e-3])) -> float:
        """
        Position-Orientation objective function to be minimized for the state transition feasibility.

        Simulates the next state using the inputs and calculates the norm of the difference between the
        simulated next state and actual next state. Position, velocity and orientation state fields will
        be used for calculation of the norm.

        :param u: input values
        :param x0: initial state values
        :param x1: next state values
        :param dt: delta time
        :param vehicle_dynamics: the vehicle dynamics model to be used for forward simulation
        :param ftol: ftol parameter used by the optimizer
        :param e: error margin, function will return norm of the error vector multiplied with 100 as cost
            if the input violates the friction circle constraint or input bounds.
        :return: cost
        """
        x0 = {
            'px': x0_array[0],
            'py': x0_array[1],
            'v': x0_array[2],
            'heading': x0_array[3],
        }

        x1_target = x1_array
        x1_sim = self.vehicle_dynamic(x0, u, vel_l, dt)
        x1_sim_array = np.array([x1_sim['px'], x1_sim['py'], x1_sim['v'] *
                                np.cos(x1_sim['heading']), x1_sim['v']*np.sin(x1_sim['heading'])])

        # if the input violates the constraints
        if x1_sim is None:
            return np.linalg.norm(e * 100)

        else:
            diff = np.subtract(x1_target, x1_sim_array)
            cost = np.linalg.norm(diff)
            return cost
    
    def calc_car_dist(self, car_1, car_2):
        dist = np.linalg.norm([car_1[0] - car_2[0], car_1[1] - car_2[1]]) - \
            (car_1[4] + car_2[4]) / 2
        return dist

    def get_lead_follow_car_info(self, car_found, car_info, which_lane):
        if car_found:
            car_pos = np.array([
                [car_info[which_lane]['x'], car_info[which_lane]['y'], car_info[which_lane]['yaw']]])
            car_length = np.array([
                [car_info[which_lane]['length']]])
            car_speed = np.array([
                [car_info[which_lane]['v']]])
        else:
            car_pos = np.array([
                [99999.0, 99999.0, 0.0]])
            car_length = np.array([
                [99999.0]])
            car_speed = np.array([
                [99999.0]])
            
        return car_pos, car_length, car_speed
    
    def create_controller_output_dir(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def write_control_file(self, u0):
        self.create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
        file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'control.txt')
        with open(file_name, 'a') as control_file:
                control_file.write('%.2f\t%.2f\n' %\
                                    (u0[0], u0[1]))

## 递归的方式实现贝塞尔曲线
def bezier(Ps,n,t):
    """递归的方式实现贝塞尔曲线

    Args:
        Ps (_type_): 控制点，格式为numpy数组：array([[x1,y1],[x2,y2],...,[xn,yn]])
        n (_type_): n个控制点，即Ps的第一维度
        t (_type_): 步长t

    Returns:
        _type_: 当前t时刻的贝塞尔点
    """
    if n==1:
        return Ps[0]
    return (1-t)*bezier(Ps[0:n-1],n-1,t)+t*bezier(Ps[1:n],n-1,t)
