import math
import numpy as np
import pandas as pd


def avoid_rear_colission(observation):
    new_acc = 0
    frame = pd.DataFrame()
    for key, value in observation['vehicle_info'].items():
        sub_frame = pd.DataFrame(value, columns=['x', 'y', 'v', 'yaw', 'length', 'width'], index=[key])
        frame = pd.concat([frame, sub_frame])
    state = frame.to_numpy()
    if state[0, 3] < np.pi / 2 or state[0, 3] > np.pi * 3 / 2:
        direction = 1.0
    else:
        direction = -1.0
    state[:, 0] = state[:, 0] * direction
    ego = state[0, :]
    v = ego[2]
    # 在本车后侧
    x_ind = ego[0] > state[:, 0]
    y_ind = (np.abs(ego[1] - state[:, 1])) < ((ego[5] + state[:, 5]) / 2)
    ind = x_ind & y_ind
    if ind.sum() > 0:
        state_ind = state[ind, :]
        rear = state_ind[(ego[0] - state_ind[:, 0]).argmin(), :]
        rv = rear[2]
        dist = ego[0] - rear[0] - (ego[4] + rear[4]) / 2
        if abs(rv) > abs(v):
            relative_speed = abs(rv) - abs(v)
            time_to_collision = dist / relative_speed
            # 判断短时间内是否可能发生碰撞
            if time_to_collision < 5:  # 可根据需要调整时间阈值
                # 计算避免碰撞的加速度
                new_acc = 5 * relative_speed / time_to_collision
        if dist < 10:
            new_acc = 3
            # 在本车前侧
            x_ind = ego[0] < state[:, 0]
            y_ind = (np.abs(ego[1] - state[:, 1])) < ((ego[5] + state[:, 5]) / 2)
            ind = x_ind & y_ind
            if ind.sum() > 0:
                state_ind = state[ind, :]
                front = state_ind[(state_ind[:, 0] - ego[0]).argmin(), :]
                dist2 = front[0] - ego[0] - (ego[4] + front[4]) / 2
                if dist2 < dist:
                    new_acc = -3
    return new_acc





