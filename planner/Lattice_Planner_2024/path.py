import numpy as np
import matplotlib.pyplot as plt

def path(sx, sy, gx, gy, road_width):
    # 先换道
    if gx > sx:
        gx = gx + 5
        x1 = sx + abs((gy - sy)/road_width) * 30
        y1 = gy
    else:
        gx = gx - 5
        x1 = sx - abs((gy - sy)/road_width) * 30
        y1 = gy

    # print(abs((gy - sy)/road_width))

    # if gx > sx:
    #     gx = gx + 5
    #     x1 = sx + 25
    #     y1 = gy
    # else:
    #     gx = gx - 5
    #     x1 = sx - 25
    #     y1 = gy

    # if gx > sx:
    #     x1 = gx - 40
    #     y1 = sy
    # else:
    #     x1 = gx + 40
    #     y1 = sy

    # 在点(x1, y1)与(sx, sy)之间进行线性插值
    num_points1 = int(abs(sx - x1)) + 1  # 插值点的数量，根据距离差进行计算
    x_values1 = np.linspace(sx, x1, num_points1)  # 在x轴方向上进行线性插值
    y_values1 = np.linspace(sy, y1, num_points1)  # 在y轴方向上进行线性插值

    # 在点(x1, y1)与(gx, gy)之间进行线性插值
    num_points2 = int(abs(gx - x1)) + 1  # 插值点的数量，根据距离差进行计算
    x_values2 = np.linspace(x1, gx, num_points2)  # 在x轴方向上进行线性插值
    y_values2 = np.linspace(y1, gy, num_points2)  # 在y轴方向上进行线性插值

    # 合并路径点，确保路径点从起点sx到终点gx有顺序
    x_values = np.concatenate((x_values1, x_values2[1:]))
    y_values = np.concatenate((y_values1, y_values2[1:]))
    x_values = np.flip(x_values)
    y_values = np.flip(y_values)
    # 返回插值后的路径点
    return x_values, y_values

