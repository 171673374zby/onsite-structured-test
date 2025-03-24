import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt
import math
def calculate_ego_circle_properties(path, self_Car, predict_t, dt):
    # 计算根节点圆的半径
    # print('self_Car:',self_Car)
    vehicle_length = self_Car[-1]
    vehicle_width = self_Car[-2]
    radius = math.sqrt(vehicle_length ** 2 / 3 ** 2 + vehicle_width ** 2) / 2
    # 计算偏移量
    offset = math.sqrt(radius ** 2 - vehicle_width ** 2 / 4) * 2
    # distance = path[0] * predict_t

    path_2 = path[:2]
    # print(path)
    # 将路径转换为 NumPy 数组以便计算
    path_array = np.array(path_2)
    current_position = [path_2[0][0], path_2[1][0]]
    # print('current_position:', current_position)
    # print('current_position:',current_position)
    distances = np.linalg.norm(path_array - np.array(current_position)[:, np.newaxis], axis=0)

    desired_distance = self_Car[0] * predict_t * dt * 10

    # 找到最接近的距离
    index = np.argmin(np.abs(distances - desired_distance))
    # angle_radians = path[2][0]
    center_x = path[0][index]
    center_y = path[1][index]
    angle_radians = path[2][index]
    # 计算中点前后的圆心横坐标
    center_front_x = center_x + offset * math.cos(angle_radians)
    center_rear_x = center_x - offset * math.cos(angle_radians)

    # 计算中点前后的圆心纵坐标
    center_front_y = center_y + offset * math.sin(angle_radians)
    center_rear_y = center_y - offset * math.sin(angle_radians)
    circle_locations =[[center_rear_x, center_rear_y], [center_x, center_y], [center_front_x, center_front_y]]
    # print('circle_locations',circle_locations)
    return circle_locations, radius
def calculate_obstacle_circle_properties(center_x, center_y, vehicle_length, vehicle_width, angle, n):

    radius = math.sqrt(vehicle_length ** 2 / n ** 2 + vehicle_width ** 2) / 2

    # 计算偏移量
    offset = math.sqrt(radius ** 2 - vehicle_width ** 2 / 4) * 2

    # 转换角度为弧度
    angle_radians = angle
    center_circle = np.zeros((n, 2))
    t = n//2
    # 计算中点前后的圆心横坐标
    for i in range(t):
        center_circle[i, 0] = center_x + (i + 1) * offset * math.cos(angle_radians)
        center_circle[i, 1] = center_y + (i + 1) * offset * math.sin(angle_radians)
        center_circle[-i - 1, 0] = center_x - (i + 1) * offset * math.cos(angle_radians)
        center_circle[-i - 1, 1] = center_y - (i + 1) * offset * math.sin(angle_radians)
    center_circle[t, 0] = center_x
    center_circle[t, 1] = center_y

    return center_circle, radius
def collision_check(paths, predict_data, self_Car, dt):
    collision_check_array = np.zeros(len(paths), dtype=bool)
    for u in range(len(paths)):
        path           = paths[u]
        # print(path[0][0], path[1][0])
        # circle_locations = np.empty((3, 3), dtype=object)
        collision_check_array[u] = True

        for j in range(4):
            predict_t = j * 0.5
            circle_locations, ego_radius = calculate_ego_circle_properties(path, self_Car, predict_t, dt)
            # for k in range(3):
            # k = 0
            for l in range(len(predict_data)):
                # print('len(predict_data)', len(predict_data))
                vehicle_length = predict_data[l][4]
                if vehicle_length < 4.5:
                    n = 3
                elif vehicle_length < 6:
                    n = 5
                elif vehicle_length < 8:
                    n = 7
                else:
                    n = 9
                for m in range(n):
                # obstacles = obstacles[j]

                    center_x = predict_data[l][j * 5]
                    center_y = predict_data[l][j * 5 + 1]
                    vehicle_width = predict_data[l][j * 5 + 3]
                    angle = predict_data[l][j * 5 + 2]
                    center_circle_ob, radius_ob = calculate_obstacle_circle_properties(center_x, center_y, vehicle_length, vehicle_width, angle, n)
                    safe_distance = ego_radius + radius_ob
                    test_distance1 = math.sqrt((circle_locations[0][0] - center_circle_ob[m][0])**2 + (center_circle_ob[m][1] - circle_locations[0][1])**2)
                    test_distance2 = math.sqrt((circle_locations[1][0] - center_circle_ob[m][0])**2 + (center_circle_ob[m][1] - circle_locations[1][1])**2)
                    test_distance3 = math.sqrt((circle_locations[2][0] - center_circle_ob[m][0]) ** 2 + ( center_circle_ob[m][1] - circle_locations[2][1]) ** 2)
                    # if j == 0:
                        # print(test_distance1,test_distance2,test_distance3)
                    if safe_distance > test_distance1 or safe_distance > test_distance2 or safe_distance > test_distance3:
                        collision_check_array[u] = False
                        # print(test_distance1, test_distance2, test_distance3)
                    # break
                # break
        # if not collision_check_array[u]:
        #     break
        # break
    return collision_check_array
# collision_check_array = collision_check(paths, predict_data, self_Car)
# print('collision_check_array',collision_check_array)
