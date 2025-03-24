import numpy as np
import math
def extract_nearby_vehicles(observation, highway, nearby_vehicles_ago=[]):
    if highway == True:
        pre_dis = 50
    else:
        pre_dis = 30
    ego_x = observation.ego_info.x
    ego_y = observation.ego_info.y
    objects_to_store_vehicle = []
    objects_to_store_bicycle = []
    objects_to_store_pedestrian = []
    for vehicle_id, vehicle_info in observation.object_info['vehicle'].items():
        vehicle_x = vehicle_info.x
        vehicle_y = vehicle_info.y
        distance = ((vehicle_x - ego_x) ** 2 + (vehicle_y - ego_y) ** 2) ** 0.5
        if distance < pre_dis:
            objects_to_store_vehicle.append([vehicle_id, vehicle_info])
    for bicycle_id, bicycle_info in observation.object_info['bicycle'].items():
        vehicle_x = bicycle_info.x
        vehicle_y = bicycle_info.y
        distance = ((vehicle_x - ego_x) ** 2 + (vehicle_y - ego_y) ** 2) ** 0.5
        if distance < 30:
            objects_to_store_bicycle.append([bicycle_id, bicycle_info])
    for pedestrian_id, pedestrian_info in observation.object_info['pedestrian'].items():
        pedestrian_x = pedestrian_info.x
        pedestrian_y = pedestrian_info.y
        distance = ((pedestrian_x - ego_x) ** 2 + (pedestrian_y - ego_y) ** 2) ** 0.5
        if distance < 30:
            objects_to_store_pedestrian.append([pedestrian_id, pedestrian_info])
    objects_to_store = objects_to_store_vehicle + objects_to_store_bicycle + objects_to_store_pedestrian
    # print(objects_to_store)
    num_rows = len(objects_to_store)
    nearby_vehicles = np.empty((num_rows, 8), dtype=object)
    for i, object_info in enumerate(objects_to_store):
        vehicle_id, vehicle_info = object_info
        nearby_vehicles[i, 0] = vehicle_id
        nearby_vehicles[i, 1] = vehicle_info.x
        nearby_vehicles[i, 2] = vehicle_info.y
        nearby_vehicles[i, 3] = vehicle_info.v
        nearby_vehicles[i, 4] = vehicle_info.width
        nearby_vehicles[i, 5] = vehicle_info.length
        nearby_vehicles[i, 6] = vehicle_info.a
        nearby_vehicles[i, 7] = vehicle_info.yaw

    nearby_vehicles_ago.append(nearby_vehicles)
    # nearby_vehicles_ago.append(nearby_vehicles)
    if len(nearby_vehicles_ago) > 5:
        nearby_vehicles_ago.pop(0)
    # print(nearby_vehicles_ago)
    # print(nearby_vehicles)
    # 获取最后一帧的车辆名称
    last_frame_vehicles = nearby_vehicles_ago[-1][:, 0]

    # 创建一个字典用于存储每个车辆的数据
    vehicle_data_dict = {}

    # 遍历最后一帧的车辆名称
    for vehicle_name in last_frame_vehicles:
        # 在前几帧中查找该车辆的数据
        vehicle_data = []
        found = False
        for i in range(len(nearby_vehicles_ago) - 1, -1, -1):  # 从最后一帧向前遍历
            # 在当前时间步的数据中查找目标车辆的索引
            target_index = np.where(nearby_vehicles_ago[i][:, 0] == vehicle_name)[0]
            if len(target_index) > 0:
                # 找到目标车辆的索引，获取其数据并存储到列表中
                vehicle_data.insert(0, nearby_vehicles_ago[i][target_index[0]])  # 插入到列表的第一个位置
                found = True
            # 如果已经找到五帧数据或者到达最前边的一帧，则停止查找
            if len(vehicle_data) == 5 or i == 0:
                break

        # 如果在前几帧中未找到该车辆的数据，则使用最近出现的一帧数据补充
        if not found:
            vehicle_data.extend([nearby_vehicles_ago[-2][-1]] * (5 - len(vehicle_data)))  # 使用倒数第二帧的最后一个数据进行补充

        # 将车辆数据存储到字典中
        vehicle_data_dict[vehicle_name] = np.array(vehicle_data)
    return vehicle_data_dict
# return grouped_vehicle_data
def obstacle_prediction(vehicle_data_dict, dt):
    # dt = dt
    for vehicle_name, data_array in vehicle_data_dict.items():
        num_frames = len(data_array)
        if num_frames < 5:
            # 补齐不足五帧的数据
            missing_frames = 5 - num_frames
            last_frame = data_array[-1]
            missing_data = np.tile(last_frame, (missing_frames, 1))
            vehicle_data_dict[vehicle_name] = np.vstack((data_array, missing_data))

            # print(vehicle_data)
    list1 = []
    for data in vehicle_data_dict:
        # if data != 'car1':
            for j in range(1, 8):
                for i in range(5):
                    list1.append(vehicle_data_dict[data][i, j])
    # print(list1)
    cell_arrays = [list1[i:i+5] for i in range(0, len(list1), 5)]
    # print(cell_arrays)
    # 将每七个元胞数组作为一行存入二维数组
    rows = [cell_arrays[i:i+7] for i in range(0, len(cell_arrays), 7)]
    def prediction_angle(angle_list,num):
        x_data = np.array([1, 2, 3, 4, 5])
        y_data = angle_list

        # 进行多项式拟合
        degree = 2  # 多项式的次数
        coefficients = np.polyfit(x_data, y_data, degree)

        # 定义一个函数来计算多项式的值
        def polynomial(x, coeffs):
            return sum(c * x ** i for i, c in enumerate(coeffs[::-1]))

        # 估算未知点的值
        x_unknown = num + 5
        angle_predict = polynomial(x_unknown, coefficients)
        return angle_predict
    # print(rows)
    # print(rows[2])
    predict_array = np.empty((len(rows), 20))
    for i in range(len(rows)):
        vehicle_list = rows[i]
        # print(vehicle_list)
        x_list = vehicle_list[0]
        y_list = vehicle_list[1]
        v_list = vehicle_list[2]
        a_list = vehicle_list[5]
        angle_list = vehicle_list[6]
        # longth_list = vehicle_list[4]
        # print(angle_list)
        angle_predict = np.zeros(15)
        for j in range(15):
            angle_predict[j] = prediction_angle(angle_list, j + 1)
        # print(angle_predict)
        predict_array[i, 0:5] = [x_list[-1], y_list[-1], angle_list[-1], vehicle_list[3][0], vehicle_list[4][0]]
        v = 0.07 * v_list[0] + 0.11 * v_list[1] + 0.14 * v_list[2] + 0.18 * v_list[3] + 0.23 * v_list[4] + 0.27 * ( v_list[0] + a_list[-1] * 0.1)
        detax = 0
        detay = 0
        for k in range(5):
            angle = angle_predict[k]
            detax = detax + v * dt * math.cos(angle)
            detay = detay + v * dt * math.sin(angle)

        x_predict_5_1 = x_list[-1] + detax
        y_predict_5_1 = y_list[-1] + detay

        x_predict_5_2 = prediction_angle(x_list,5)
        y_predict_5_2 = prediction_angle(y_list,5)
        x_predict_5 = 0.95 * x_predict_5_1 + 0.05 * x_predict_5_2
        y_predict_5 = 0.95 * y_predict_5_1 + 0.05 * y_predict_5_2

        predict_array[i, 5:10] = [x_predict_5, y_predict_5, angle_predict[4], vehicle_list[3][0], vehicle_list[4][0]]
        detax = 0
        detay = 0
        for l in range(10):
            angle = angle_predict[l]
            detax = detax + v * dt * math.cos(angle)
            detay = detay + v * dt * math.sin(angle)
        # print('detax, detay:', detax, detay,v,angle)
        x_predict_10_1 = x_list[-1] + detax
        y_predict_10_1 = y_list[-1] + detay
        x_predict_10_2 = prediction_angle(x_list, 10)
        y_predict_10_2 = prediction_angle(y_list, 10)
        x_predict_10 = 0.95 * x_predict_10_1 + 0.05 * x_predict_10_2
        y_predict_10 = 0.95 * y_predict_10_1 + 0.05 * y_predict_10_2
        predict_array[i, 10:15] = [x_predict_10, y_predict_10, angle_predict[9], vehicle_list[3][0], vehicle_list[4][0]]
        detax = 0
        detay = 0
        for l in range(15):
            angle = angle_predict[l]
            detax = detax + v * dt * math.cos(angle)
            detay = detay + v * dt * math.sin(angle)
        x_predict_15_1 = x_list[-1] + detax
        y_predict_15_1 = y_list[-1] + detay
        x_predict_15_2 = prediction_angle(x_list, 15)
        y_predict_15_2 = prediction_angle(y_list, 15)
        x_predict_15 = 0.95 * x_predict_15_1 + 0.05 * x_predict_15_2
        y_predict_15 = 0.95 * y_predict_15_1 + 0.05 * y_predict_15_2
        predict_array[i, 15:20] = [x_predict_15, y_predict_15, angle_predict[14], vehicle_list[3][0], vehicle_list[4][0]]
        # print('predict_array:', predict_array)
    return predict_array
    # return grouped_vehicle_data
