import numpy as np
# import matplotlib.pyplot as plt

def JCK_point(points):
    # # 加载点的数据集
    # points = np.load("8_3_2_418.npy")
    # 按照x坐标对点进行排序
    sorted_points_by_x = points[np.argsort(points[:, 0])]
    # 获取x最小的九个点
    x_min_points = sorted_points_by_x[:9]
    # 按照y坐标对x最小的九个点进行排序
    sorted_x_min_points_by_y = x_min_points[np.argsort(x_min_points[:, 1])]
    # 获取y的最大和最小值
    y_max = sorted_x_min_points_by_y[-1, 1]
    index = np.argmax(x_min_points[:, 1])
    x_a = x_min_points[index, 0]
    y_a = y_max
    # 获取x最大的九个点
    x_max_points = sorted_points_by_x[-9:]
    # 按照y坐标对x最大的九个点进行排序
    sorted_x_max_points_by_y = x_max_points[np.argsort(x_max_points[:, 1])]
    # 获取y的最大和最小值
    y_max2 = sorted_x_max_points_by_y[-1, 1]
    index2 = np.argmax(x_max_points[:, 1])
    x_b = x_max_points[index2, 0]
    y_b = y_max2

    # 按照y坐标对点进行排序
    sorted_points_by_y = points[np.argsort(points[:, 1])]
    # 获取y最小的九个点
    y_min_points = sorted_points_by_y[:9]
    # 按照x坐标对y最小的九个点进行排序
    sorted_y_min_points_by_x = y_min_points[np.argsort(y_min_points[:, 0])]
    # 获取x的最大和最小值
    x_min = sorted_y_min_points_by_x[0, 0]
    index = np.argmin(y_min_points[:, 0])
    y_c = y_min_points[index, 1]
    x_c = x_min
    # 获取y最大的九个点
    y_max_points = sorted_points_by_y[-9:]
    # 按照x坐标对y最大的九个点进行排序
    sorted_y_max_points_by_x = y_max_points[np.argsort(y_max_points[:, 0])]
    # 获取x的最小值
    x_min2 = sorted_y_max_points_by_x[0, 0]
    index2 = np.argmax(y_max_points[:, 0])
    y_d = y_max_points[index2, 1]
    x_d = x_min2

    # 获取y的最大和最小值
    y_min = sorted_x_min_points_by_y[0, 1]
    y_avg = (y_min + y_max) / 2
    # 获取x的最大和最小值
    x_max = sorted_y_min_points_by_x[-1, 0]
    x_avg = (x_min + x_max) / 2

    # 设置x轴范围和间隔
    xmin = int(np.min(points[:, 0]))
    xmax = int(np.max(points[:, 0]))
    x_interval = 0.4  # x轴的间隔
    threshold = 0.2
    for x in np.arange(xmin, xmax + x_interval, x_interval):
        x_points = points[np.abs(points[:, 0] - x) <= threshold]
        if len(x_points) > 0:
            if np.max(x_points[:, 1]) - y_max > 0.2:
                x1 = x
                y1 = np.max(x_points[:, 1])
                break
    y0 = y_max
    for x in np.arange(x1, xmax + x_interval, x_interval):
        x_points = points[np.abs(points[:, 0] - x) <= threshold]
        if len(x_points) > 0:
            max_y_index = np.argmax(x_points[:, 1])
            if np.max(x_points[:, 1]) - y0 > 5:
                x2 = x - x_interval
                y2 = y0
                break
            y0 = x_points[max_y_index, 1]
    x11 = 2 * x_avg - x1
    y22 = 2 * y_avg - y2

    interval = 0.2
    # 区间 [x_a, x1] 内的等间隔插值点
    x_values_1 = np.arange(x_a, x1 + interval, interval)
    y_values_1 = np.interp(x_values_1, [x_a, x_b], [y_a, y_b])
    # 区间 [x11, x_b] 内的等间隔插值点
    x_values_2 = np.arange(x11, x_b + interval, interval)
    y_values_2 = np.interp(x_values_2, [x_a, x_b], [y_a, y_b])
    # 合并两个区间的插值点
    x_values = np.concatenate((x_values_1, x_values_2))
    y_values = np.concatenate((y_values_1, y_values_2))
    y_values_shift1 = y_values - (y_max - y_avg)
    y_values_shift2 = y_values - 2 * (y_max - y_avg)
    points1 = np.column_stack((x_values, y_values))
    points_shift1 = np.column_stack((x_values, y_values_shift1))
    points_shift2 = np.column_stack((x_values, y_values_shift2))
    # 合并插值点集和平移后的点集
    points_combined1 = np.concatenate((points1, points_shift1, points_shift2))

    # 区间 [x_c, x_d] 内的等间隔插值点
    y_values_3 = np.arange(y_c, y22 + interval, interval)
    x_values_3 = np.interp(y_values_3, [y_c, y_d], [x_c, x_d])
    y_values_4 = np.arange(y2, y_d + interval, interval)
    x_values_4 = np.interp(y_values_4, [y_c, y_d], [x_c, x_d])
    # 合并两个区间的插值点
    x_values = np.concatenate((x_values_3, x_values_4))
    y_values = np.concatenate((y_values_3, y_values_4))
    x_values_shift3 = x_values + (x_max - x_avg)
    x_values_shift4 = x_values + 2 * (x_max - x_avg)
    # 平移后的点集
    points2 = np.column_stack((x_values, y_values))
    points_shift3 = np.column_stack((x_values_shift3, y_values))
    points_shift4 = np.column_stack((x_values_shift4, y_values))
    # 合并插值点集和平移后的点集
    points_combined2 = np.concatenate((points2, points_shift3, points_shift4))

    # 在点(x_values_1(-1),y_values_1(-1))与(x_value_4(0),y_values_4(0))之间插值
    x_values_5 = np.arange(x_values_1[-1], x_values_4[0] + interval, interval)
    y_values_5 = np.interp(x_values_5, [x_values_1[-1], x_values_4[0]], [y_values_1[-1], y_values_4[0]])
    points3 = np.column_stack((x_values_5, y_values_5))
    # 在点(x_value_4(0)+2*(x_max - x_avg),y_values_4(0))与(x_values_2(0),y_values_2(0))之间插值
    x_values_6 = np.arange(x_values_4[0] + 2 * (x_max - x_avg), x_values_2[0] + interval, interval)
    y_values_6 = np.interp(x_values_6, [x_values_4[0] + 2 * (x_max - x_avg), x_values_2[0]], [y_values_4[0], y_values_2[0]])
    points4 = np.column_stack((x_values_6, y_values_6))
    # 在点(x_value_3(-1)+2*(x_max - x_avg),y_values_3(-1))与(x_values_2(0),y_values_2(0)-2 * (y_max - y_avg))之间插值
    x_values_7 = np.arange(x_values_3[-1] + 2 * (x_max - x_avg), x_values_2[0] + interval, interval)
    y_values_7 = np.interp(x_values_7, [x_values_3[-1] + 2 * (x_max - x_avg), x_values_2[0]], [y_values_3[-1], y_values_2[0] - 2 * (y_max - y_avg)])
    points5 = np.column_stack((x_values_7, y_values_7))
    # 在点(x_values_1(-1),y_values_1(-1)-2*(y_max - y_avg))与(x_value_3(-1),y_values_3(-1))之间插值
    x_values_8 = np.arange(x_values_1[-1], x_values_3[-1] + interval, interval)
    y_values_8 = np.interp(x_values_8, [x_values_1[-1], x_values_3[-1]], [y_values_1[-1] - 2 * (y_max - y_avg), y_values_3[-1]])
    points6 = np.column_stack((x_values_8, y_values_8))
    # 合并插值点集
    points_combined3 = np.concatenate((points3, points4, points5, points6))

    points2 = np.concatenate((points_combined1, points_combined2, points_combined3))
    return points2

    # # 画出处理前后的点集对比图
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].scatter(points[:, 0], points[:, 1], s=1)
    # ax[0].set_title("Before")
    # ax[1].scatter(points2[:, 0], points2[:, 1], s=1)
    # ax[1].set_title("After")
    # plt.show()

