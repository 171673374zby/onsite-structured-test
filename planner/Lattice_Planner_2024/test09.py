import numpy as np
from scipy.interpolate import splprep, splev

def project_curvature_and_heading(path, x_new, y_new, curvatures):
    projected_curvatures = []
    projected_headings_degrees = []
    min_dist = float('inf')
    index_list = []
    for i in range(len(path[0])):
        dist = []
        for j in range(len(x_new)):
        # 计算原始路径点在增密后路径上的最近点索引
            distances = np.sqrt((x_new[j] - path[0][i]) ** 2 + (y_new[j] - path[1][i]) ** 2)
            dist.append(distances)
            if distances < min_dist:
                min_dist = distances
        index = np.argmin(np.abs(np.array(dist) - min_dist))
        index_list.append(index)
    result = np.array([curvatures[i] for i in index_list])

        #     # 获取最近点处的曲率和航向角
        # projected_curvature = curvatures[nearest_index]
        # projected_heading = headings_degrees[nearest_index]
        #
        # # 将计算得到的曲率和航向角添加到结果数组中
        # projected_curvatures.append(projected_curvature)
        # projected_headings_degrees.append(projected_heading)

    return result

def cal_localwaypoints(best_path, s_dot_qp):
    v_extended = s_dot_qp.tolist() + [0] * (len(best_path[0]) - len(s_dot_qp))
    # print('pre_v:', v_extended)

    # 组合成新的数组
    local_waypoints = [[best_path[0][i], best_path[1][i], v_extended[i]] for i in range(len(best_path[0]))]
    return local_waypoints