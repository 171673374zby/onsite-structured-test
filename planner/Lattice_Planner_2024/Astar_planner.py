# import heapq
# import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy

# from pylab import *

show_animation = False


class AStar():
    def __init__(self, map_grid, start, goal, max_index, scene_round):
        self.g = 0  # g初始化为0
        # self.start = numpy.array([5, 2])  # 起点坐标
        # self.goal = numpy.array([15, 15])  # 终点坐标
        self.start = numpy.array([start[0], start[1]])  # 起点坐标
        self.goal = numpy.array([goal[0], goal[1]])  # 终点坐标
        self.open = numpy.array([[], [], [], [], [], []])  # 先创建一个空的open表, 记录坐标，方向，g值，f值
        self.closed = numpy.array([[], [], [], [], [], []])  # 先创建一个空的closed表
        self.best_path_array = numpy.array([[], []])  # 回溯路径表
        self.map_grid = map_grid
        self.max_x = max_index[0]
        self.max_y = max_index[1]
        self.scene_round = scene_round

    def h_value_tem(self, son_p):
        """
        计算拓展节点和终点的h值
        :param son_p:子搜索节点坐标
        :return:
        """
        h = (son_p[0] - self.goal[0]) ** 2 + (son_p[1] - self.goal[1]) ** 2
        h = numpy.sqrt(h)  # 计算h
        return h

    def g_accumulation(self, son_point, father_point):
        """
        累计的g值
        :return:
        """
        g1 = father_point[0] - son_point[0]
        g2 = father_point[1] - son_point[1]
        g = g1 ** 2 + g2 ** 2
        g = numpy.sqrt(g) + father_point[4]  # 加上累计的g值
        return g

    def f_value_tem(self, son_p, father_p):
        """
        求出的是临时g值和h值加上累计g值得到全局f值
        :param father_p: 父节点坐标
        :param son_p: 子节点坐标
        :return:f
        """
        f = self.g_accumulation(son_p, father_p) + self.h_value_tem(son_p) + self.map_grid[int(son_p[0]), int(son_p[1])]
        return f

    def child_point(self, x):
        """
        拓展的子节点坐标
        :param x: 父节点坐标
        :return: 子节点存入open表，返回值是每一次拓展出的子节点数目，用于撞墙判断
        当搜索的节点撞墙后，如果不加处理，会陷入死循环
        """
        # 开始遍历周围8个节点
        # print(x)
        for j in range(-1, 2, 1):
            for q in range(-1, 2, 1):

                if j == 0 and q == 0:  # 搜索到父节点去掉
                    continue
                m = [x[0] + j, x[1] + q]

                # print(m)
                if m[0] < 0 or m[0] > self.max_x - 1 or m[1] < 0 or m[1] > self.max_y - 1:  # 搜索点出了边界去掉
                    continue
                #
                if self.map_grid[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                    continue

                record_g = self.g_accumulation(m, x)
                record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                # print(para)

                # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                a, index = self.judge_location(m, self.open)
                if a == 1:
                    # 说明open中已经存在这个点

                    if record_f <= self.open[5][index]:
                        self.open[5][index] = record_f
                        self.open[4][index] = record_g
                        self.open[3][index] = y_direction
                        self.open[2][index] = x_direction

                    continue

                # 在closed表中,则去掉搜索点
                b, index2 = self.judge_location(m, self.closed)
                if b == 1:

                    if record_f <= self.closed[5][index2]:
                        self.closed[5][index2] = record_f
                        self.closed[4][index2] = record_g
                        self.closed[3][index2] = y_direction
                        self.closed[2][index2] = x_direction
                        self.closed = numpy.delete(self.closed, index2, axis=1)
                        self.open = numpy.c_[self.open, para]
                    continue

                self.open = numpy.c_[self.open, para]  # 参数添加到open中
                # print(self.open)

    def child_point_round(self, x):
        """
        拓展的子节点坐标
        :param x: 父节点坐标
        :return: 子节点存入open表，返回值是每一次拓展出的子节点数目，用于撞墙判断
        当搜索的节点撞墙后，如果不加处理，会陷入死循环
        """
        # 开始遍历周围8个节点
        if 67 < x[1] < 85 and 100 < x[0] < 140:
            for j in range(-2, 0, 1):
                for q in range(-1, 2, 1):

                    if j == 0 and q == 0:  # 搜索到父节点去掉
                        continue
                    m = [x[0] + j, x[1] + q]

                    # print(m)
                    if m[0] < 0 or m[0] > self.max_x - 1 or m[1] < 0 or m[1] > self.max_y - 1:  # 搜索点出了边界去掉
                        continue
                    #
                    if self.map_grid[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                        continue

                    record_g = self.g_accumulation(m, x)
                    record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                    x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                    para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                    # print(para)

                    # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                    # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                    a, index = self.judge_location(m, self.open)
                    if a == 1:
                        # 说明open中已经存在这个点

                        if record_f <= self.open[5][index]:
                            self.open[5][index] = record_f
                            self.open[4][index] = record_g
                            self.open[3][index] = y_direction
                            self.open[2][index] = x_direction

                        continue

                    # 在closed表中,则去掉搜索点
                    b, index2 = self.judge_location(m, self.closed)
                    if b == 1:

                        if record_f <= self.closed[5][index2]:
                            self.closed[5][index2] = record_f
                            self.closed[4][index2] = record_g
                            self.closed[3][index2] = y_direction
                            self.closed[2][index2] = x_direction
                            self.closed = numpy.delete(self.closed, index2, axis=1)
                            self.open = numpy.c_[self.open, para]
                        continue

                    self.open = numpy.c_[self.open, para]  # 参数添加到open中
                    # print(self.open)
        if 47 < x[1] < 67 and 100 < x[0] < 140:
            for j in range(0, 2, 1):
                for q in range(-1, 2, 1):

                    if j == 0 and q == 0:  # 搜索到父节点去掉
                        continue
                    m = [x[0] + j, x[1] + q]

                    # print(m)
                    if m[0] < 0 or m[0] > self.max_x - 1 or m[1] < 0 or m[1] > self.max_y - 1:  # 搜索点出了边界去掉
                        continue
                    #
                    if self.map_grid[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                        continue

                    record_g = self.g_accumulation(m, x)
                    record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                    x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                    para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                    # print(para)

                    # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                    # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                    a, index = self.judge_location(m, self.open)
                    if a == 1:
                        # 说明open中已经存在这个点

                        if record_f <= self.open[5][index]:
                            self.open[5][index] = record_f
                            self.open[4][index] = record_g
                            self.open[3][index] = y_direction
                            self.open[2][index] = x_direction

                        continue

                    # 在closed表中,则去掉搜索点
                    b, index2 = self.judge_location(m, self.closed)
                    if b == 1:

                        if record_f <= self.closed[5][index2]:
                            self.closed[5][index2] = record_f
                            self.closed[4][index2] = record_g
                            self.closed[3][index2] = y_direction
                            self.closed[2][index2] = x_direction
                            self.closed = numpy.delete(self.closed, index2, axis=1)
                            self.open = numpy.c_[self.open, para]
                        continue

                    self.open = numpy.c_[self.open, para]  # 参数添加到open中
                    # print(self.open)

        else:
            for j in range(-1, 2, 1):
                for q in range(-1, 2, 1):

                    if j == 0 and q == 0:  # 搜索到父节点去掉
                        continue
                    m = [x[0] + j, x[1] + q]

                    # print(m)
                    if m[0] < 0 or m[0] > self.max_x - 1 or m[1] < 0 or m[1] > self.max_y - 1:  # 搜索点出了边界去掉
                        continue
                    #
                    if self.map_grid[int(m[0]), int(m[1])] == 0:  # 搜索到障碍物去掉
                        continue

                    record_g = self.g_accumulation(m, x)
                    record_f = self.f_value_tem(m, x)  # 计算每一个节点的f值

                    x_direction, y_direction = self.direction(x, m)  # 每产生一个子节点，记录一次方向

                    para = [m[0], m[1], x_direction, y_direction, record_g, record_f]  # 将参数汇总一下
                    # print(para)

                    # 在open表中，则去掉搜索点，但是需要更新方向指针和self.g值
                    # 而且只需要计算并更新self.g即可，此时建立一个比较g值的函数
                    a, index = self.judge_location(m, self.open)
                    if a == 1:
                        # 说明open中已经存在这个点

                        if record_f <= self.open[5][index]:
                            self.open[5][index] = record_f
                            self.open[4][index] = record_g
                            self.open[3][index] = y_direction
                            self.open[2][index] = x_direction

                        continue

                    # 在closed表中,则去掉搜索点
                    b, index2 = self.judge_location(m, self.closed)
                    if b == 1:

                        if record_f <= self.closed[5][index2]:
                            self.closed[5][index2] = record_f
                            self.closed[4][index2] = record_g
                            self.closed[3][index2] = y_direction
                            self.closed[2][index2] = x_direction
                            self.closed = numpy.delete(self.closed, index2, axis=1)
                            self.open = numpy.c_[self.open, para]
                        continue

                    self.open = numpy.c_[self.open, para]  # 参数添加到open中
                    # print(self.open)

    def judge_location(self, m, list_co):
        """
        判断拓展点是否在open表或者closed表中
        :return:返回判断是否存在，和如果存在，那么存在的位置索引
        """
        jud = 0
        index = 0
        for i in range(list_co.shape[1]):

            if m[0] == list_co[0, i] and m[1] == list_co[1, i]:

                jud = jud + 1

                index = i
                break
            else:
                jud = jud
        # if a != 0:
        #     continue
        return jud, index

    def direction(self, father_point, son_point):
        """
        建立每一个节点的方向，便于在closed表中选出最佳路径
        非常重要的一步，不然画出的图像参考1.1版本
        x记录子节点和父节点的x轴变化
        y记录子节点和父节点的y轴变化
        如（0，1）表示子节点在父节点的方向上变化0和1
        :return:
        """
        x = son_point[0] - father_point[0]
        y = son_point[1] - father_point[1]
        return x, y

    def path_backtrace(self):
        """
        回溯closed表中的最短路径
        :return:
        """
        best_path = [self.goal[0], self.goal[1]]  # 回溯路径的初始化
        self.best_path_array = numpy.array([[self.goal[0]], [self.goal[1]]])
        j = 0
        while j <= self.closed.shape[1]:
            for i in range(self.closed.shape[1]):
                if best_path[0] == self.closed[0][i] and best_path[1] == self.closed[1][i]:
                    x = self.closed[0][i] - self.closed[2][i]
                    y = self.closed[1][i] - self.closed[3][i]
                    best_path = [x, y]
                    self.best_path_array = numpy.c_[self.best_path_array, best_path]
                    break  # 如果已经找到，退出本轮循环，减少耗时
                else:
                    continue
            j = j + 1
        return self.best_path_array

    def main(self):
        """
        main函数
        :return:
        """
        best = self.start  # 起点放入当前点，作为父节点
        h0 = self.h_value_tem(best)
        init_open = [best[0], best[1], 0, 0, 0, h0]  # 将方向初始化为（0，0），g_init=0,f值初始化h0
        # print('init_open:', init_open)
        self.open = numpy.column_stack((self.open, init_open))  # 起点放入open,open初始化
        path_as = []

        ite = 1  # 设置迭代次数小于200，防止程序出错无限循环
        while ite <= 20000:  # 1000
            # print('open_shape:', self.open.shape)
            # open列表为空，退出
            if self.open.shape[1] == 0:
                # print('No Astar path')
                return path_as

            self.open = self.open.T[numpy.lexsort(self.open)].T  # open表中最后一行排序(联合排序）

            # 选取open表中最小f值的节点作为best，放入closed表

            best = self.open[:, 0]
            # print('检验第%s次当前点坐标*******************' % ite)
            # print(best)
            self.closed = numpy.c_[self.closed, best]

            if best[0] == self.goal[0] and best[1] == self.goal[1]:  # 如果best是目标点，退出
                # print('Find Astar path')
                path_as = self.path_backtrace()
                return path_as
                # return
            # print('best_x:', best[0], 'best_y:', best[1])
            # print('goal_x:', self.goal[0], 'goal_y:', self.goal[1])

            # self.child_point(best)  # 生成子节点并判断数目
            if self.scene_round:
                self.child_point_round(best)  # 生成子节点并判断数目
            else:
                self.child_point(best)
            # print(self.open)
            self.open = numpy.delete(self.open, 0, axis=1)  # 删除open中最优点

            # print(self.open)

            ite = ite + 1

        # print('Astar loop time over 20000')
        return path_as


# class AStarPlanner:
#
#     def __init__(self, ox, oy, resolution, rr):
#         """
#         Initialize grid map for a star planning
#
#         ox: x position list of Obstacles [m]
#         oy: y position list of Obstacles [m]
#         resolution: grid resolution [m],地图的像素
#         rr: robot radius[m]
#         """
#
#         self.resolution = resolution
#         self.rr = rr
#         self.min_x, self.min_y = 0, 0
#         self.max_x, self.max_y = 0, 0
#         self.obstacle_map = None
#         self.x_width, self.y_width = 0, 0
#         self.motion = self.get_motion_model()
#         self.calc_obstacle_map(ox, oy)
#         # self.calc_guide_map(ox, oy)
#         # print('obs_map:', self.obstacle_map)
#
#     class Node:
#         """定义搜索区域节点类,每个Node都包含坐标x和y, 移动代价cost和父节点索引。
#         """
#         def __init__(self, x, y, cost, parent_index):
#             self.x = x  # index of grid
#             self.y = y  # index of grid
#             self.cost = cost
#             self.parent_index = parent_index
#
#         def __str__(self):
#             return str(self.x) + "," + str(self.y) + "," + str(
#                 self.cost) + "," + str(self.parent_index)
#
#     def planning(self, sx, sy, gx, gy):
#         """
#         A star path search
#         输入起始点和目标点的坐标(sx,sy)和(gx,gy)，
#         最终输出的结果是路径包含的点的坐标集合rx和ry。
#         input:
#             s_x: start x position [m]
#             s_y: start y position [m]
#             gx: goal x position [m]
#             gy: goal y position [m]
#
#         output:
#             rx: x position list of the final path
#             ry: y position list of the final path
#         """
#
#         start_node = self.Node(self.calc_xy_index(sx, self.min_x),
#                                self.calc_xy_index(sy, self.min_y), 0.0, -1)
#         goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
#                               self.calc_xy_index(gy, self.min_y), 0.0, -1)
#
#         open_set, closed_set = dict(), dict()
#         open_set[self.calc_grid_index(start_node)] = start_node
#
#         while 1:
#             if len(open_set) == 0:
#                 print("Open set is empty..")
#                 break
#
#             c_id = min(
#                 open_set,
#                 key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
#             current = open_set[c_id]
#
#             # show graph
#             if show_animation:  # pragma: no cover
#                 plt.plot(self.calc_grid_position(current.x, self.min_x),
#                          self.calc_grid_position(current.y, self.min_y), "xc")
#                 # for stopping simulation with the esc key.
#                 plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
#                 if len(closed_set.keys()) % 10 == 0:
#                     plt.pause(0.001)
#
#             # 通过追踪当前位置current.x和current.y来动态展示路径寻找
#             if current.x == goal_node.x and current.y == goal_node.y:
#                 print("Find goal")
#                 goal_node.parent_index = current.parent_index
#                 goal_node.cost = current.cost
#                 break
#
#             # Remove the item from the open set
#             del open_set[c_id]
#
#             # Add it to the closed set
#             closed_set[c_id] = current
#
#             # expand_grid search grid based on motion model
#             for i, _ in enumerate(self.motion):
#                 node = self.Node(current.x + self.motion[i][0],
#                                  current.y + self.motion[i][1],
#                                  current.cost + self.motion[i][2], c_id)
#                 n_id = self.calc_grid_index(node)
#
#                 # If the node is not safe, do nothing
#                 if not self.verify_node(node):
#                     continue
#
#                 if n_id in closed_set:
#                     continue
#
#                 if n_id not in open_set:
#                     open_set[n_id] = node  # discovered a new node
#                 else:
#                     if open_set[n_id].cost > node.cost:
#                         # This path is the best until now. record it
#                         open_set[n_id] = node
#
#         rx, ry = self.calc_final_path(goal_node, closed_set)
#
#         return rx, ry
#
#     def calc_final_path(self, goal_node, closed_set):
#         # generate final course
#         rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
#             self.calc_grid_position(goal_node.y, self.min_y)]
#         parent_index = goal_node.parent_index
#         while parent_index != -1:
#             n = closed_set[parent_index]
#             rx.append(self.calc_grid_position(n.x, self.min_x))
#             ry.append(self.calc_grid_position(n.y, self.min_y))
#             parent_index = n.parent_index
#
#         return rx, ry
#
#     @staticmethod
#     def calc_heuristic(n1, n2):
#         """计算启发函数
#
#         Args:
#             n1 (_type_): _description_
#             n2 (_type_): _description_
#
#         Returns:
#             _type_: _description_
#         """
#         w = 1.0  # weight of heuristic
#         d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
#         return d
#
#     def calc_grid_position(self, index, min_position):
#         """
#         calc grid position
#
#         :param index:
#         :param min_position:
#         :return:
#         """
#         pos = index * self.resolution + min_position
#         return pos
#
#     def calc_xy_index(self, position, min_pos):
#         return round((position - min_pos) / self.resolution)
#
#     def calc_grid_index(self, node):
#         return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)
#
#     def verify_node(self, node):
#         px = self.calc_grid_position(node.x, self.min_x)
#         py = self.calc_grid_position(node.y, self.min_y)
#
#         if px < self.min_x:
#             return False
#         elif py < self.min_y:
#             return False
#         elif px >= self.max_x:
#             return False
#         elif py >= self.max_y:
#             return False
#
#         # collision check
#         if self.obstacle_map[node.x][node.y]:
#             return False
#
#         return True
#
#     def calc_obstacle_map(self, ox, oy):
#
#         self.min_x = round(min(ox))
#         self.min_y = round(min(oy))
#         self.max_x = round(max(ox))
#         self.max_y = round(max(oy))
#         # print("min_x:", self.min_x)
#         # print("min_y:", self.min_y)
#         # print("max_x:", self.max_x)
#         # print("max_y:", self.max_y)
#
#         self.x_width = round((self.max_x - self.min_x) / self.resolution)
#         self.y_width = round((self.max_y - self.min_y) / self.resolution)
#         # print("x_width:", self.x_width)
#         # print("y_width:", self.y_width)
#
#         # obstacle map generation
#         self.obstacle_map = [[False for _ in range(self.y_width)]
#                              for _ in range(self.x_width)]
#         for ix in range(self.x_width):
#             x = self.calc_grid_position(ix, self.min_x)
#             for iy in range(self.y_width):
#                 y = self.calc_grid_position(iy, self.min_y)
#                 for iox, ioy in zip(ox, oy):
#                     d = math.hypot(iox - x, ioy - y)
#                     if d <= self.rr:
#                         self.obstacle_map[ix][iy] = True
#                         break
#
#     def calc_guide_map(self, ox, oy):
#
#         self.min_x = round(min(ox))
#         self.min_y = round(min(oy))
#         self.max_x = round(max(ox))
#         self.max_y = round(max(oy))
#         # print("min_x:", self.min_x)
#         # print("min_y:", self.min_y)
#         # print("max_x:", self.max_x)
#         # print("max_y:", self.max_y)
#
#         self.x_width = round((self.max_x - self.min_x) / self.resolution)
#         self.y_width = round((self.max_y - self.min_y) / self.resolution)
#         # print("x_width:", self.x_width)
#         # print("y_width:", self.y_width)
#
#         # obstacle map generation
#         self.obstacle_map = [[True for _ in range(self.y_width)]
#                              for _ in range(self.x_width)]
#         for ix in range(self.x_width):
#             x = self.calc_grid_position(ix, self.min_x)
#             for iy in range(self.y_width):
#                 y = self.calc_grid_position(iy, self.min_y)
#                 for iox, ioy in zip(ox, oy):
#                     d = math.hypot(iox - x, ioy - y)
#                     # if iox == ix:
#                     #     if ioy == iy:
#                     #         # print('obs_true')
#                     #         self.obstacle_map[ix][iy] = False
#                     # self.obstacle_map[ix][iy] = True
#                     if d <= self.rr:
#                         self.obstacle_map[ix][iy] = False
#                         break
#                     # else:
#                     #     self.obstacle_map[ix][iy] = True
#
#         # for iox, ioy in zip(ox, oy):
#         #     # d = math.hypot(iox - x, ioy - y)
#         #     self.obstacle_map[iox][ioy] = True
#         #     # if d <= self.rr:
#         #     #     self.obstacle_map[ix][iy] = True
#         #     #     break
#
#     @staticmethod
#     def get_motion_model():
#         # dx, dy, cost
#         motion = [[1, 0, 1],
#                   [0, 1, 1],
#                   [-1, 0, 1],
#                   [0, -1, 1],
#                   [-1, -1, math.sqrt(2)],
#                   [-1, 1, math.sqrt(2)],
#                   [1, -1, math.sqrt(2)],
#                   [1, 1, math.sqrt(2)]]
#
#         return motion
# class Node:
#     def __init__(self, x, y, cost, pind):
#         self.x = x  # x position of node
#         self.y = y  # y position of node
#         self.cost = cost  # g cost of node
#         self.pind = pind  # parent index of node
#
#
# class Para:
#     def __init__(self, minx, miny, maxx, maxy, xw, yw, reso, motion):
#         self.minx = minx
#         self.miny = miny
#         self.maxx = maxx
#         self.maxy = maxy
#         self.xw = xw
#         self.yw = yw
#         self.reso = reso  # resolution of grid world
#         self.motion = motion  # motion set
#
#
# def astar_planning(sx, sy, gx, gy, ox, oy, reso, rr):
#     """
#     return path of A*.
#     :param sx: starting node x [m]
#     :param sy: starting node y [m]
#     :param gx: goal node x [m]
#     :param gy: goal node y [m]
#     :param ox: obstacles x positions [m]
#     :param oy: obstacles y positions [m]
#     :param reso: xy grid resolution
#     :param rr: robot radius
#     :return: path
#     """
#     print('astar')
#     n_start = Node(round(sx / reso), round(sy / reso), 0.0, -1)
#     n_goal = Node(round(gx / reso), round(gy / reso), 0.0, -1)
#     # print('start ox')
#
#     ox = [x / reso for x in ox]
#     oy = [y / reso for y in oy]
#     print('start obsmap')
#     P, obsmap = calc_parameters(ox, oy, rr, reso)
#
#     open_set, closed_set = dict(), dict()
#     open_set[calc_index(n_start, P)] = n_start
#
#     q_priority = []
#     heapq.heappush(q_priority,
#                    (fvalue(n_start, n_goal), calc_index(n_start, P)))
#     print('start while')
#     while True:
#         if not open_set:
#             break
#         print('in the while')
#         _, ind = heapq.heappop(q_priority)
#         n_curr = open_set[ind]
#         closed_set[ind] = n_curr
#         open_set.pop(ind)
#
#         for i in range(len(P.motion)):
#             node = Node(n_curr.x + P.motion[i][0],
#                         n_curr.y + P.motion[i][1],
#                         n_curr.cost + u_cost(P.motion[i]), ind)
#             # print('in the loop')
#             if not check_node(node, P, obsmap):
#                 continue
#
#             n_ind = calc_index(node, P)
#             if n_ind not in closed_set:
#                 if n_ind in open_set:
#                     if open_set[n_ind].cost > node.cost:
#                         open_set[n_ind].cost = node.cost
#                         open_set[n_ind].pind = ind
#                 else:
#                     open_set[n_ind] = node
#                     heapq.heappush(q_priority,
#                                    (fvalue(node, n_goal), calc_index(node, P)))
#
#     pathx, pathy = extract_path(closed_set, n_start, n_goal, P)
#     print('find astar path')
#     return pathx, pathy
#
#
# def calc_holonomic_heuristic_with_obstacle(node, ox, oy, reso, rr):
#     n_goal = Node(round(node.x[-1] / reso), round(node.y[-1] / reso), 0.0, -1)
#
#     ox = [x / reso for x in ox]
#     oy = [y / reso for y in oy]
#
#     P, obsmap = calc_parameters(ox, oy, reso, rr)
#
#     open_set, closed_set = dict(), dict()
#     open_set[calc_index(n_goal, P)] = n_goal
#
#     q_priority = []
#     heapq.heappush(q_priority, (n_goal.cost, calc_index(n_goal, P)))
#
#     while True:
#         if not open_set:
#             break
#
#         _, ind = heapq.heappop(q_priority)
#         n_curr = open_set[ind]
#         closed_set[ind] = n_curr
#         open_set.pop(ind)
#
#         for i in range(len(P.motion)):
#             node = Node(n_curr.x + P.motion[i][0],
#                         n_curr.y + P.motion[i][1],
#                         n_curr.cost + u_cost(P.motion[i]), ind)
#
#             if not check_node(node, P, obsmap):
#                 continue
#
#             n_ind = calc_index(node, P)
#             if n_ind not in closed_set:
#                 if n_ind in open_set:
#                     if open_set[n_ind].cost > node.cost:
#                         open_set[n_ind].cost = node.cost
#                         open_set[n_ind].pind = ind
#                 else:
#                     open_set[n_ind] = node
#                     heapq.heappush(q_priority, (node.cost, calc_index(node, P)))
#
#     hmap = [[np.inf for _ in range(P.yw)] for _ in range(P.xw)]
#
#     for n in closed_set.values():
#         hmap[n.x - P.minx][n.y - P.miny] = n.cost
#
#     return hmap
#
#
# def check_node(node, P, obsmap):
#     if node.x <= P.minx or node.x >= P.maxx or \
#             node.y <= P.miny or node.y >= P.maxy:
#         return False
#
#     if obsmap[node.x - P.minx][node.y - P.miny]:
#         return False
#
#     return True
#
#
# def u_cost(u):
#     return math.hypot(u[0], u[1])
#
#
# def fvalue(node, n_goal):
#     return node.cost + h(node, n_goal)
#
#
# def h(node, n_goal):
#     return math.hypot(node.x - n_goal.x, node.y - n_goal.y)
#
#
# def calc_index(node, P):
#     return (node.y - P.miny) * P.xw + (node.x - P.minx)
#
#
# def calc_parameters(ox, oy, rr, reso):
#     minx, miny = round(min(ox)), round(min(oy))
#     maxx, maxy = round(max(ox)), round(max(oy))
#     xw, yw = maxx - minx, maxy - miny
#
#     motion = get_motion()
#     P = Para(minx, miny, maxx, maxy, xw, yw, reso, motion)
#     obsmap = calc_obsmap(ox, oy, rr, P)
#
#     return P, obsmap
#
#
# def calc_obsmap(ox, oy, rr, P):
#     obsmap = [[False for _ in range(P.yw)] for _ in range(P.xw)]
#
#     for x in range(P.xw):
#         xx = x + P.minx
#         for y in range(P.yw):
#             yy = y + P.miny
#             for oxx, oyy in zip(ox, oy):
#                 if math.hypot(oxx - xx, oyy - yy) <= rr / P.reso:
#                     obsmap[x][y] = True
#                     break
#
#     return obsmap
#
#
# def extract_path(closed_set, n_start, n_goal, P):
#     pathx, pathy = [n_goal.x], [n_goal.y]
#     n_ind = calc_index(n_goal, P)
#
#     while True:
#         node = closed_set[n_ind]
#         pathx.append(node.x)
#         pathy.append(node.y)
#         n_ind = node.pind
#
#         if node == n_start:
#             break
#
#     pathx = [x * P.reso for x in reversed(pathx)]
#     pathy = [y * P.reso for y in reversed(pathy)]
#
#     return pathx, pathy
#
#
# def get_motion():
#     motion = [[-1, 0], [-1, 1], [0, 1], [1, 1],
#               [1, 0], [1, -1], [0, -1], [-1, -1]]
#
#     return motion
#
#
# def get_env():
#     ox, oy = [], []
#
#     for i in range(60):
#         ox.append(i)
#         oy.append(0.0)
#     for i in range(60):
#         ox.append(60.0)
#         oy.append(i)
#     for i in range(61):
#         ox.append(i)
#         oy.append(60.0)
#     for i in range(61):
#         ox.append(0.0)
#         oy.append(i)
#     for i in range(40):
#         ox.append(20.0)
#         oy.append(i)
#     for i in range(40):
#         ox.append(40.0)
#         oy.append(60.0 - i)
#
#     return ox, oy
#
#
# def main():
#     sx = 10.0  # [m]
#     sy = 10.0  # [m]
#     gx = 50.0  # [m]
#     gy = 50.0  # [m]
#
#     robot_radius = 2.0
#     grid_resolution = 1.0
#     ox, oy = get_env()
#
#     pathx, pathy = astar_planning(sx, sy, gx, gy, ox, oy, grid_resolution, robot_radius)
#
#     plt.plot(ox, oy, 'sk')
#     plt.plot(pathx, pathy, '-r')
#     plt.plot(sx, sy, 'sg')
#     plt.plot(gx, gy, 'sb')
#     plt.axis("equal")
#     plt.show()
