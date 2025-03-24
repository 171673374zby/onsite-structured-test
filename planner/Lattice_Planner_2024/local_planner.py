import numpy as np
import copy
import path_optimizer
import collision_checker
import velocity_planner
from math import sin, cos, pi


class LocalPlanner:
    def __init__(self, num_paths, path_offset, circle_offsets, circle_radii,
                 path_select_weight, time_gap, a_max, slow_speed,
                 stop_line_buffer):
        self._num_paths = num_paths
        self._path_offset = path_offset
        self._path_optimizer = path_optimizer.PathOptimizer()
        self._collision_checker = \
            collision_checker.CollisionChecker(circle_offsets,
                                               circle_radii,
                                               path_select_weight)
        self._velocity_planner = \
            velocity_planner.VelocityPlanner(time_gap, a_max, slow_speed,
                                             stop_line_buffer)
        self._prev_best_path = None

    def get_goal_state_set(self, goal_index, goal_state, waypoints, ego_state):

        if goal_index < len(waypoints)-1:
            delta_x = waypoints[goal_index+1][0] - waypoints[goal_index][0]
            delta_y = waypoints[goal_index+1][1] - waypoints[goal_index][1]
        else:
            delta_x = waypoints[goal_index][0] - waypoints[goal_index-1][0]
            delta_y = waypoints[goal_index][1] - waypoints[goal_index-1][1]
        heading = np.arctan2(delta_y, delta_x)

        goal_state_local = copy.copy(goal_state)

        goal_state_local[0] -= ego_state[0]
        goal_state_local[1] -= ego_state[1]

        theta = -ego_state[2]
        goal_x = cos(theta)*goal_state_local[0] - sin(theta)*goal_state_local[1]
        goal_y = sin(theta)*goal_state_local[0] + cos(theta)*goal_state_local[1]

        goal_t = heading - ego_state[2]

        # Velocity is preserved after the transformation.
        goal_v = goal_state[2]

        # Keep the goal heading within [-pi, pi] so the optimizer behaves well.
        if goal_t > pi:
            goal_t -= 2*pi
        elif goal_t < -pi:
            goal_t += 2*pi

        # Compute and apply the offset for each path such that
        # all of the paths have the same heading of the goal state,
        # but are laterally offset with respect to the goal heading.
        goal_state_set = []
        for i in range(self._num_paths):
            # Compute offsets that span the number of paths set for the local
            # planner. Each offset goal will be used to generate a potential
            # path to be considered by the local planner.
            offset = (i - self._num_paths // 2) * self._path_offset

            x_offset = offset*cos(goal_t + pi/2)
            y_offset = offset*sin(goal_t + pi/2)

            goal_state_set.append([goal_x + x_offset,
                                   goal_y + y_offset,
                                   goal_t,
                                   goal_v])

        return goal_state_set

    # Plans the path set using polynomial spiral optimization to
    # each of the goal states.
    def plan_paths(self, goal_state_set):

        paths         = []
        path_validity = []
        for goal_state in goal_state_set:
            path = self._path_optimizer.optimize_spiral(goal_state[0],
                                                        goal_state[1],
                                                        goal_state[2])
            if np.linalg.norm([path[0][-1] - goal_state[0],
                               path[1][-1] - goal_state[1],
                               path[2][-1] - goal_state[2]]) > 0.1:
                paths.append(path)
                path_validity.append(False)
            else:
                paths.append(path)
                path_validity.append(True)

        return paths, path_validity

def transform_paths(paths, ego_state):

    transformed_paths = []
    for path in paths:
        x_transformed = []
        y_transformed = []
        t_transformed = []

        for i in range(len(path[0])):
            x_transformed.append(ego_state[0] + path[0][i]*cos(ego_state[2]) - \
                                                path[1][i]*sin(ego_state[2]))
            y_transformed.append(ego_state[1] + path[0][i]*sin(ego_state[2]) + \
                                                path[1][i]*cos(ego_state[2]))
            t_transformed.append(path[2][i] + ego_state[2])

        transformed_paths.append([x_transformed, y_transformed, t_transformed])

    return transformed_paths
