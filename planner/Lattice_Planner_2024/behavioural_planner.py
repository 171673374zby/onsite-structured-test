import numpy as np
import math

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# Number of cycles before moving from stop sign.
STOP_COUNTS = 10

class BehaviouralPlanner:
    def __init__(self, lookahead, stopsign_fences):#  stopsign_fences,
        self._lookahead                     = lookahead
        self._stopsign_fences               = stopsign_fences
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_index                    = 0
        self._stop_count                    = 0
        self._stopsign_index                = 0

    def set_lookahead(self, lookahead):
        self._lookahead = lookahead


    def transition_state(self, waypoints, ego_state, closed_loop_speed):

        if self._state == FOLLOW_LANE:

            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

            # goal_index, stop_sign_found = self.check_for_stop_signs(waypoints, closest_index, goal_index)
            self._goal_index = goal_index
            self._goal_state = waypoints[self._goal_index]

            # if stop_sign_found:
            #     self._stopsign_index = goal_index
                # self._goal_state[2] = 0.0
                # self._state = DECELERATE_TO_STOP
                # print("Switch state to DECELERATE TO STOP")

        elif self._state == DECELERATE_TO_STOP:

            if abs(closed_loop_speed) < STOP_THRESHOLD:
                self._state = STAY_STOPPED
                self._stop_count = 0
                print("Switch state to STAY STOPPED")

        elif self._state == STAY_STOPPED:

            if self._stop_count == STOP_COUNTS:

                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)

                stop_sign_found = False
                self._goal_index = goal_index
                self._goal_state = waypoints[self._goal_index]

                if not stop_sign_found:
                    self._state = FOLLOW_LANE
                    print("Switch state to FOLLOW LANE")

            else:
                self._stop_count += 1

        else:
            raise ValueError('Invalid state value.')


    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):

        arc_length = closest_len
        wp_index = closest_index


        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        while wp_index < len(waypoints) - 1:
            delta_x = waypoints[wp_index][0] - waypoints[wp_index+1][0]
            delta_y = waypoints[wp_index][1] - waypoints[wp_index+1][1]
            arc_length += np.sqrt( delta_x**2 + delta_y**2 )
            if arc_length > self._lookahead:
                return wp_index
            wp_index += 1


        return wp_index

    # Checks the given segment of the waypoint list to see if it
    # intersects with a stop line. If any index does, return the
    # new goal state accordingly.
    def check_for_stop_signs(self, waypoints, closest_index, goal_index):

        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the stop lines.
            intersect_flag = False
            for stopsign_fence in self._stopsign_fences:
                wp_1   = np.array(waypoints[i][0:2])
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(stopsign_fence[0:2])
                s_2    = np.array(stopsign_fence[2:4])

                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a stop line, update
                # the goal state to stop before the goal line.
                if intersect_flag and (i != self._stopsign_index):
                    goal_index = i
                    return goal_index, True

        return goal_index, False

    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position, LEAD_VEHICLE_LOOKAHEAD):

        if not self._follow_lead_vehicle:
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is too far away.
            if lead_car_distance > LEAD_VEHICLE_LOOKAHEAD:
                return

            lead_car_delta_vector = np.divide(lead_car_delta_vector,
                                              lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]),
                                  math.sin(ego_state[2])]
            # Check to see if the relative angle between the lead vehicle and the ego
            # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
            if np.dot(lead_car_delta_vector,
                      ego_heading_vector) < (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = True

        else:
            lead_car_delta_vector = [lead_car_position[0] - ego_state[0],
                                     lead_car_position[1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)

            # Add a 15m buffer to prevent oscillations for the distance check.
            if lead_car_distance < LEAD_VEHICLE_LOOKAHEAD + 15:
                return
            # Check to see if the lead vehicle is still within the ego vehicle's
            # frame of view.
            lead_car_delta_vector = np.divide(lead_car_delta_vector, lead_car_distance)
            ego_heading_vector = [math.cos(ego_state[2]), math.sin(ego_state[2])]
            if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)):
                return

            self._follow_lead_vehicle = False
def get_closest_index(waypoints, ego_state):

    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        distance = np.sqrt( (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2 )
        if distance <= closest_len:
            closest_len = distance
            closest_index = i


    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False

