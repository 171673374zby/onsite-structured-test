import numpy as np
from math import sin, cos, pi, sqrt

class VelocityPlanner:
    def __init__(self, time_gap, a_max, slow_speed, stop_line_buffer):
        self._time_gap         = time_gap
        self._a_max            = a_max
        self._slow_speed       = slow_speed
        self._stop_line_buffer = stop_line_buffer
        self._prev_trajectory  = [[0.0, 0.0, 0.0]]

    # Computes an open loop speed estimate based on the previously planned
    # trajectory, and the timestep since the last planning cycle.
    # Input: timestep is in seconds
    def get_open_loop_speed(self, timestep, ego_speed):
        if len(self._prev_trajectory) == 1:
            return ego_speed

        # If simulation time step is zero, give the start of the trajectory as the
        # open loop estimate.
        if timestep < 1e-4:
            return ego_speed

        for i in range(len(self._prev_trajectory)-1):
            distance_step = np.linalg.norm(np.subtract(self._prev_trajectory[i+1][0:2],
                                                       self._prev_trajectory[i][0:2]))
            velocity = self._prev_trajectory[i][2]
            time_delta = distance_step / (velocity + 1e-8)

            # If time_delta exceeds the remaining time in our simulation timestep,
            # interpolate between the velocity of the current step and the velocity
            # of the next step to estimate the open loop velocity.
            if time_delta > timestep:
                v1 = self._prev_trajectory[i][2]
                v2 = self._prev_trajectory[i+1][2]
                v_delta = v2 - v1
                interpolation_ratio = timestep / time_delta
                return v1 + interpolation_ratio * v_delta

            # Otherwise, keep checking.
            else:
                timestep -= time_delta

        # Simulation time step exceeded the length of the path, which means we have likely
        # stopped. Return the end velocity of the trajectory.
        return self._prev_trajectory[-1][2]


    def compute_velocity_profile(self, path, desired_speed, ego_state,
                                 closed_loop_speed, decelerate_to_stop,
                                 lead_car_state, follow_lead_vehicle):

        profile = []
        # For our profile, use the open loop speed as our initial speed.
        start_speed = ego_state[3]
        # Generate a trapezoidal profile to decelerate to stop.
        if decelerate_to_stop:
            profile = self.decelerate_profile(path, start_speed)

        # If we need to follow the lead vehicle, make sure we decelerate to its
        # speed by the time we reach the time gap point.
        elif follow_lead_vehicle:
            profile = self.follow_profile(path, start_speed, desired_speed,
                                          lead_car_state, ego_state)

        # Otherwise, compute the profile to reach our desired speed.
        else:
            profile = self.nominal_profile(path, start_speed, desired_speed)

        # Interpolate between the zeroth state and the first state.
        # This prevents the myopic controller from getting stuck at the zeroth
        # state.
        if len(profile) > 1:
            interpolated_state = [(profile[1][0] - profile[0][0]) * 0.1 + profile[0][0],
                                  (profile[1][1] - profile[0][1]) * 0.1 + profile[0][1],
                                  (profile[1][2] - profile[0][2]) * 0.1 + profile[0][2]]
            del profile[0]
            profile.insert(0, interpolated_state)

        # Save the planned profile for open loop speed estimation.
        self._prev_trajectory = profile

        return profile

    # Computes a trapezoidal profile for decelerating to stop.
    def decelerate_profile(self, path, start_speed):

        profile          = []
        slow_speed       = self._slow_speed
        stop_line_buffer = self._stop_line_buffer

        # Using d = (v_f^2 - v_i^2) / (2 * a), compute the two distances
        # used in the trapezoidal stop behaviour. decel_distance goes from
        #  start_speed to some coasting speed (slow_speed), then brake_distance
        #  goes from slow_speed to 0, both at a constant deceleration.
        decel_distance = calc_distance(start_speed, slow_speed, -self._a_max)
        brake_distance = calc_distance(slow_speed, 0, -self._a_max)

        # compute total path length
        path_length = 0.0
        for i in range(len(path[0])-1):
            path_length += np.linalg.norm([path[0][i+1] - path[0][i],
                                           path[1][i+1] - path[1][i]])

        stop_index = len(path[0]) - 1
        temp_dist = 0.0
        # Compute the index at which we should stop.
        while (stop_index > 0) and (temp_dist < stop_line_buffer):
            temp_dist += np.linalg.norm([path[0][stop_index] - path[0][stop_index-1],
                                         path[1][stop_index] - path[1][stop_index-1]])
            stop_index -= 1

        # If the brake distance exceeds the length of the path, then we cannot
        # perform a smooth deceleration and require a harder deceleration. Build
        # the path up in reverse to ensure we reach zero speed at the required
        # time.
        if brake_distance + decel_distance + stop_line_buffer > path_length:
            speeds = []
            vf = 0.0
            # The speeds past the stop line buffer should be zero.
            for i in reversed(range(stop_index, len(path[0]))):
                speeds.insert(0, 0.0)
            # The rest of the speeds should be a linear ramp from zero,
            # decelerating at -self._a_max.
            for i in reversed(range(stop_index)):
                dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                       path[1][i+1] - path[1][i]])
                vi = calc_final_speed(vf, -self._a_max, dist)
                # We don't want to have points above the starting speed
                # along our profile, so clamp to start_speed.
                if vi > start_speed:
                    vi = start_speed

                speeds.insert(0, vi)
                vf = vi

            # Generate the profile, given the computed speeds.
            for i in range(len(speeds)):
                profile.append([path[0][i], path[1][i], speeds[i]])

        # Otherwise, we will perform a full trapezoidal profile. The
        # brake_index will be the index of the path at which we start
        # braking, and the decel_index will be the index at which we stop
        # decelerating to our slow_speed. These two indices denote the
        # endpoints of the ramps in our trapezoidal profile.
        else:
            brake_index = stop_index
            temp_dist = 0.0
            # Compute the index at which to start braking down to zero.
            while (brake_index > 0) and (temp_dist < brake_distance):
                temp_dist += np.linalg.norm([path[0][brake_index] - path[0][brake_index-1],
                                             path[1][brake_index] - path[1][brake_index-1]])
                brake_index -= 1

            # Compute the index to stop decelerating to the slow speed.  This is
            # done by stepping through the points until accumulating
            # decel_distance of distance to said index, starting from the the
            # start of the path.
            decel_index = 0
            temp_dist = 0.0
            while (decel_index < brake_index) and (temp_dist < decel_distance):
                temp_dist += np.linalg.norm([path[0][decel_index+1] - path[0][decel_index],
                                             path[1][decel_index+1] - path[1][decel_index]])
                decel_index += 1

            # The speeds from the start to decel_index should be a linear ramp
            # from the current speed down to the slow_speed, decelerating at
            # -self._a_max.
            vi = start_speed
            for i in range(decel_index):
                dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                       path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                # We don't want to overshoot our slow_speed, so clamp it to that.
                if vf < slow_speed:
                    vf = slow_speed

                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            # In this portion of the profile, we are maintaining our slow_speed.
            for i in range(decel_index, brake_index):
                profile.append([path[0][i], path[1][i], vi])

            # The speeds from the brake_index to stop_index should be a
            # linear ramp from the slow_speed down to the 0, decelerating at
            # -self._a_max.
            for i in range(brake_index, stop_index):
                dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                       path[1][i+1] - path[1][i]])
                vf = calc_final_speed(vi, -self._a_max, dist)
                profile.append([path[0][i], path[1][i], vi])
                vi = vf

            # The rest of the profile consists of our stop_line_buffer, so
            # it contains zero speed for all points.
            for i in range(stop_index, len(path[0])):
                profile.append([path[0][i], path[1][i], 0.0])

        return profile

    # Computes a profile for following a lead vehicle..
    def follow_profile(self, path, start_speed, desired_speed, lead_car_state, ego_state):

        profile = []
        # Find the closest point to the lead vehicle on our planned path.
        min_index = len(path[0]) - 1
        min_dist = float('Inf')
        for i in range(len(path[0])):
            vec_x = path[0][i] - lead_car_state[0]
            vec_y = path[1][i] - lead_car_state[1]
            dist = np.linalg.norm([vec_x, vec_y])
            long = vec_x * cos(lead_car_state[2]) + vec_y * sin(lead_car_state[2])
            if dist < min_dist and dist > (lead_car_state[4] + ego_state[4]) / 2 and long < 0:
                min_dist = dist
                min_index = i

        # Compute the time gap point, assuming our velocity is held constant at
        # the minimum of the desired speed and the ego vehicle's velocity, from
        # the closest point to the lead vehicle on our planned path.
        # desired_speed = min(lead_car_state[3], desired_speed)
        ramp_end_index = min_index
        distance = min_dist
        distance_gap = desired_speed * self._time_gap
        while (ramp_end_index > 0) and (distance > distance_gap):
            distance += np.linalg.norm([path[0][ramp_end_index] - path[0][ramp_end_index-1],
                                        path[1][ramp_end_index] - path[1][ramp_end_index-1]])
            ramp_end_index -= 1

        # We now need to reach the ego vehicle's speed by the time we reach the
        # time gap point, ramp_end_index, which therefore is the end of our ramp
        # velocity profile.
        if desired_speed < start_speed:
            decel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            decel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        # Here we will compute the speed profile from our initial speed to the
        # end of the ramp.
        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                   path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
            else:
                vf = calc_final_speed(vi, self._a_max, dist)

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # Once we hit the time gap point, we need to be at the desired speed.
        # If we can't get there using a_max, do an abrupt change in the profile
        # to use the controller to decelerate more quickly.
        for i in range(ramp_end_index, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile

    # Computes a profile for nominal speed tracking.
    def nominal_profile(self, path, start_speed, desired_speed):

        profile = []
        # Compute distance travelled from start speed to desired speed using
        # a constant acceleration.
        if desired_speed < start_speed:
            accel_distance = calc_distance(start_speed, desired_speed, -self._a_max)
        else:
            accel_distance = calc_distance(start_speed, desired_speed, self._a_max)

        # Here we will compute the end of the ramp for our velocity profile.
        # At the end of the ramp, we will maintain our final speed.
        ramp_end_index = 0
        distance = 0.0
        while (ramp_end_index < len(path[0])-1) and (distance < accel_distance):
            distance += np.linalg.norm([path[0][ramp_end_index+1] - path[0][ramp_end_index],
                                        path[1][ramp_end_index+1] - path[1][ramp_end_index]])
            ramp_end_index += 1

        # Here we will actually compute the velocities along the ramp.
        vi = start_speed
        for i in range(ramp_end_index):
            dist = np.linalg.norm([path[0][i+1] - path[0][i],
                                   path[1][i+1] - path[1][i]])
            if desired_speed < start_speed:
                vf = calc_final_speed(vi, -self._a_max, dist)
                # clamp speed to desired speed
                if vf < desired_speed:
                    vf = desired_speed
            else:
                vf = calc_final_speed(vi, self._a_max, dist)
                # clamp speed to desired speed
                if vf > desired_speed:
                    vf = desired_speed

            profile.append([path[0][i], path[1][i], vi])
            vi = vf

        # If the ramp is over, then for the rest of the profile we should
        # track the desired speed.
        for i in range(ramp_end_index+1, len(path[0])):
            profile.append([path[0][i], path[1][i], desired_speed])

        return profile




def calc_distance(v_i, v_f, a):

    pass

    d = (v_f**2 - v_i**2) / (2 * a)
    return d

def calc_final_speed(v_i, a, d):
    # pass
    if (v_i**2 + 2*a*d) < 0:
        return 1e-5
    else:
        return sqrt(v_i**2 + 2*a*d)

def speed_gen(paths, start_speed, desired_speed, slow_speed, high_speed):

    path_speed=[]
    if desired_speed < slow_speed:
        desired_speed = slow_speed

    if desired_speed > high_speed:
        desired_speed = high_speed


    speed_gen = np.linspace(start_speed, desired_speed, 50)

    for i in range(0, len(paths[0])):
        path_speed.append([paths[0][i], paths[1][i], speed_gen[i]])

    return path_speed


