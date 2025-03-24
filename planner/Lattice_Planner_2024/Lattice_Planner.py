#!/usr/bin/env python3

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA waypoint follower assessment client script.

A controller assessment to follow a given trajectory, where the trajectory
can be defined using way-points.

STARTING in a moment...
"""
from __future__ import print_function
from __future__ import division

# System level imports
import sys
import os
import argparse
import logging
import time
import math
import numpy as np
import csv
import controller2d
import local_planner
import behavioural_planner

# Script level imports
sys.path.append(os.path.abspath(sys.path[0] + '/..'))
# import live_plotter as lv   # Custom live plotting library
'''
from carla            import sensor
from carla.client     import make_carla_client, VehicleControl
from carla.settings   import CarlaSettings
from carla.tcp        import TCPConnectionError
from carla.controller import utils
'''
"""
Configurable params
"""
ITER_FOR_SIM_TIMESTEP  = 10     # no. iterations to compute approx sim timestep
WAIT_TIME_BEFORE_START = 1.00   # game seconds (time before controller start)
TOTAL_RUN_TIME         = 100.00 # game seconds (total runtime before sim end)
TOTAL_FRAME_BUFFER     = 300    # number of frames to buffer after total runtime
NUM_PEDESTRIANS        = 0      # total number of pedestrians to spawn
NUM_VEHICLES           = 2      # total number of vehicles to spawn
SEED_PEDESTRIANS       = 0      # seed for pedestrian spawn randomizer
SEED_VEHICLES          = 0      # seed for vehicle spawn randomizer
CLIENT_WAIT_TIME       = 3      # wait time for client before starting episode
                                # used to make sure the server loads
                                # consistently

WEATHERID = {
    "DEFAULT": 0,
    "CLEARNOON": 1,
    "CLOUDYNOON": 2,
    "WETNOON": 3,
    "WETCLOUDYNOON": 4,
    "MIDRAINYNOON": 5,
    "HARDRAINNOON": 6,
    "SOFTRAINNOON": 7,
    "CLEARSUNSET": 8,
    "CLOUDYSUNSET": 9,
    "WETSUNSET": 10,
    "WETCLOUDYSUNSET": 11,
    "MIDRAINSUNSET": 12,
    "HARDRAINSUNSET": 13,
    "SOFTRAINSUNSET": 14,
}
SIMWEATHER = WEATHERID["CLEARNOON"]     # set simulation weather

PLAYER_START_INDEX = 1      # spawn index for player (keep to 1)
FIGSIZE_X_INCHES   = 8      # x figure size of feedback in inches
FIGSIZE_Y_INCHES   = 8      # y figure size of feedback in inches
PLOT_LEFT          = 0.1    # in fractions of figure width and height
PLOT_BOT           = 0.1
PLOT_WIDTH         = 0.8
PLOT_HEIGHT        = 0.8

WAYPOINTS_FILENAME = 'waypoints.txt'  # waypoint file to load
DIST_THRESHOLD_TO_LAST_WAYPOINT = 2.0  # some distance from last position before
                                       # simulation ends

# Planning Constants
NUM_PATHS = 11
BP_LOOKAHEAD_BASE      = 15.0             # m
BP_LOOKAHEAD_TIME      = 1.5              # s
PATH_OFFSET            = 1.0              # m
CIRCLE_OFFSETS         = [-1.0, 1.0, 3.0] # m
CIRCLE_RADII           = [1.5, 1.5, 1.5]  # m
TIME_GAP               = 1.0              # s
PATH_SELECT_WEIGHT     = 10
A_MAX                  = 1.5              # m/s^2
SLOW_SPEED             = 2.0              # m/s
STOP_LINE_BUFFER       = 3.5              # m
LEAD_VEHICLE_LOOKAHEAD = 20.0             # m
LP_FREQUENCY_DIVISOR   = 2                # Frequency divisor to make the
                                          # local planner operate at a lower
                                          # frequency than the controller
                                          # (which operates at the simulation
                                          # frequency). Must be a natural
                                          # number.

# Course 4 specific parameters
C4_STOP_SIGN_FILE        = 'stop_sign_params.txt'
C4_STOP_SIGN_FENCELENGTH = 5        # m
C4_PARKED_CAR_FILE       = 'parked_vehicle_params.txt'

# Path interpolation parameters
INTERP_MAX_POINTS_PLOT    = 10   # number of points used for displaying
                                 # selected path
INTERP_DISTANCE_RES       = 0.01 # distance between interpolated points

# controller output directory
CONTROLLER_OUTPUT_FOLDER = os.path.dirname(os.path.realpath(__file__)) +\
                           '/controller_output/'
'''
def make_carla_settings(args):
    """Make a CarlaSettings object with the settings we need.
    """
    settings = CarlaSettings()

    # There is no need for non-agent info requests if there are no pedestrians
    # or vehicles.
    get_non_player_agents_info = False
    if (NUM_PEDESTRIANS > 0 or NUM_VEHICLES > 0):
        get_non_player_agents_info = True

    # Base level settings
    settings.set(
        SynchronousMode=True,
        SendNonPlayerAgentsInfo=get_non_player_agents_info,
        NumberOfVehicles=NUM_VEHICLES,
        NumberOfPedestrians=NUM_PEDESTRIANS,
        SeedVehicles=SEED_VEHICLES,
        SeedPedestrians=SEED_PEDESTRIANS,
        WeatherId=SIMWEATHER,
        QualityLevel=args.quality_level)
    return settings
'''

class Timer(object):

    def __init__(self, period):
        self.step = 0
        self._lap_step = 0
        self._lap_time = time.time()
        self._period_for_lap = period

    def tick(self):
        self.step += 1

    def has_exceeded_lap_period(self):
        if self.elapsed_seconds_since_lap() >= self._period_for_lap:
            return True
        else:
            return False

    def lap(self):
        self._lap_step = self.step
        self._lap_time = time.time()

    def ticks_per_second(self):
        return float(self.step - self._lap_step) /\
                     self.elapsed_seconds_since_lap()

    def elapsed_seconds_since_lap(self):
        return time.time() - self._lap_time

def get_current_pose(measurement):

    x   = measurement.player_measurements.transform.location.x
    y   = measurement.player_measurements.transform.location.y
    yaw = math.radians(measurement.player_measurements.transform.rotation.yaw)

    return (x, y, yaw)

def get_start_pos(scene):

    x = scene.player_start_spots[0].location.x
    y = scene.player_start_spots[0].location.y
    yaw = math.radians(scene.player_start_spots[0].rotation.yaw)

    return (x, y, yaw)

def get_player_collided_flag(measurement,
                             prev_collision_vehicles,
                             prev_collision_pedestrians,
                             prev_collision_other):

    player_meas = measurement.player_measurements
    current_collision_vehicles = player_meas.collision_vehicles
    current_collision_pedestrians = player_meas.collision_pedestrians
    current_collision_other = player_meas.collision_other

    collided_vehicles = current_collision_vehicles > prev_collision_vehicles
    collided_pedestrians = current_collision_pedestrians > \
                           prev_collision_pedestrians
    collided_other = current_collision_other > prev_collision_other

    return (collided_vehicles or collided_pedestrians or collided_other,
            current_collision_vehicles,
            current_collision_pedestrians,
            current_collision_other)

# 控制输出接口
def send_control_command(throttle, steer,brake,
                         hand_brake=False, reverse=False):

    # control = VehicleControl()

    # Clamp all values within their limits
    steer = np.fmax(np.fmin(steer, 1.0), -1.0)
    throttle = np.fmax(np.fmin(throttle, 1.0), 0)
    brake = np.fmax(np.fmin(brake, 1.0), 0)

    if throttle > 0:
        accelerate = throttle
    else:
        accelerate = - brake

def create_controller_output_dir(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def store_trajectory_plot(graph, fname):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)

    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, fname)
    graph.savefig(file_name)

def write_trajectory_file(x_list, y_list, v_list, t_list, collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'trajectory.txt')

    with open(file_name, 'w') as trajectory_file:
        for i in range(len(x_list)):
            trajectory_file.write('%3.3f, %3.3f, %2.3f, %6.3f %r\n' %\
                                  (x_list[i], y_list[i], v_list[i], t_list[i],
                                   collided_list[i]))

def write_collisioncount_file(collided_list):
    create_controller_output_dir(CONTROLLER_OUTPUT_FOLDER)
    file_name = os.path.join(CONTROLLER_OUTPUT_FOLDER, 'collision_count.txt')

    with open(file_name, 'w') as collision_file:
        collision_file.write(str(sum(collided_list)))

def exec_waypoint_nav_demo(args):

    # print('Carla client connected.')

    # settings = make_carla_settings(args)

    # Now we load these settings into the server. The server replies
    # with a scene description containing the available start spots for
    # the player. Here we can provide a CarlaSettings object or a
    # CarlaSettings.ini file as string.
    # scene = client.load_settings(settings)

    # Refer to the player start folder in the WorldOutliner to see the
    # player start information
    player_start = PLAYER_START_INDEX

    # client.start_episode(player_start)

    time.sleep(CLIENT_WAIT_TIME)

    # Notify the server that we want to start the episode at the
    # player_start index. This function blocks until the server is ready
    # to start the episode.
    # print('Starting new episode at %r...' % scene.map_name)
    # client.start_episode(player_start)

    #############################################
    # Load Configurations
    #############################################

    # Load configuration file (options.cfg) and then parses for the various
    # options. Here we have two main options:
    # live_plotting and live_plotting_period, which controls whether
    # live plotting is enabled or how often the live plotter updates
    # during the simulation run.
    '''
    config = configparser.ConfigParser()
    config.read(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'options.cfg'))
    demo_opt = config['Demo Parameters']
    '''

    # Get options
    '''
    enable_live_plot = demo_opt.get('live_plotting', 'true').capitalize()
    enable_live_plot = enable_live_plot == 'True'
    live_plot_period = float(demo_opt.get('live_plotting_period', 0))
    

    # Set options
    live_plot_timer = Timer(live_plot_period)
    '''


    #############################################
    # Load stop sign and parked vehicle parameters
    # Convert to input params for LP
    #############################################
    '''
    # Stop sign (X(m), Y(m), Z(m), Yaw(deg))
    stopsign_data = None
    stopsign_fences = []  # [x0, y0, x1, y1]
    with open(C4_STOP_SIGN_FILE, 'r') as stopsign_file:
        next(stopsign_file)  # skip header
        stopsign_reader = csv.reader(stopsign_file,
                                     delimiter=',',
                                     quoting=csv.QUOTE_NONNUMERIC)
        stopsign_data = list(stopsign_reader)
        # convert to rad
        for i in range(len(stopsign_data)):
            stopsign_data[i][3] = stopsign_data[i][3] * np.pi / 180.0

    # obtain stop sign fence points for LP
    for i in range(len(stopsign_data)):
        x = stopsign_data[i][0]
        y = stopsign_data[i][1]
        z = stopsign_data[i][2]
        yaw = stopsign_data[i][3] + np.pi / 2.0  # add 90 degrees for fence
        spos = np.array([
            [0, 0],
            [0, C4_STOP_SIGN_FENCELENGTH]])
        rotyaw = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)]])
        spos_shift = np.array([
            [x, x],
            [y, y]])
        spos = np.add(np.matmul(rotyaw, spos), spos_shift)
        stopsign_fences.append([spos[0, 0], spos[1, 0], spos[0, 1], spos[1, 1]])
    '''


    # Parked car(s) (X(m), Y(m), Z(m), Yaw(deg), RADX(m), RADY(m), RADZ(m))
    # 进行速度判断，若小于当前车速按照障碍车绕行

    if 1:#obs.v < self.v: #与onsite接口对接

        parkedcar_box_pts = []  # [x,y]
        parkcaridx=1 # “障碍物车辆”计数

        for i in parkcaridx:
            x = 0  # 位置信息
            y = 1
            yaw = 2
            xrad = 2.5 # 0.5*车长
            yrad = 1.0 # 0.5*车宽

            cpos = np.array([
                [-xrad, -xrad, -xrad, 0, xrad, xrad, xrad, 0],
                [-yrad, 0, yrad, yrad, yrad, 0, -yrad, -yrad]])
            rotyaw = np.array([
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)]])
            cpos_shift = np.array([
                [x, x, x, x, x, x, x, x],
                [y, y, y, y, y, y, y, y]])
            cpos = np.add(np.matmul(rotyaw, cpos), cpos_shift)
        for j in range(cpos.shape[1]):
                parkedcar_box_pts.append([cpos[0, j], cpos[1, j]])

    #############################################
    # Load Waypoints
    #############################################
    # Opens the waypoint file and stores it to "waypoints"
    waypoints_file = WAYPOINTS_FILENAME
    waypoints_filepath = \
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     WAYPOINTS_FILENAME)
    waypoints_np = None
    with open(waypoints_filepath) as waypoints_file_handle:
        waypoints = list(csv.reader(waypoints_file_handle,
                                    delimiter=',',
                                    quoting=csv.QUOTE_NONNUMERIC))
        waypoints_np = np.array(waypoints)

    #############################################
    # Controller 2D Class Declaration
    #############################################
    # This is where we take the controller2d.py class
    # and apply it to the simulator
    controller = controller2d.Controller2D(waypoints)

    #############################################
    # Determine simulation average timestep (and total frames)
    #############################################
    # Ensure at least one frame is used to compute average timestep
    num_iterations = ITER_FOR_SIM_TIMESTEP
    if (ITER_FOR_SIM_TIMESTEP < 1):
        num_iterations = 1

    # Gather current data from the CARLA server. This is used to get the
    # simulator starting game time. Note that we also need to
    # send a command back to the CARLA server because synchronous mode
    # is enabled.

    '''
    measurement_data, sensor_data = client.read_data()
    sim_start_stamp = measurement_data.game_timestamp / 1000.0
    # Send a control command to proceed to next iteration.
    # This mainly applies for simulations that are in synchronous mode.
    send_control_command(client, throttle=0.0, steer=0, brake=1.0)
    # Computes the average timestep based on several initial iterations
    sim_duration = 0
    for i in range(num_iterations):
        # Gather current data
        measurement_data, sensor_data = client.read_data()
        # Send a control command to proceed to next iteration
        send_control_command(client, throttle=0.0, steer=0, brake=1.0)
        # Last stamp
        if i == num_iterations - 1:
            sim_duration = measurement_data.game_timestamp / 1000.0 - \
                           sim_start_stamp
    '''

    # Outputs average simulation timestep and computes how many frames
    # will elapse before the simulation should end based on various
    # parameters that we set in the beginning.

    '''
    SIMULATION_TIME_STEP = sim_duration / float(num_iterations)
    print("SERVER SIMULATION STEP APPROXIMATION: " + \
          str(SIMULATION_TIME_STEP))
    TOTAL_EPISODE_FRAMES = int((TOTAL_RUN_TIME + WAIT_TIME_BEFORE_START) / \
                               SIMULATION_TIME_STEP) + TOTAL_FRAME_BUFFER

    #############################################
    # Frame-by-Frame Iteration and Initialization
    #############################################
    # Store pose history starting from the start position
    measurement_data, sensor_data = client.read_data()
    start_timestamp = measurement_data.game_timestamp / 1000.0
    '''
    # 起点参数获取
    start_x, start_y, start_yaw = 0
    send_control_command(throttle=0.0, steer=0, brake=0)
    x_history = [start_x]
    y_history = [start_y]
    yaw_history = [start_yaw]
    time_history = [0]
    speed_history = [0]
    collided_flag_history = [False]  # assume player starts off non-collided


    #############################################
    # Vehicle Trajectory Live Plotting Setup
    #############################################
    # Uses the live plotter to generate live feedback during the simulation
    # The two feedback includes the trajectory feedback and
    # the controller feedback (which includes the speed tracking).
    # lp_traj = lv.LivePlotter(tk_title="Trajectory Trace")
    # lp_1d = lv.LivePlotter(tk_title="Controls Feedback")

    ###
    # Add 2D position / trajectory plot
    ###
    '''
    trajectory_fig = lp_traj.plot_new_dynamic_2d_figure(
            title='Vehicle Trajectory',
            figsize=(FIGSIZE_X_INCHES, FIGSIZE_Y_INCHES),
            edgecolor="black",
            rect=[PLOT_LEFT, PLOT_BOT, PLOT_WIDTH, PLOT_HEIGHT])

    trajectory_fig.set_invert_x_axis() # Because UE4 uses left-handed
                                       # coordinate system the X
                                       # axis in the graph is flipped
    trajectory_fig.set_axis_equal()    # X-Y spacing should be equal in size

    # Add waypoint markers
    trajectory_fig.add_graph("waypoints", window_size=waypoints_np.shape[0],
                             x0=waypoints_np[:,0], y0=waypoints_np[:,1],
                             linestyle="-", marker="", color='g')
    # Add trajectory markers
    trajectory_fig.add_graph("trajectory", window_size=TOTAL_EPISODE_FRAMES,
                             x0=[start_x]*TOTAL_EPISODE_FRAMES,
                             y0=[start_y]*TOTAL_EPISODE_FRAMES,
                             color=[1, 0.5, 0])
    # Add starting position marker
    trajectory_fig.add_graph("start_pos", window_size=1,
                             x0=[start_x], y0=[start_y],
                             marker=11, color=[1, 0.5, 0],
                             markertext="Start", marker_text_offset=1)
    # Add end position marker
    trajectory_fig.add_graph("end_pos", window_size=1,
                             x0=[waypoints_np[-1, 0]],
                             y0=[waypoints_np[-1, 1]],
                             marker="D", color='r',
                             markertext="End", marker_text_offset=1)
    # Add car marker
    trajectory_fig.add_graph("car", window_size=1,
                             marker="s", color='b', markertext="Car",
                             marker_text_offset=1)
    # Add lead car information
    trajectory_fig.add_graph("leadcar", window_size=1,
                             marker="s", color='g', markertext="Lead Car",
                             marker_text_offset=1)
    # Add stop sign position
    trajectory_fig.add_graph("stopsign", window_size=1,
                             x0=[stopsign_fences[0][0]], y0=[stopsign_fences[0][1]],
                             marker="H", color="r",
                             markertext="Stop Sign", marker_text_offset=1)
    # Add stop sign "stop line"
    trajectory_fig.add_graph("stopsign_fence", window_size=1,
                             x0=[stopsign_fences[0][0], stopsign_fences[0][2]],
                             y0=[stopsign_fences[0][1], stopsign_fences[0][3]],
                             color="r")

    # Load parked car points
    parkedcar_box_pts_np = np.array(parkedcar_box_pts)
    trajectory_fig.add_graph("parkedcar_pts", window_size=parkedcar_box_pts_np.shape[0],
                             x0=parkedcar_box_pts_np[:,0], y0=parkedcar_box_pts_np[:,1],
                             linestyle="", marker="+", color='b')

    # Add lookahead path
    trajectory_fig.add_graph("selected_path",
                             window_size=INTERP_MAX_POINTS_PLOT,
                             x0=[start_x]*INTERP_MAX_POINTS_PLOT,
                             y0=[start_y]*INTERP_MAX_POINTS_PLOT,
                             color=[1, 0.5, 0.0],
                             linewidth=3)

    # Add local path proposals
    for i in range(NUM_PATHS):
        trajectory_fig.add_graph("local_path " + str(i), window_size=200,
                                 x0=None, y0=None, color=[0.0, 0.0, 1.0])

    ###
    # Add 1D speed profile updater
    ###
    forward_speed_fig =\
            lp_1d.plot_new_dynamic_figure(title="Forward Speed (m/s)")
    forward_speed_fig.add_graph("forward_speed",
                                label="forward_speed",
                                window_size=TOTAL_EPISODE_FRAMES)
    forward_speed_fig.add_graph("reference_signal",
                                label="reference_Signal",
                                window_size=TOTAL_EPISODE_FRAMES)

    # Add throttle signals graph
    throttle_fig = lp_1d.plot_new_dynamic_figure(title="Throttle")
    throttle_fig.add_graph("throttle",
                          label="throttle",
                          window_size=TOTAL_EPISODE_FRAMES)
    # Add brake signals graph
    brake_fig = lp_1d.plot_new_dynamic_figure(title="Brake")
    brake_fig.add_graph("brake",
                          label="brake",
                          window_size=TOTAL_EPISODE_FRAMES)
    # Add steering signals graph
    steer_fig = lp_1d.plot_new_dynamic_figure(title="Steer")
    steer_fig.add_graph("steer",
                          label="steer",
                          window_size=TOTAL_EPISODE_FRAMES)

    # live plotter is disabled, hide windows
    if not enable_live_plot:
        lp_traj._root.withdraw()
        lp_1d._root.withdraw()
    '''

    #############################################
    # Local Planner Variables
    #############################################
    wp_goal_index = 0
    local_waypoints = None
    path_validity = np.zeros((NUM_PATHS, 1), dtype=bool)
    lp = local_planner.LocalPlanner(NUM_PATHS,
                                    PATH_OFFSET,
                                    CIRCLE_OFFSETS,
                                    CIRCLE_RADII,
                                    PATH_SELECT_WEIGHT,
                                    TIME_GAP,
                                    A_MAX,
                                    SLOW_SPEED,
                                    STOP_LINE_BUFFER)
    bp = behavioural_planner.BehaviouralPlanner(BP_LOOKAHEAD_BASE, LEAD_VEHICLE_LOOKAHEAD)

    #############################################
    # Scenario Execution Loop
    #############################################

    # Iterate the frames until the end of the waypoints is reached or
    # the TOTAL_EPISODE_FRAMES is reached. The controller simulation then
    # ouptuts the results to the controller output directory.
    reached_the_end = False
    skip_first_frame = True

    # Initialize the current timestamp.
    # current_timestamp = start_timestamp

    # Initialize collision history
    prev_collision_vehicles = 0
    prev_collision_pedestrians = 0
    prev_collision_other = 0

    ''''''
    for frame in range(100):
        # Gather current data from the CARLA server
        # measurement_data, sensor_data = client.read_data()

        # Update pose and timestamp
        prev_timestamp = current_timestamp

        # 起点参数获取

        current_x = 0
        current_y = 0
        current_yaw = 0
        current_speed = 0

        # 时间
        current_timestamp = 0

        x_history.append(current_x)
        y_history.append(current_y)
        yaw_history.append(current_yaw)
        speed_history.append(current_speed)
        time_history.append(current_timestamp)

        # Store collision history
        '''
        collided_flag, prev_collision_vehicles, prev_collision_pedestrians, prev_collision_other = \
            get_player_collided_flag(measurement_data,prev_collision_vehicles,prev_collision_pedestrians,prev_collision_other)
        collided_flag_history.append(collided_flag)
        '''
        # Obtain Lead Vehicle information.
        lead_car_pos = []
        lead_car_length = []
        lead_car_speed = []
        # for agent in measurement_data.non_player_agents:
            # agent_id = agent.id
            # if agent.HasField('vehicle'):

        # 获取跟随车辆信息
        if 1: #obs.v>=self.v # 若前车速度小于自车，跟随
            lead_car_pos.append([x, y])
            lead_car_length.append(5)
            lead_car_speed.append(5)

            if frame % LP_FREQUENCY_DIVISOR == 0:

                open_loop_speed = lp._velocity_planner.get_open_loop_speed(current_timestamp - prev_timestamp)

                ego_state = [current_x, current_y, current_yaw, open_loop_speed]

                # Set lookahead based on current speed.
                bp.set_lookahead(BP_LOOKAHEAD_BASE + BP_LOOKAHEAD_TIME * open_loop_speed)

                # Perform a state transition in the behavioural planner.
                bp.transition_state(waypoints, ego_state, current_speed)

                # Check to see if we need to follow the lead vehicle.
                bp.check_for_lead_vehicle(ego_state, lead_car_pos[1])

                # Compute the goal state set from the behavioural planner's computed goal state.
                goal_state_set = lp.get_goal_state_set(bp._goal_index, bp._goal_state, waypoints, ego_state)

                # Calculate planned paths in the local frame.
                paths, path_validity = lp.plan_paths(goal_state_set)

                # Transform those paths back to the global frame.
                paths = local_planner.transform_paths(paths, ego_state)

                # Perform collision checking.
                collision_check_array = lp._collision_checker.collision_check(paths, [parkedcar_box_pts])

                # Compute the best local path.
                best_index = lp._collision_checker.select_best_path_index(paths, collision_check_array, bp._goal_state)
                # If no path was feasible, continue to follow the previous best path.
                if best_index == None:
                    best_path = lp._prev_best_path
                else:
                    best_path = paths[best_index]
                    lp._prev_best_path = best_path

                # Compute the velocity profile for the path, and compute the waypoints.
                # Use the lead vehicle to inform the velocity profile's dynamic obstacle handling.
                # In this scenario, the only dynamic obstacle is the lead vehicle at index 1.
                desired_speed = bp._goal_state[2]
                lead_car_state = [lead_car_pos[1][0], lead_car_pos[1][1], lead_car_speed[1]]
                decelerate_to_stop = bp._state == behavioural_planner.DECELERATE_TO_STOP
                local_waypoints = lp._velocity_planner.compute_velocity_profile(best_path, desired_speed, ego_state,
                                                                                current_speed, decelerate_to_stop,
                                                                                lead_car_state, bp._follow_lead_vehicle)
                # --------------------------------------------------------------

                if local_waypoints != None:
                    # Update the controller waypoint path with the best local path.
                    # This controller is similar to that developed in Course 1 of this
                    # specialization.  Linear interpolation computation on the waypoints
                    # is also used to ensure a fine resolution between points.
                    wp_distance = []  # distance array
                    local_waypoints_np = np.array(local_waypoints)
                    for i in range(1, local_waypoints_np.shape[0]):
                        wp_distance.append(
                            np.sqrt((local_waypoints_np[i, 0] - local_waypoints_np[i - 1, 0]) ** 2 +
                                    (local_waypoints_np[i, 1] - local_waypoints_np[i - 1, 1]) ** 2))
                    wp_distance.append(0)  # last distance is 0 because it is the distance
                    # from the last waypoint to the last waypoint

                    # Linearly interpolate between waypoints and store in a list
                    wp_interp = []  # interpolated values
                    # (rows = waypoints, columns = [x, y, v])
                    for i in range(local_waypoints_np.shape[0] - 1):
                        # Add original waypoint to interpolated waypoints list (and append
                        # it to the hash table)
                        wp_interp.append(list(local_waypoints_np[i]))

                        # Interpolate to the next waypoint. First compute the number of
                        # points to interpolate based on the desired resolution and
                        # incrementally add interpolated points until the next waypoint
                        # is about to be reached.
                        num_pts_to_interp = int(np.floor(wp_distance[i] / \
                                                         float(INTERP_DISTANCE_RES)) - 1)
                        wp_vector = local_waypoints_np[i + 1] - local_waypoints_np[i]
                        wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                        for j in range(num_pts_to_interp):
                            next_wp_vector = INTERP_DISTANCE_RES * float(j + 1) * wp_uvector
                            wp_interp.append(list(local_waypoints_np[i] + next_wp_vector))
                    # add last waypoint at the end
                    wp_interp.append(list(local_waypoints_np[-1]))

                    # Update the other controller values and controls
                    controller.update_waypoints(wp_interp)
                    pass

        ###
        # Controller Update
        ###
        if local_waypoints != None and local_waypoints != []:
            controller.update_values(current_x, current_y, current_yaw,
                                     current_speed,
                                     current_timestamp, frame)
            controller.update_controls()
            cmd_throttle, cmd_steer, cmd_brake = controller.get_commands()
            # 尝试输出控制值
            send_control_command(throttle=cmd_throttle, steer=cmd_steer, brake=cmd_brake)
        else:
            cmd_throttle = 0.0
            cmd_steer = 0.0
            cmd_brake = 0.0

        # Skip the first frame or if there exists no local paths
        if skip_first_frame and frame == 0:
            pass
        elif local_waypoints == None:
            pass
        '''
        else:
            # Update live plotter with new feedback
            trajectory_fig.roll("trajectory", current_x, current_y)
            trajectory_fig.roll("car", current_x, current_y)
            if lead_car_pos:    # If there exists a lead car, plot it
                trajectory_fig.roll("leadcar", lead_car_pos[1][0],
                                    lead_car_pos[1][1])
            forward_speed_fig.roll("forward_speed",
                                   current_timestamp,
                                   current_speed)
            forward_speed_fig.roll("reference_signal",
                                   current_timestamp,
                                   controller._desired_speed)
            throttle_fig.roll("throttle", current_timestamp, cmd_throttle)
            brake_fig.roll("brake", current_timestamp, cmd_brake)
            steer_fig.roll("steer", current_timestamp, cmd_steer)

            # Local path plotter update
            if frame % LP_FREQUENCY_DIVISOR == 0:
                path_counter = 0
                for i in range(NUM_PATHS):
                    # If a path was invalid in the set, there is no path to plot.
                    if path_validity[i]:
                        # Colour paths according to collision checking.
                        if not collision_check_array[path_counter]:
                            colour = 'r'
                        elif i == best_index:
                            colour = 'k'
                        else:
                            colour = 'b'
                        trajectory_fig.update("local_path " + str(i), paths[path_counter][0], paths[path_counter][1], colour)
                        path_counter += 1
                    else:
                        trajectory_fig.update("local_path " + str(i), [ego_state[0]], [ego_state[1]], 'r')
            # When plotting lookahead path, only plot a number of points
            # (INTERP_MAX_POINTS_PLOT amount of points). This is meant
            # to decrease load when live plotting
            wp_interp_np = np.array(wp_interp)
            path_indices = np.floor(np.linspace(0,
                                                wp_interp_np.shape[0]-1,
                                                INTERP_MAX_POINTS_PLOT))
            trajectory_fig.update("selected_path",
                    wp_interp_np[path_indices.astype(int), 0],
                    wp_interp_np[path_indices.astype(int), 1],
                    new_colour=[1, 0.5, 0.0])


            # Refresh the live plot based on the refresh rate
            # set by the options
            if enable_live_plot and \
               live_plot_timer.has_exceeded_lap_period():
                lp_traj.refresh()
                lp_1d.refresh()
                live_plot_timer.lap()
        '''

        # Output controller command to CARLA server
        '''
        send_control_command(client,
                             throttle=cmd_throttle,
                             steer=cmd_steer,
                             brake=cmd_brake)
        '''

        # Find if reached the end of waypoint. If the car is within
        # DIST_THRESHOLD_TO_LAST_WAYPOINT to the last waypoint,
        # the simulation will end.
        dist_to_last_waypoint = np.linalg.norm(np.array([
            waypoints[-1][0] - current_x,
            waypoints[-1][1] - current_y]))
        if dist_to_last_waypoint < DIST_THRESHOLD_TO_LAST_WAYPOINT:
            reached_the_end = True
        if reached_the_end:
            break

    # End of demo - Stop vehicle and Store outputs to the controller output
    # directory.
    if reached_the_end:
        print("Reached the end of path. Writing to controller_output...")
    else:
        print("Exceeded assessment time. Writing to controller_output...")
    # Stop the car
    # send_control_command(client, throttle=0.0, steer=0.0, brake=1.0)
    # Store the various outputs
    '''
    store_trajectory_plot(trajectory_fig.fig, 'trajectory.png')
    store_trajectory_plot(forward_speed_fig.fig, 'forward_speed.png')
    store_trajectory_plot(throttle_fig.fig, 'throttle_output.png')
    store_trajectory_plot(brake_fig.fig, 'brake_output.png')
    store_trajectory_plot(steer_fig.fig, 'steer_output.png')
    write_trajectory_file(x_history, y_history, speed_history, time_history,
                          collided_flag_history)
    write_collisioncount_file(collided_flag_history)
    '''


def main():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Low',
        help='graphics quality level.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    args = argparser.parse_args()

    # Logging startup info
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    args.out_filename_format = '_out/episode_{:0>4d}/{:s}/{:0>6d}'

    # Execute when server connection is established
    while True:
        # try:
            exec_waypoint_nav_demo(args)
            print('Done.')
            return
'''
        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
'''
if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

