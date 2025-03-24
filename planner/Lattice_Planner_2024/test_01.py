from Astar_planner import astar_planning
from Astar_planner import AStarPlanner

# path_x, path_y = astar_planning(1, 1, 9, 10, [5, 6], [5, 6], 1, 2)
path_x=[]
path_y=[]
ox = []
oy = []
sx = 10.0  # [m]
sy = 10.0  # [m]
gx = 50.0  # [m]
gy = 50.0  # [m]
grid_size = 2.0  # [m]
robot_radius = 1.0  # [m]
#
# ox.append(10.0)
# oy.append(60.0)
#
# ox.append(20.0)
# oy.append(60.0)
ox, oy = [], []
for i in range(-10, 60):
    ox.append(i)
    oy.append(-10.0)

# Astar = AStarPlanner(ox, oy, grid_size, robot_radius)
# path_x, path_y = Astar.planning(sx, sy, gx, gy)
# path_x, path_y = AStarPlanner.planning([2, 4], [5, 6], 0.1, 0.2, 1, 1, 9, 10)

a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
rx, ry = a_star.planning(sx, sy, gx, gy)

# print('path_x:', path_x)
# print('path_y:', path_y)

print('rx:', rx)
print('ry:', ry)
