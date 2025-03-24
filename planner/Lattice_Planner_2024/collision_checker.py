import numpy as np
import scipy.spatial
from math import sin, cos, pi, sqrt

class CollisionChecker:
    def __init__(self, circle_offsets, circle_radii, weight):
        self._circle_offsets = circle_offsets
        self._circle_radii   = circle_radii
        self._weight         = weight

    # '''
    def collision_check(self, paths, obstacles_1, obstacles_2, obstacles_3, self_Car):

        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            collision_free_1 = True
            collision_free_2 = True
            collision_free_3 = True
            collision_free_4 = True
            collision_free_5 = True
            collision_free_6 = True
            path           = paths[i]

            # Iterate over the points in the path.
            # lead car
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.5
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.5
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break

                if not collision_free_1:
                    break


            # follow
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.8
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.8
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break

                if not collision_free_2:
                    break

            # side lead
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))
                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_3 = np.zeros((len(self._circle_offsets), 2))
                circle_locations_4 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets * cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets * sin(path[2][j])
                circle_locations_2[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 0.3
                circle_locations_2[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 0.3
                circle_locations_3[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.0
                circle_locations_3[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.0
                circle_locations_4[:, 0] = path[0][j] + circle_offsets * cos(path[2][j]) + self_Car[0] * np.cos(
                    self_Car[1]) * 1.5
                circle_locations_4[:, 1] = path[1][j] + circle_offsets * sin(path[2][j]) + self_Car[0] * np.sin(
                    self_Car[1]) * 1.5

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_3)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations_4)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                       not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break

                if not collision_free_3:
                    break


            if collision_free_1 == collision_free_2 == collision_free_3:
                collision_free = collision_free_1
            else:
                collision_free = False

            collision_check_array[i] = collision_free

        return collision_check_array
    # '''

    '''
    def collision_check(self, paths, obstacles_1, obstacles_2, obstacles_3, self_Car):

        collision_check_array = np.zeros(len(paths), dtype=bool)
        for i in range(len(paths)):
            collision_free = True
            collision_free_1 = True
            collision_free_2 = True
            collision_free_3 = True
            collision_free_4 = True
            collision_free_5 = True
            collision_free_6 = True
            path           = paths[i]

            # Iterate over the points in the path.
            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_1 = collision_free_1 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_1:
                        break
                if not collision_free_1:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 1.5
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 1.5

                # n=len(obstacles)/2
                for n in range(len(obstacles_1)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_1[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_2 = collision_free_2 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_2:
                        break
                if not collision_free_2:
                    break

            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_3 = collision_free_3 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_3:
                        break
                if not collision_free_3:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 1.0
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 1.0

                # n=len(obstacles)/2
                for n in range(len(obstacles_2)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_2[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_4 = collision_free_4 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_4:
                        break
                if not collision_free_4:
                    break

            for j in range(len(path[0])):

                circle_locations = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations[:, 0] = path[0][j] + circle_offsets*cos(path[2][j])
                circle_locations[:, 1] = path[1][j] + circle_offsets*sin(path[2][j])

                for k in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[k],
                                                     circle_locations)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_5 = collision_free_5 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_5:
                        break
                if not collision_free_5:
                    break



            for m in range(len(path[0])):

                circle_locations_2 = np.zeros((len(self._circle_offsets), 2))

                circle_offsets = np.array(self._circle_offsets)
                circle_locations_2[:, 0] = path[0][m] + circle_offsets*cos(path[2][m]) + self_Car[0] * np.cos(self_Car[1]) * 0.5
                circle_locations_2[:, 1] = path[1][m] + circle_offsets*sin(path[2][m]) + self_Car[0] * np.sin(self_Car[1]) * 0.5


                for n in range(len(obstacles_3)):
                    collision_dists = \
                        scipy.spatial.distance.cdist(obstacles_3[n],
                                                     circle_locations_2)
                    collision_dists = np.subtract(collision_dists,
                                                  self._circle_radii)
                    collision_free_6 = collision_free_6 and \
                                     not np.any(collision_dists < 0)

                    if not collision_free_6:
                        break
                if not collision_free_6:
                    break


            if collision_free_1 == collision_free_2 == collision_free_3 == collision_free_4 == collision_free_5 == collision_free_6:
                collision_free = collision_free_1
            else:
                collision_free = False

            collision_check_array[i] = collision_free

        return collision_check_array
    '''


    def select_best_path_index(self, paths, collision_check_array):

        best_index = int(len(paths) / 2)
        best_score = float('Inf')
        score = best_score
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i] == True:

                diff = i-5
                score = np.sqrt(diff**2)
                if i == 1:
                    score = score - 3

                if i == 9:
                    score = score - 3

            if score < best_score:
                best_score = score
                best_index = i

        if best_score == float('Inf'):
            best_index = int(len(paths) / 2)

        return best_index

    def select_best_path_index_city(self, paths, collision_check_array):

        best_index = int(len(paths) / 2)
        best_score = float('Inf')
        score = best_score
        for i in range(len(paths)):
            # Handle the case of collision-free paths.
            if collision_check_array[i] == True:

                diff = i-5
                score = np.sqrt(diff**2)
                # if i == 1:
                #     score = score - 3
                #
                # if i == 9:
                #     score = score - 3

            if score < best_score:
                best_score = score
                best_index = i

        if best_score == float('Inf'):
            best_index = int(len(paths) / 2)

        return best_index
