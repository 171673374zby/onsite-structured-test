# import math
import time
from collections import deque
import numpy as np
class PIDAngleController(object):

    def __init__(
        self,
        K_P: float = 1.0,
        K_D: float = 0.0,
        K_I: float = 0.0,
        dt: float = 0.03,
        use_real_time: bool = False,
    ):
        self._k_p = K_P
        self._k_d = K_D
        self._k_i = K_I
        self._dt = dt
        self._use_real_time = use_real_time
        self._last_time = time.time()
        self._error_buffer = deque(maxlen=10)

    def get_angle(self, target_angle: float, current_angle: float, max_pid_output: float, min_pid: float):

        error = target_angle - current_angle
        self._error_buffer.append(error)

        if self._use_real_time:
            time_now = time.time()
            dt = time_now - self._last_time
            self._last_time = time_now
        else:
            dt = self._dt
            # print(dt)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / dt
            _ie = sum(self._error_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0


        return np.clip(
            (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie),
            -min_pid * dt,
            max_pid_output * dt,
        )