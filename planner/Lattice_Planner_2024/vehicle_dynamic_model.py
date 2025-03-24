import math
import numpy as np

class Vehicle_Dynamic():
    def __init__(self):
        self.m = 1134  # kg 整车质量
        self.ms = 1008.1  # kg 簧上质量


    def model_equation(self):
        pass

        # self.m * (v_x_d - v_x * side_silp * yae_rate) = F_x_sum - F_w - F_f