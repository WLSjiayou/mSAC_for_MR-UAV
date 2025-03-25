from entity import *
from channel import *
from math_tool import *
from data_manager import DataManager
import numpy as np
import time
import math
from datetime import datetime
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


class MiniSystem(object):
    """
    define mini RISUAV communication system with three user
    """

    def __init__(self, step=500, attacker_num=1, user_num=3, RIS_ant_num=16, seed=0,
                 BS_ant_num=16, p_total=60, awgn_power=-114, fre=28e9,
                 hot_noise_power=-114, max_ARIS_power=30):
        self.user_num = user_num
        self.t = 0  #计步器
        self.p_total = p_total  # BS发射总功率dBm
        self.seed(seed)
        self.T = step
        self.max_ARIS_power = max_ARIS_power
        self.energy = 0
        # self.safe_amp = []

        # 1.init entities: 1 ARISUAV, 1 BS, many users, and attacker
        self.data_manager = DataManager(file_path='./data')
        # 1.1 init RISUAV position
        self.ARISUAV = ARISUAV(
            coordinate=self.data_manager.read_init_location('ARISUAV', 0), ant_num=RIS_ant_num,
            hot_noise_power=hot_noise_power)

        # 1.2 init BS and beamforming matrix
        self.BS = BS(coordinate=self.data_manager.read_init_location('BS', 0), user_num=user_num, ant_num=BS_ant_num)

        # 1.3 init users
        self.user_list = []
        for i in range(user_num):
            user_coordinate = self.data_manager.read_init_location('user', i)
            user = User(coordinate=user_coordinate, index=i)
            self.user_list.append(user)

        # 1.4 init attacker
        self.attacker_list = []
        for i in range(attacker_num):
            attacker_coordinate = self.data_manager.read_init_location('attacker', i)
            attacker = Attacker(coordinate=attacker_coordinate, index=i)
            attacker.capacity = np.zeros(user_num)
            self.attacker_list.append(attacker)
        # 1.5 generate the eavesdrop capacity array , shape: P X K
        self.eavesdrop_capacity_array = np.zeros((attacker_num, user_num))
        # 2.init channel
        self.H_BA = channel(self.BS, self.ARISUAV, fre)
        #print('yuan',self.H_BA.channel_matrix.shape)
        self.h_A_k = []
        # print("self.user_list",self.user_list)
        for user_k in self.user_list:
            self.h_A_k.append(channel(self.ARISUAV, user_k, fre))
        self.h_A_a = []
        for attacker_p in self.attacker_list:
            self.h_A_a.append(channel(self.ARISUAV, attacker_p, fre))

    # s.t every simulition is the same model
    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        """
        reset ARISUAV, users,  beamforming matrix, reflecting coefficient
        """
        self.t = 0
        self.energy = 0
        # self.safe_amp = []
        # 1 reset RISUAV
        self.ARISUAV.reset(coordinate=self.data_manager.read_init_location('ARISUAV', 0))
        # 2 reset beamforming matrix
        self.BS.W = np.array(np.ones((self.BS.ant_num, self.user_num), dtype=complex), dtype=complex)
        #print('W',self.BS.W)
        # 3 reset reflecting coefficient
        self.ARISUAV.Phi = np.array(np.diag(np.ones(self.ARISUAV.ant_num, dtype=complex)), dtype=complex)
        # print('111111',self.RISUAV.Phi)
        # 4 reset CSI
        self.H_BA.update_CSI()
        for h in self.h_A_k + self.h_A_a:
            h.update_CSI()
        # 5 reset capacity and secrecy rate
        self.update_channel_capacity()
        return self.t

    def step(self, action, t):
        """
        test step only move UAV and update channel
        """
        self.t = t + 1
        #print("action11",action)
        # 1 update entities
        amp = action[3:3 + self.ARISUAV.ant_num]
        Phi = action[3 + self.ARISUAV.ant_num:3 + 2 * self.ARISUAV.ant_num]
        #print('22222',Phi)
        W = action[3 + 2 * self.ARISUAV.ant_num:]
        uav_location_pre = np.zeros([1, 3])  # make a copy of the uav's location2*2
        uav_location_pre[0][0] = self.ARISUAV.coordinate[0]
        uav_location_pre[0][1] = self.ARISUAV.coordinate[1]
        uav_location_pre[0][2] = self.ARISUAV.coordinate[2]
        self.ARISUAV.update_coordinate(action[0], action[1], action[2])  # execute the action
        fa3, bound, out = self.boundary_margin(self.ARISUAV.coordinate)
        while not bound:
            self.ARISUAV.coordinate[0] = uav_location_pre[0][0]
            self.ARISUAV.coordinate[1] = uav_location_pre[0][1]
            self.ARISUAV.coordinate[2] = uav_location_pre[0][2]
            if out[0] == 1:
                #print('a[1]',action[1])
                action[1] = (action[1] + math.pi / 8)
                if action[1] > math.pi:
                    action[1] = action[1] - 2*math.pi
                #print('a[1]', action[1])
            if out[1] == 1:
                action[0] = action[0] + math.pi / 16
                if action[0] > math.pi / 2:
                    action[0] = action[0] - math.pi
            #print("0,1",action[0],action[1])
            self.ARISUAV.update_coordinate(action[0], action[1], action[2])
            _, bound, out = self.boundary_margin(self.ARISUAV.coordinate)
        # 2 update channel CSI
        for h in self.h_A_k + self.h_A_a:
            h.update_CSI()
        self.H_BA.update_CSI()
        # 3 update beamforming matrix & reflecting phase shift
        """
        self.BS.W = W
        self.ARISUAV.Phi = Phi
        """
        # print('333333', self.RISUAV.Phi)
        self.BS.W = convert_list_to_complex_matrix(W, (self.BS.ant_num, self.user_num))
        # print('W2', self.BS.W)
        P = abs(np.trace(self.BS.W.conj().T @ self.BS.W))
        P_total = dB_to_normal(self.p_total)
        scaling_factor = np.sqrt(P_total / P)
        self.BS.W *= scaling_factor
        # print("a1",amp)
        amp = self.up_elements_amp(amp)
        # print("a2",amp)
        # self.safe_amp = np.append(self.safe_amp, amp)
        # print("d",vector_to_diag(amp))
        # print("p",convert_list_to_complex_diag(Phi, self.ARISUAV.ant_num))
        self.ARISUAV.Phi = vector_to_diag(amp) * convert_list_to_complex_diag(Phi, self.ARISUAV.ant_num)
        # print("pp",self.ARISUAV.Phi)
        # for i in range(self.ARISUAV.ant_num):
        #     e_c = 0
        #     for user in range(3):
        #         e_c += math.pow(np.linalg.norm(np.dot(self.H_BA.channel_matrix[i, :], self.BS.W[:, user])), 2)
        #     e_c += dB_to_normal(self.ARISUAV.hot_noise_power) * 1e-3
        #     # print(e_c)
        #     e_c *= action[3 + i] ** 2
        #     # print('e_c', e_c)
        #     if e_c > dB_to_normal(self.max_ARIS_power) * 1e-3:
        #         fa1 = 1
        #       print("111111111111111111111111111111111111111111111")
        # print('fa1 =', fa1)
        # 4 update channel capacity in every user
        self.update_channel_capacity()
        self.ARISUAV_C(amp)
        # 5 get new state
        new_state = self.observe()
        # 6 get reward
        # calculate penalty
        secrecy, energy_uav = self.reward(action)
        reward = (math.pow(10, secrecy/energy_uav+1)-10) - 0.01*fa3
        done = False
        if self.t >= self.T:
            done = True
            # print("amp", self.safe_amp)
            print("energy:", self.energy)
            self.energy = 0
            # self.safe_amp = []
        return secrecy, energy_uav, reward, new_state, done, self.t

    def boundary_margin(self, RISUAV_coordinate):
        bound = True
        fa = 0
        out = [0, 0]
        if RISUAV_coordinate[0] < 0.0 or RISUAV_coordinate[0] > 7:
            bound = False
            fa = 1
            out[0] = 1
        if RISUAV_coordinate[1] < 0.0 or RISUAV_coordinate[1] > 7:
            bound = False
            fa = 1
            out[0] = 1
        if RISUAV_coordinate[2] < 1 or RISUAV_coordinate[2] > 5:
            bound = False
            fa = 1
            out[1] = 1
        return fa, bound, out

    def reward(self, action):
        """
        used in function step to get the reward of current step
        """
        secrecy = 0
        for user in self.user_list:
            #print('ea',self.eavesdrop_capacity_array)
            r = max(0, user.capacity - max(self.eavesdrop_capacity_array[:, user.index]))
            secrecy += r
        energy_consume = self.energy_consume_uav(action)
        #print('energy_consume1 :', energy_consume1)
        return secrecy, energy_consume

    def energy_consume_uav(self, action):
        """
        used in function step to get the energy consume of current step
        """
        v1 = abs(action[2] * 100 * np.cos(action[0]) / 2)
        v2 = abs(action[2] * 100 * np.sin(action[0]) / 2)
        P0 = 79.8563
        P1 = 88.6279
        P2 = 11.46
        U_tip = 120
        v0 = 4.30
        d1 = 0.6
        s = 0.05
        rho = 1.225
        A = 0.503
        e_uav = P0 * (1 + 3 * v1 ** 2 / U_tip ** 2) + P1 * math.sqrt(math.sqrt(1 + v1 ** 4 / (4 * v0 ** 4)) - v1 ** 2 / (2 * v0 ** 2)) + 0.5 * d1 * rho * s * A * v1 ** 3 + P2 * v2
        return e_uav

    def up_elements_amp(self, action):
        e_c = 0
        e_h = 0
        an_sum = 0
        for item, an in enumerate(action):
            e_h_low = 0
            e_c_low = 0
            if an < 1:
                for user in range(self.user_num):
                    e_h_low += math.pow(np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])),2)
                e_h_low *= (1 - an ** 2)
            if an > 1:
                an_sum += an
                for user in range(self.user_num):
                    e_c_low += math.pow(np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])),
                                        2)
                e_c_low *= (1 - an ** 2)
            e_c += e_c_low
            e_h += e_h_low
        up_action = action
        if an_sum != 0:
            x = e_h / an_sum
            for item, an in enumerate(action):
                g = 0
                if an > 1:
                    for user in range(self.user_num):
                        g += math.pow(
                            np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])),
                            2)
                    up_action[item] = 1+math.sqrt(x*an/g)

        # if energy_consume > 0:
        #     greater_than_one = [(i, x) for i, x in enumerate(action) if x >= 1]
        #     greater_than_one.sort(key=lambda x: x[1])
        #     sort_amp = [index for index, value in greater_than_one]
        #     for i in sort_amp:
        #         up_action = action/action[i]
        #         if self.calculate_c_h(up_action) <= 0:
        #             sum_low = 0
        #             sum_up = 0
        #             for item, an in enumerate(action):
        #                 e_c_low_1 = 0
        #                 e_c_low_2 = 0
        #                 if an > 1 and an >= action[i]:
        #                     for user in range(self.user_num):
        #                         e_c_low_2 += math.pow(
        #                             np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])), 2)
        #                     e_c_low_2 += dB_to_normal(self.ARISUAV.hot_noise_power) * 1e-3
        #                     sum_low += e_c_low_2
        #                     sum_up += (an ** 2) * e_c_low_2
        #                 else:
        #                     for user in range(self.user_num):
        #                         e_c_low_1 += math.pow(
        #                             np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])), 2)
        #                     sum_low += e_c_low_1
        #                     sum_up += (an ** 2) * e_c_low_1
        #             x = sqrt(sum_low / sum_up)
        #             up_action = action * x
        #             break

        return up_action

    def ARISUAV_C(self, action):
        e_c = 0
        e_h = 0
        for item, an in enumerate(action):
            e_c_low = 0
            e_h_low = 0
            if an > 1:
                for user in range(self.user_num):
                    e_c_low += math.pow(np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])),
                                        2)
                # e_c_low += dB_to_normal(self.ARISUAV.hot_noise_power) * 1e-3
                e_c_low *= (an - 1)** 2

            if an < 1:
                for user in range(self.user_num):
                    e_h_low += math.pow(np.linalg.norm(np.dot(self.H_BA.channel_matrix[item, :], self.BS.W[:, user])),
                                        2)
                e_h_low *= (1 - an ** 2)
            e_c += e_c_low
            e_h += e_h_low
        energy_consume = e_h - e_c
        self.energy += (e_h - e_c)

    def observe(self):
        """
        Used in function main to get current state
        The state is a list with users' and attackers' comprehensive channel
        """
        # Initialize an empty list to hold the real and imaginary parts
        comprehensive_channel_elements_list = []

        # Loop through each entity in the user list
        for entity in self.user_list:
            # Extract the real and imaginary parts of the channel matrix for the current entity
            real_parts = np.real(self.h_A_k[entity.index].channel_matrix @  self.H_BA.channel_matrix)
            imag_parts = np.imag(self.h_A_k[entity.index].channel_matrix @  self.H_BA.channel_matrix)

            # Flatten the real and imaginary parts and append to the list
            if len(comprehensive_channel_elements_list) == 0:
                comprehensive_channel_elements_list = np.concatenate((real_parts.flatten(), imag_parts.flatten()))
            else:
                comprehensive_channel_elements_list = np.concatenate(
                    (comprehensive_channel_elements_list, real_parts.flatten(), imag_parts.flatten()))

        # Combine with the ARISUAV coordinates
        ARISUAV_coordinates = np.array(self.ARISUAV.coordinate)
        # Ensure comprehensive_channel_elements_list is a NumPy array before concatenation
        state = np.concatenate((ARISUAV_coordinates, comprehensive_channel_elements_list))

        return state

    def update_channel_capacity(self):
        """
        function used in step to calculate user and attackers' capacity
        """
        for user in self.user_list:
            # print("self.sorted_channel_list",self.sorted_channel_list)
            user.capacity = self.calculate_capacity_of_user_k(user.index)
        for attacker in self.attacker_list:
            attacker.capacity = self.calculate_capacity_of_attacker(attacker.index)
            self.eavesdrop_capacity_array[attacker.index, :] = attacker.capacity

    def calculate_capacity_of_user_k(self, k):
        """
        function used in update_channel_capacity to calculate one user
        """
        noise_power = self.user_list[k].noise_power
        h_A_k = self.h_A_k[k].channel_matrix
        Phi = self.ARISUAV.Phi
        H_BA = self.H_BA.channel_matrix  #N*M
        W_k = self.BS.W[:, k]
        beta_k = 0
        for i in range(self.user_num):
            if i != k:
                W_k_1 = self.BS.W[:, i]  #M*1
                beta_k += math.pow(abs(h_A_k @ Phi @ H_BA @ W_k_1), 2)
        beta_k += dB_to_normal(noise_power) * 1e-3
        # beta_k += dB_to_normal(self.ARISUAV.hot_noise_power) * 1e-3 * math.pow(np.linalg.norm(np.dot(h_A_k, Phi)), 2)
        alpha_k = abs(h_A_k @ Phi @ H_BA @ W_k) ** 2
        return math.log2(1 + alpha_k / beta_k)

    def calculate_capacity_of_attacker(self, k):
        """
        function used in update_channel_capacity to calculate one user
        """
        noise_power = self.attacker_list[k].noise_power
        h_A_a = self.h_A_a[k].channel_matrix
        Phi = self.ARISUAV.Phi
        H_BA = self.H_BA.channel_matrix  #N*M
        attacker_capacity = []
        for i in range(self.user_num):
            W_k = self.BS.W[:, i]
            beta_k = 0
            for j in range(self.user_num):
                if j != i:
                    W_k_1 = self.BS.W[:, j]  # M*1
                    beta_k += math.pow(abs(h_A_a @ Phi @ H_BA @ W_k_1), 2)
            beta_k += dB_to_normal(noise_power) * 1e-3
            # beta_k += dB_to_normal(self.ARISUAV.hot_noise_power) * 1e-3 * math.pow(np.linalg.norm(np.dot(h_A_a, Phi)), 2)
            alpha_k = abs(h_A_a @ Phi @ H_BA @ W_k) ** 2
            attacker_capacity = np.append(attacker_capacity, math.log2(1 + alpha_k / beta_k))
        return attacker_capacity

    def get_system_action_dim(self):
        """
        function used in main function to get the dimention of actions
        """
        result = 0
        # 0 UAV movement
        result += 3
        # 1 RIS reflecting elements
        result += 2 * self.ARISUAV.ant_num
        # 2 beamforming matrix dimention
        result += 2 * self.BS.ant_num * self.user_num
        return 3, 2*self.ARISUAV.ant_num, 2 * self.BS.ant_num * self.user_num, result

    def get_system_state_dim(self):
        """
        function used in main function to get the dimention of states
        """
        result = 3
        result += 2 * self.user_num * self.BS.ant_num
        return result
