from math_tool import *
import numpy as np
import math
import cmath
class channel(object):
    """
    generate MmWave under UMi open
    input: distance, angle, pair entity object
    output: Instantaneous CSI
    """
    def __init__(self, transmitter, receiver, frequncy):
        """
        transmitter: object in entity.py
        receiver: object in entity.py
        
        """
        self.channel_name = ''
        self.transmitter = transmitter 
        self.receiver = receiver
        self.channel_type = self.init_type()    # 'BS_ARISUAV', 'ARISUAV_user'
        # self.distance = np.linalg.norm(transmitter.coordinate - receiver.coordinate)
        self.frequncy = frequncy
        # init & update channel CSI matrix
        self.channel_matrix = self.get_estimated_channel_matrix()
        
    def init_type(self):
        channel_type = self.transmitter.type+'_'+self.receiver.type
        if channel_type == 'BS_ARISUAV':
            self.rician_factor = 1000
            self.beta = 0.001
            self.c_0 = 2.8
        elif channel_type == 'ARISUAV_user' or channel_type == 'ARISUAV_attacker':
            self.rician_factor = 1000
            self.beta = 0.001
            self.c_0 = 2.8
        return channel_type

    def get_estimated_channel_matrix(self):
        """
        init & update channel matrix
        """
        # init matrix
        N_t = self.transmitter.ant_num
        N_r = self.receiver.ant_num
        #print('NT',N_t)
        channel_matrix = np.array(np.ones(shape=(N_r, N_t), dtype=complex), dtype=complex)

        # get relevant spherical_coordinate (出发角)
        r_t_r, r_t_theta, r_t_fai = cartesian_coordinate_to_spherical_coordinate(cartesian_coordinate=self.receiver.coordinate - self.transmitter.coordinate)
        #print('sssssss',self.receiver.type,self.transmitter.type)
        #print('sda',self.receiver.coordinate - self.transmitter.coordinate)
        #print('theta,fai',r_t_theta, r_t_fai)

        # get relevant spherical_coordinate（到达角）
        t_r_r, t_r_theta, t_r_fai = cartesian_coordinate_to_spherical_coordinate(cartesian_coordinate=self.transmitter.coordinate - self.receiver.coordinate)

        # calculate array response
        t_array_response = self.generate_array_response1(self.transmitter, r_t_theta, r_t_fai)
        # print('t_array_response',t_array_response)
        r_array_response = self.generate_array_response2(self.receiver, t_r_theta, t_r_fai)
        # print('r_array_response', r_array_response)
        # array_response_product = r_array_response * t_array_response
        # get H_LOS
        d = np.linalg.norm(self.transmitter.coordinate - self.receiver.coordinate)*100
        #print('TT',self.transmitter.type,self.receiver.type)
        #print('T',self.transmitter.coordinate)
        #print('R',self.receiver.coordinate)
        #print('d',d)
        #   get LOS phase shift
        # channel_matrix = cmath.exp(1j*LOS_fai)* math.pow(PL, 0.5) * array_response_product
        array_response_product = r_array_response.T @ t_array_response
        # print('array_response_product', array_response_product)
        shape = array_response_product.shape
        # print("array_response_product",array_response_product)
        # print("self.channel_type",self.channel_type)
        random_channel_real = np.random.normal(0, 1, size=shape)
        random_channel_imag = np.random.normal(0, 1, size=shape)
        random_channel = (random_channel_real + 1j * random_channel_imag) / np.sqrt(2)
        # print('random',random_channel)
        channel_matrix = (np.sqrt(self.beta * math.pow(d, -self.c_0)) *
                          (np.sqrt(self.rician_factor / (1 + self.rician_factor)) * array_response_product + np.sqrt(
                              1 / (1 + self.rician_factor)) * random_channel))
        return channel_matrix

    def generate_array_response1(self, transmitter, theta, fai):
        """
        transmitter: BS OR RISUAV_user object

        """
        ant_type = transmitter.ant_type
        ant_num = transmitter.ant_num
        type = transmitter.type
        Lambda = self.frequncy / 3e8

        if ant_type == 'UPA':
            row_num = int(math.sqrt(ant_num))
            Planar_response = np.array(np.ones(shape=(1, ant_num)), dtype=complex)
            a1 = np.array(np.ones(shape=(1, row_num)), dtype=complex)
            a2 = np.array(np.ones(shape=(1, row_num)), dtype=complex)
            for i in range(row_num):
                if type == 'BS':
                    a1[0, i] = cmath.exp(1j*(2 * math.pi / Lambda) * 0.5 * Lambda*(math.cos(theta))*i)
                    a2[0, i] = cmath.exp(1j*(2 * math.pi / Lambda) * 0.5 * Lambda*(math.sin(theta) * math.sin(fai))*i)
                elif type == 'ARISUAV':
                    a1[0, i] = cmath.exp(1j * (2 * math.pi / Lambda) * 0.5 * Lambda * (math.sin(theta)*math.sin(fai)) * i)
                    a2[0, i] = cmath.exp(1j * (2 * math.pi / Lambda) * 0.5 * Lambda * (math.sin(theta) * math.cos(fai)) * i)
            #print('a1',a1,'a2',a2)
            Planar_response = np.kron(a1, a2)
            #print('aa',Planar_response)
            return Planar_response
        elif ant_type == 'single':#not use
            return 1
        else:
            return False


    def generate_array_response2(self, receiver, theta, fai):
        """
        receiver: RISUAV_user OR User object
        """
        ant_type = receiver.ant_type
        ant_num = receiver.ant_num
        Lambda = self.frequncy / 3e8
        type = receiver.type
        if ant_type == 'UPA':
            row_num = int(math.sqrt(ant_num))
            Planar_response = np.array(np.ones(shape=(1, ant_num)), dtype=complex)
            a1 = np.array(np.ones(shape=(1, row_num)), dtype=complex)
            a2 = np.array(np.ones(shape=(1, row_num)), dtype=complex)
            for i in range(row_num):
                if type == 'ARISUAV':
                    a1[0, i] = cmath.exp(1j * (2 * math.pi / Lambda) * 0.5 * Lambda * (math.sin(theta)*math.sin(fai)) * i)
                    a2[0, i] = cmath.exp(1j * (2 * math.pi / Lambda) * 0.5 * Lambda * (math.sin(theta) * math.cos(fai)) * i)
            Planar_response = np.kron(a1, a2)
            return Planar_response
        elif ant_type == 'single':
            return np.array([1])
        else:
            return False

        
    def update_CSI(self):
        """
        update pathloss and channel matrix
        """
        # init & update channel CSI matrix
        self.channel_matrix = self.get_estimated_channel_matrix()
