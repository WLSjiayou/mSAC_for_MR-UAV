import numpy as np

class ARISUAV(object):
    """
    UAV object with coordinate
    And with ULA antenas, default 8
    And limited power
    And with fixed rotation angle
    """
    def __init__(self, coordinate, hot_noise_power, ant_num=16, ant_type='UPA'):
        """
        coordinate is the init coordinate of UAV, meters, np.array
        """
        self.type = 'ARISUAV'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.Phi = np.mat(np.diag(np.ones(self.ant_num, dtype=complex)), dtype=complex)
        self.hot_noise_power = hot_noise_power

    def reset(self, coordinate):
        """
        reset UAV coordinate
        """
        self.coordinate = coordinate

    def update_coordinate(self, phi, alpha, dist):
        """
        used in function move to update UAV cordinate
        """
        # print('l1', self.coordinate)
        # print('a',phi,alpha,dist)
        self.coordinate[0] += dist * np.cos(phi) * np.cos(alpha)
        self.coordinate[1] += dist * np.cos(phi) * np.sin(alpha)
        self.coordinate[2] += dist * np.sin(phi)
        # print('l2', self.coordinate)

class BS(object):

    def __init__(self, coordinate, user_num=3, ant_num=16, ant_type='UPA'):
        self.type = 'BS'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.User_num = user_num
        self.W = np.mat(np.ones((self.ant_num, self.User_num), dtype=complex), dtype=complex)


class User(object):
    """
    user with single antenas
    """
    def __init__(self, coordinate, index, ant_num=1, ant_type='single'):
        """
        coordinate is the init coordinate of user, meters, np.array
        ant_num is the antenas number of user
        """
        self.type = 'user'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        # init the capacity
        self.capacity = 0
        # init receive noise sigma in dB
        self.noise_power = -114


class Attacker(object):
    """
    Attacker with single antenas
    """
    def __init__(self, coordinate, index, ant_num=1, ant_type='single'):
        """
        coordinate is the init coordinate of Attacker, meters, np.array
        ant_num is the antenas number of Attacker
        """
        self.type = 'attacker'
        self.coordinate = coordinate
        self.ant_num = ant_num
        self.ant_type = ant_type
        self.index = index
        # init the capacity, this is a K length np.array ,shape: (K,)
        # represent the attack rate for kth user, (must init in env.py)
        self.capacity = 0
        # init receive noise sigma in dBmW
        self.noise_power = -75
