import math
import cmath
import numpy as np
import pandas as pd
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

def cartesian_coordinate_to_spherical_coordinate(cartesian_coordinate):
    """
    transmit cartesian_coordinate_to_spherical_coordinate
    input 1 X 3 np.array,   [x, y, z]
    output 1 X 3 np.array,  [r, theta, fai]
    """
    r = np.linalg.norm(cartesian_coordinate)
    x = cartesian_coordinate[0]
    y = cartesian_coordinate[1]
    if cartesian_coordinate[2] == 0:
        theta = math.pi/2
    elif cartesian_coordinate[2] < 0:
        theta = math.pi - math.atan(np.sqrt(x**2+y**2)/abs(cartesian_coordinate[2]))
    else:
        theta = math.atan(np.sqrt(x**2+y**2)/abs(cartesian_coordinate[2]))
    # if theta < 0:
    #     print("1111111111111111111")
    fai = 0
    if y == 0:
        if x > 0:
            fai = 0
        elif x < 0:
            fai = math.pi
        else:
            fai = 0
    if x == 0:
        if y > 0:
            fai = math.pi/2
        elif y < 0:
            fai = math.pi*3/2
    if y > 0 and x > 0:
        fai = math.atan(y/x)
    elif x < 0 and y > 0:
        fai = math.atan(y/x) + math.pi
    elif x < 0 and y < 0:
        fai = math.atan(y/x) + math.pi
    elif x > 0 and y < 0:
        fai = math.atan(y/x) + 2*math.pi
    return r, theta, fai

def vecter_normalization(cartesian_coordinate):
    return cartesian_coordinate/np.linalg.norm(cartesian_coordinate)

def dB_to_normal(dB):
    """
    input: dB
    output: normal vaule
    """
    return math.pow(10, (dB/10))

def normal_to_dB(normal):
    """
    input: normal
    output: dB value
    """
    return -10 * math.log10(normal)


def vector_to_diag(vector):
    """
    transfer a vector into a diagnal matrix
    """
    vec_size = vector.size
    # print("vec_size",vec_size)
    diag = np.array(np.zeros((vec_size, vec_size)))
    for i in range(vec_size):
        diag[i, i] = vector[i]
    return diag

def diag_to_vector(diag):
    """
    transfer a diagnal matrix into a vector
    """
    vec_size = np.shape(diag)[0]
    vector = np.zeros(vec_size, dtype=complex)
    for i in range(vec_size):
        vector[i] = diag[i, i]
    return vector

def bigger_than_zero(value):
    """
    max(0,value)
    """
    return max(0, value)

def dataframe_to_dictionary(df):
    """
    docstring
    """
    return {col_name: df[col_name].values for col_name in df.columns.values}

def convert_list_to_complex_matrix(list_real, shape):
    """
    list_real is a 2* N*K dim list, convert it to N X K complex matrix
    shape is a tuple (N, K)
    """
    N = shape[0]
    K = shape[1]
    matrix_complex = np.array(np.zeros((N, K), dtype=complex), dtype=complex)
    for i in range(N):
        for j in range(K):
            matrix_complex[i, j] = list_real[2*(i*K + j)] + 1j * list_real[2*(i*K + j) + 1]
    return matrix_complex

def convert_list_to_complex_diag(list_real, diag_row_num):
    """
    list_real is a M dim list, convert it to M X M complex diag matrix
    diag_row_num is the M
    """
    M = diag_row_num
    diag_matrix_complex = np.array(np.diag(np.ones(M, dtype=complex)), dtype=complex)
    for i in range(M):
        diag_matrix_complex[i, i] = cmath.exp(1j * (list_real[i]))
        # diag_matrix_complex[i, i] = cmath.exp(1j * list_real[i] * math.pi)
    return diag_matrix_complex

def map_to(x, x_range:tuple, y_range:tuple):#范围映射x到y
    x_min = x_range[0]
    x_max = x_range[1]
    y_min = y_range[0]
    y_max = y_range[1]
    y = y_min+(y_max - y_min) / (x_max - x_min) * (x - x_min)
    return y

