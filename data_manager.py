import numpy as np
import scipy.io
import pandas as pd
import os
import time, csv
class DataManager(object):
    """
    class to read and store simulation results
    before use, please create a direction under current file path './data'
    and must have a file 'init_location.xlsx' which contain the position of each entities
    """
    def __init__(self, file_path='./data'):
        # 1 init location data
        # self.store_list = store_list
        self.init_data_file = file_path + '/init_location.xlsx'
    def read_init_location(self, entity_type, index ):
        if entity_type == 'user' or 'ARISUAV' or 'BS' or 'attacker':
            return np.array([\
            pd.read_excel(self.init_data_file, sheet_name=entity_type)['x'][index],\
            pd.read_excel(self.init_data_file, sheet_name=entity_type)['y'][index],\
            pd.read_excel(self.init_data_file, sheet_name=entity_type)['z'][index]])
        else:
            return None
def ensure_directory(directory):
    """
    检查目录是否存在，如果不存在则创建。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
# 确保两个目录存在
ensure_directory('./Learning Curves/UAV_move/')
ensure_directory('./Learning Curves/UAV_coordinate/')
ensure_directory('./Learning Curves/sum_reward_episode/')