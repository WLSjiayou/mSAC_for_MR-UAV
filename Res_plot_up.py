import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_manager import DataManager

class Res_plot(object):
    def __init__(self, num_users, num_attacker1):
        self._build_result()
        self.data_manager = DataManager(file_path='./data')
        self.num_users = num_users
        self.num_attacker = num_attacker1

    def _build_result(self):
        return

    def draw_location(self, x_uav, y_uav, z_uav, save_path):
        x_uav = np.transpose(x_uav)  # 转置
        y_uav = np.transpose(y_uav)
        z_uav = np.transpose(z_uav)
        fig = plt.figure(figsize=(10, 8))  # 设置图形大小
        ax = fig.add_subplot(projection='3d')

        # 读取用户的位置
        for i in range(self.num_users):
            x_user, y_user, z_user = self.data_manager.read_init_location('user', i)
            ax.scatter(x_user, y_user, z_user, c='green', marker='.', s=100, label='User')  # 散点大小增加

        # 绘制基站位置
        [x_bs, y_bs, z_bs] = self.data_manager.read_init_location('BS', 0)
        ax.scatter(x_bs, y_bs, z_bs, c='darkred', marker='o', s=200, label='Base Station')  # 基站使用不同的标记和颜色
        # 绘制窃听者位置
        [x_bs, y_bs, z_bs] = self.data_manager.read_init_location('attacker', 0)
        ax.scatter(x_bs, y_bs, z_bs, c='red', marker='.', s=100, label='attacker')

        # 绘制无人机轨迹
        ax.plot(x_uav[:-1], y_uav[:-1], z_uav[:-1], c='blue', linewidth=2, label='UAV Path')  # 增加线宽和点大小
        ax.plot(x_uav[-1], y_uav[-1], z_uav[-1], c='green', marker='*', markersize=15, label='End Point')  # 最后一个点用绿色星号表示

        # 设置图形标题和坐标轴标签
        plt.title('UAV Location Over Time', fontsize=20, fontweight='bold')  # 增加标题字体大小和加粗
        ax.set_zlim(0, 5)
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 7)
        ax.set_xlabel('X Coordinate', fontsize=16)  # 增加坐标轴标签字体大小
        ax.set_ylabel('Y Coordinate', fontsize=16)
        ax.set_zlabel('Z Coordinate', fontsize=16)
        #ax.view_init(elev=30, azim=270)

        # 设置图例
        plt.legend(loc='upper right', prop={'size': 12})  # 增加图例字体大小
        #plt.close()  # 关闭图形窗口

        # 保存并显示图形
        plt.savefig(save_path, dpi=300)  # 增加保存图形的分辨率
        #plt.show()

    def draw_location_2d_xoy(self, x_uav, y_uav, save_path_xoy):
        # 绘制无人机在yoz平面上的轨迹
        plt.figure(figsize=(10, 8))  # 设置图形大小
        # 读取用户的位置
        for i in range(self.num_users):
            x_user, y_user, _ = self.data_manager.read_init_location('user', i)
            plt.scatter(x_user, y_user, c='green', marker='.', s=100, label='User')  # 散点大小增加

        # 绘制基站位置
        [x_bs, y_bs, _] = self.data_manager.read_init_location('BS', 0)
        plt.scatter(x_bs, y_bs, c='darkred', marker='o', s=200, label='Base Station')  # 基站使用不同的标记和颜色
        # 绘制窃听者位置
        [x_att, y_att, _] = self.data_manager.read_init_location('attacker', 0)
        plt.scatter(x_att, y_att, c='red', marker='.', s=100, label='attacker')

        # 绘制无人机轨迹
        plt.plot(x_uav, y_uav, c='blue', linewidth=2, label='UAV Path')  # 直接使用y_uav作为x轴，z_uav作为y轴

        # 绘制终点，使用绿色星号标记
        plt.plot(x_uav[-1], y_uav[-1], c='green', marker='*', markersize=15, label='End Point')

        # 设置图形标题和坐标轴标签
        plt.title('UAV Location on XOY Plane', fontsize=20, fontweight='bold')
        plt.xlabel('x Coordinate', fontsize=16)
        plt.ylabel('y Coordinate', fontsize=16)
        plt.xlim(0, 7.5)  # 设置x轴的范围为0到5
        plt.ylim(0, 7.5)  # 设置y轴的范围为0到5

        # 添加网格
        plt.grid(True)

        # 设置图例
        plt.legend(loc='upper right', prop={'size': 12})

        # 保存并显示图形
        plt.savefig(save_path_xoy, dpi=300)
        # plt.show()

    def draw_location_2d_yoz(self, y_uav, z_uav, save_path_yoz):
        # 绘制无人机在yoz平面上的轨迹
        plt.figure(figsize=(10, 8))  # 设置图形大小

        # 绘制无人机轨迹
        plt.plot(y_uav, z_uav, c='blue', linewidth=2, label='UAV Path')  # 直接使用y_uav作为x轴，z_uav作为y轴

        # 绘制终点，使用绿色星号标记
        plt.plot(y_uav[-1], z_uav[-1], c='green', marker='*', markersize=15, label='End Point')

        # 设置图形标题和坐标轴标签
        plt.title('UAV Location on YOZ Plane', fontsize=20, fontweight='bold')
        plt.xlabel('Y Coordinate', fontsize=16)
        plt.ylabel('Z Coordinate', fontsize=16)
        plt.xlim(0, 5)  # 设置x轴的范围为0到5
        plt.ylim(0, 5)  # 设置y轴的范围为0到5

        # 添加网格
        plt.grid(True)

        # 设置图例
        plt.legend(loc='upper right', prop={'size': 12})

        # 保存并显示图形
        plt.savefig(save_path_yoz, dpi=300)
        # plt.show()

# 假设这是Res_plot类的实例化和方法调用
# res_plot = Res_plot()
# res_plot.draw_location(x_uav_data, y_uav_data, z_uav_data, 'path_to_save_image_3d.png')
# res_plot.draw_location_2d(x_uav_data, y_uav_data, 'path_to_save_image_2d.png')