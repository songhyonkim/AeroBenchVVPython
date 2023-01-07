import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

# 第三部分
# 一些功能函数的实现

# 3.1 计算欧几里得距离
def Euclid(wpt, wpt_target):
    
    d = np.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    
    return d

# 3.3 速度矢量计算函数
def get_vector(point_a, point_b):

    vector_x = point_b[0] - point_a[0]
    vector_y = point_b[1] - point_a[1]
    vector_z = point_b[2] - point_a[2]

    return [vector_x, vector_y, vector_z]


# 3.4 矢量夹角计算函数
def vectors_angle(vector_a, vector_b):

    length_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[2]**2)
    length_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2 + vector_b[2]**2)
    angle = sum(np.multiply(vector_a, vector_b))/(length_a * length_b)

    return angle


# 3.5 敌机飞行函数
def EnemyFly(x, y, z, speed):
    # x,y,z：敌机当前的坐标
    # 输出：下一时刻敌机的坐标

    x_next = x + speed[0]
    y_next = y + speed[1]
    z_next = z + speed[2]

    return np.array([x_next, y_next, z_next])


# 3.6 导弹飞行函数
def Missile(m_x, m_y, m_z, my_position):
    # m_x,m_y,m_z：敌方导弹当前的坐标
    # my_x, my_y, my_z：我方当前的坐标
    # 输出：下一时刻敌机导弹的坐标

    theta_x = (m_x - my_position[0])/5
    theta_y = (m_y - my_position[1])/5
    theta_z = (m_z - my_position[2])/5

    mx_next = m_x - theta_x
    my_next = m_y - theta_y
    mz_next = m_z - theta_z

    return np.array([mx_next, my_next, mz_next])


# 3.7 归一化函数
def normalization(data_list):
    b = set(data_list)
    if len(b) == 1:
        return data_list
    else:
        _range = max(data_list) - min(data_list)
        return [(data_list[i] - min(data_list))/_range for i in range(len(data_list))]


# 3.8 画威胁球体的函数
def ball(center, radius):

    u = np.linspace(0, 2*np.pi, 10)

    v = np.linspace(0, np.pi, 10)

    u, v = np.meshgrid(u, v)
    x = radius*np.cos(u)*np.sin(v) + center[0]
    y = radius*np.sin(u)*np.sin(v) + center[1]
    z = radius*np.cos(v) + center[2]

    return x, y, z

# 3.9 三维画图函数
def plot_point(pos_data):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    ax.scatter(pos_data[:, 0], pos_data[:, 1], pos_data[:, 2], 'blue', marker='o')
    plt.show()



