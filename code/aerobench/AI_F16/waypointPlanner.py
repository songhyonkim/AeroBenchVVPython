import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

# 航迹节点扩展函数--使用SAS搜索算法
def sas(wpt_myself, wpt_target, wptPath_minLen, psi_max, theta_max, M, N, K):
    # wpt_myself:当前航迹点坐标
    # wpt_target:目标航迹点坐标
    # wptPath_minLen:最短航迹路径距离
    # psi_max:飞机最大侧偏角
    # theta_max:飞机最大爬升/俯冲角
    # M:俯仰区子扇形面个数
    # N:侧偏区子扇形面个数
    # K:半径子段数

    # 计算当前航迹点对目标的指向角度和俯仰角度
    me2enemy_vector = get_vector(wpt_myself, wpt_target[-1])

    vector_1 = [me2enemy_vector[0], me2enemy_vector[1], 0]
    vector_2 = [me2enemy_vector[0], 0, me2enemy_vector[2]]
    axis_x = [1, 0, 0]
    axis_z = [-1, 0, 0]

    me2enemy_angle = vectors_angle(vector_1, axis_x)
    me2enemy_yaw = vectors_angle(vector_2, axis_z)
    print(me2enemy_angle)
    print(me2enemy_yaw)

    # 应当考虑航迹路径分段
    epsilon = np.pi/20
    psi_list = [me2enemy_angle - psi_max/2 + psi_max/N*i for i in range(N+1)]
    theta_list = [np.pi/2 - me2enemy_yaw - theta_max/2 + theta_max/M*i for i in range(M+1)]
    radius_list = [wptPath_minLen/K*(i+1) for i in range(K)]

    nextPossible_wpts_x = []
    nextPossible_wpts_y = []
    nextPossible_wpts_z = []

    for k in range(K):
        for i in range(M):
            for j in range(N):
                x = radius_list[k]*math.sin(theta_list[i])*math.cos(psi_list[j]) + wpt_myself[0]
                y = radius_list[k]*math.sin(theta_list[i])*math.sin(psi_list[j]) + wpt_myself[1]
                z = radius_list[k]*math.cos(theta_list[i]) + wpt_myself[2]

                nextPossible_wpts_x.append(x)
                nextPossible_wpts_y.append(y)
                nextPossible_wpts_z.append(z)

    return [nextPossible_wpts_x, nextPossible_wpts_y, nextPossible_wpts_z]


# 真实航迹代价函数
def cost(wpt_myself, wpt_next, max_len, enemy_points_list, omega, threat_pt, threat_radius, threat_coef):
    # wpt_myself:当前航迹点的坐标
    # wpt_next:下一航迹点的坐标
    # enemy_points_list:目标航迹点
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数


    # 航段长度代价，油量代价主要与飞行航程相关
    l = u(wpt_myself, wpt_next, 1)/max_len  # 归一化
    
    # 受敌机攻击的代价，视线角，单位：度
    speed_vector_myself = get_vector(wpt_myself, wpt_next)
    speed_vector_enemy = get_vector(enemy_points_list[-2], enemy_points_list[-1])
    myself_to_enemy = get_vector(wpt_next, enemy_points_list[-1])
    enemy_to_myself = get_vector(enemy_points_list[-1], wpt_next)

    # 本机相对于敌机的视线角
    alpha_myself = vectors_angle(speed_vector_enemy, enemy_to_myself)
    # 敌机相对于本机的视线角
    alpha_enemy = vectors_angle(speed_vector_myself, myself_to_enemy)

    theta = (alpha_enemy - alpha_myself + np.pi)/(2*np.pi)

    # 受导弹攻击的代价
    # 下一航迹点与导弹的距离
    l_threat = u(wpt_next, threat_pt, 1)

    f_attack = threat_coef[0]*threat_coef[1]*threat_radius/l_threat

    cost_true = omega[0]*l**2 + omega[1]*theta**2 + omega[2]*f_attack**2

    return cost_true


# 启发函数--欧几里得距离
def u(wpt, wpt_target, D):
    # D:启发因子

    d = D*math.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    return d


# 速度矢量计算函数
def get_vector(point_a, point_b):

    vector_x = point_b[0] - point_a[0]
    vector_y = point_b[1] - point_a[1]
    vector_z = point_b[2] - point_a[2]

    return [vector_x, vector_y, vector_z]


# 矢量夹角计算函数
def vectors_angle(vector_a, vector_b):

    length_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[2]**2)
    length_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2 + vector_b[2]**2)
    angle = sum(np.multiply(vector_a, vector_b))/(length_a * length_b)

    return angle


# 在后继节点集中选择航迹代价最小的航迹点
def best_nextWpt(wpt_myself, min_len, enemy_points_list, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D):
    # wpt_myself:当前航迹点的坐标
    # enemy_points_list:目标航迹点的坐标
    # nextPossible_wpts:当前航迹点的后继节点集
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数
    # D:启发因子

    cost_list = [cost(wpt_myself, [nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], min_len, 
                enemy_points_list, omega, threat_pt, threat_radius, threat_coef) for i in range(len(nextPossible_wpts[0]))]
    u_list  = [u([nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], enemy_points_list[-1], D)
                for i in range(len(nextPossible_wpts[0]))]

    u_list= normalization(u_list)  # 归一化处理

    f_list = np.sum([cost_list, u_list], axis=0).tolist()

    best_Wpt_index = f_list.index(min(f_list))

    best_Wpt = [nextPossible_wpts[0][best_Wpt_index], nextPossible_wpts[1][best_Wpt_index], nextPossible_wpts[2][best_Wpt_index]]

    return best_Wpt, best_Wpt_index, min(f_list)



# 敌机飞行函数
def EnemyFly(x, y, z, speed):
    # x,y,z：敌机当前的坐标
    # 输出：下一时刻敌机的坐标

    x_next = x + speed[0]
    y_next = y + speed[1]
    z_next = z + speed[2]

    return [x_next, y_next, z_next]



# 导弹飞行函数
def Missile(m_x, m_y, m_z, my_x, my_y, my_z):
    # m_x,m_y,m_z：敌方导弹当前的坐标
    # my_x, my_y, my_z：我方当前的坐标
    # 输出：下一时刻敌机导弹的坐标

    theta_x = (m_x - my_x)/5
    theta_y = (m_y - my_y)/5
    theta_z = (m_z - my_z)/5

    mx_next = m_x - theta_x
    my_next = m_y - theta_y
    mz_next = m_z - theta_z

    return [mx_next, my_next, mz_next]


# 归一化函数
def normalization(data_list):
    _range = np.max(data_list) - np.min(data_list)

    return (data_list - np.min(data_list))/_range


# 画威胁球体
def ball(center, radius):

    u = np.linspace(0, 2*np.pi, 10)

    v = np.linspace(0, np.pi, 10)

    u, v = np.meshgrid(u, v)
    x = radius*np.cos(u)*np.sin(v) + center[0]
    y = radius*np.sin(u)*np.sin(v) + center[1]
    z = radius*np.cos(v) + center[2]

    return x, y, z


if __name__ == '__main__':

    # wpt_now = [110, -200, 0]
    # wpt_target = [1000,-100,500]

    wpt_now = [0, 0, 1000]
    wpt_target = [-200000, -50000, 12000]

    wptPath_minLen = 50
    psi_max = np.deg2rad(25)
    theta_max = np.deg2rad(30)
    M = 5
    N = 5

    omega = [0.1, 0.45, 0.45]
    # threat_pt = [1000, -100, 250]
    # threat_radius = 100
    threat_pt = [-70000, -25000, 8000]
    threat_radius = 10000
    
    threat_coef = [1.1, 1.2]
    D = 1.2

    wpts_list = [wpt_now]

    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wpt_now[0], wpt_now[1], wpt_now[2], c='b')
    ax.scatter(wpt_target[0], wpt_target[1], wpt_target[2], c='g')
    ax.scatter(threat_pt[0], threat_pt[1], threat_pt[2], c='black')
    x, y, z = ball(threat_pt, threat_radius)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='black')

    
    # error = u(wpt_now, wpt_target, 1)

    # while(error >= 10):
    #     # 当前航迹点下的后继航迹点集
    #     nextPossible_wpts = sas(wpt_now, wpt_target, wptPath_minLen, psi_max, theta_max, M, N)

    #     # 获得后继航迹点集中航迹代价最小的航迹点
    #     best_Wpt, best_Wpt_index, u_list =  best_nextWpt(wpt_now, wpt_target, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D)

    #     # 更新航迹点
    #     wpt_now = best_Wpt
    #     wpts_list.append(wpt_now)

    #     error = u(wpt_now, wpt_target, 1)
    #     print(error)

    #     ax.scatter(wpt_now[0], wpt_now[1], wpt_now[2], c='r')

    # 添加坐标轴(顺序是X,Y,Z)
    ax.set_xlim(-210000, 0)
    ax.set_ylim(-210000, 0)
    ax.set_zlim(0, 210000)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    plt.show()
