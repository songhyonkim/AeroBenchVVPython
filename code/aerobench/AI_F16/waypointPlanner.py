import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

# 航迹节点扩展函数--使用SAS搜索算法
def sas(wpt_now, wpt_target, wptPath_minLen, psi_max, theta_max, M, N):
    # wpt_now:当前航迹点坐标
    # wptPath_minLen:最短航迹路径距离
    # psi_max:飞机最大侧偏角
    # theta_max:飞机最大爬升/俯冲角
    # M:俯仰区子扇形面个数
    # N:侧偏区子扇形面个数

    # 应当考虑航迹路径分段

    theta_list = [-np.pi/2 + theta_max/M*i for i in range(M+1)]
    psi_list = [psi_max/N*i for i in range(N+1)]

    nextPossible_wpts_x = []
    nextPossible_wpts_y = []
    nextPossible_wpts_z = []

    for i in range(M):
        for j in range(N):
            x = wptPath_minLen*math.sin(theta_list[i])*math.cos(psi_list[j]) + wpt_now[0]
            y = wptPath_minLen*math.sin(theta_list[i])*math.sin(psi_list[j]) + wpt_now[1]
            z = wptPath_minLen*math.cos(theta_list[i]) + wpt_now[2]

            nextPossible_wpts_x.append(x)
            nextPossible_wpts_y.append(y)
            nextPossible_wpts_z.append(z)

    return [nextPossible_wpts_x, nextPossible_wpts_y, nextPossible_wpts_z]


# 真实航迹代价函数
def cost(wpt, wpt_target, omega, threat_pt, threat_radius, threat_coef):
    # wpt:当前航迹点的坐标
    # wpt_target:目标航迹点的坐标
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数

    l = u(wpt, wpt_target, 1)

    h = abs(wpt_target[2] - wpt[2])

    # 当前航迹点与威胁点的距离
    l_threat = u(wpt, threat_pt, 1)

    if(l_threat > threat_radius):

        f_attack = 0

    else:

        f_attack = threat_coef[0]*threat_coef[1]/l_threat**4

    cost_true = omega[0]*l**2 + omega[1]*h**2 + omega[2]*f_attack

    return cost_true


# 启发函数--欧几里得距离
def u(wpt, wpt_target, D):
    # D:启发因子

    d = D*math.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    return d


# 在后继节点集中选择航迹代价最小的航迹点
def best_nextWpt(wpt_now, wpt_target, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D):
    # wpt_now:当前航迹点的坐标
    # wpt_target:目标航迹点的坐标
    # nextPossible_wpts:当前航迹点的后继节点集
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数
    # D:启发因子

    cost_list = [cost(wpt_now, [nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], 
                omega, threat_pt, threat_radius, threat_coef) for i in range(len(nextPossible_wpts[0]))]
    u_list  = [u([nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], wpt_target, D)
                for i in range(len(nextPossible_wpts[0]))]

    f_list = np.sum([cost_list, u_list], axis=0).tolist()

    best_Wpt_index = f_list.index(min(f_list))

    best_Wpt = [nextPossible_wpts[0][best_Wpt_index], nextPossible_wpts[1][best_Wpt_index], nextPossible_wpts[2][best_Wpt_index]]

    return best_Wpt, best_Wpt_index, u_list


# 画威胁球体
def ball(center, radius):

    u = np.linspace(0, 2*np.pi, 100)

    v = np.linspace(0, 2*np.pi, 100)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]

    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]

    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    return x, y, z


if __name__ == '__main__':

    wpt_now = [110, -200, 0]
    wpt_target = [1000,-100,500]

    wptPath_minLen = 50
    psi_max = np.deg2rad(25)
    theta_max = np.deg2rad(30)
    M = 5
    N = 5

    omega = [0.1, 0.45, 0.45]
    threat_pt = [1000, -100, 250]
    threat_radius = 100
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

    
    error = u(wpt_now, wpt_target, 1)

    while(error >= 10):
        # 当前航迹点下的后继航迹点集
        nextPossible_wpts = sas(wpt_now, wpt_target, wptPath_minLen, psi_max, theta_max, M, N)

        # 获得后继航迹点集中航迹代价最小的航迹点
        best_Wpt, best_Wpt_index, u_list =  best_nextWpt(wpt_now, wpt_target, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D)

        # 更新航迹点
        wpt_now = best_Wpt
        wpts_list.append(wpt_now)

        error = u(wpt_now, wpt_target, 1)
        print(error)

        ax.scatter(wpt_now[0], wpt_now[1], wpt_now[2], c='r')

    # 添加坐标轴(顺序是X,Y,Z)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    plt.show()
