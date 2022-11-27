import math
import sys
import random
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


# 第一部分
# 1.1 节点扩展函数--SAS搜索算法
# 算法目的：在规划空间中生成备选节点集
def sas(wpt_myself, wptPath_minLen, psi_max, theta_max, M, N):
    # wpt_myself:当前点坐标
    # wpt_target:目标点坐标
    # wptPath_minLen:最短路径距离
    # psi_max:飞机最大侧偏角
    # theta_max:飞机最大爬升/俯冲角
    # M:俯仰区子扇形面个数
    # N:侧偏区子扇形面个数

    # 计算当前点对目标的指向角度和俯仰角度
    # me2enemy_vector = get_vector(wpt_myself, wpt_target[-1])

    # vector_1 = [me2enemy_vector[0], me2enemy_vector[1], 0]
    # vector_2 = [me2enemy_vector[0], 0, me2enemy_vector[2]]
    # axis_x = [1, 0, 0]
    # axis_z = [-1, 0, 0]

    # me2enemy_angle = vectors_angle(vector_1, axis_x)
    # me2enemy_yaw = vectors_angle(vector_2, axis_z)
    # print(me2enemy_angle)
    # print(me2enemy_yaw)

    # 应当考虑路径分段
    psi_list = [psi_max/N*i for i in range(N+1)]
    theta_list = [-theta_max/2 + theta_max/M*i for i in range(M+1)]

    nextPossible_wpts_x = []
    nextPossible_wpts_y = []
    nextPossible_wpts_z = []

    
    for i in range(M):
        for j in range(N):
            x = wptPath_minLen*math.sin(theta_list[i])*math.cos(psi_list[j]) + wpt_myself[0]
            y = wptPath_minLen*math.sin(theta_list[i])*math.sin(psi_list[j]) + wpt_myself[1]
            z = wptPath_minLen*math.cos(theta_list[i]) + wpt_myself[2]

            nextPossible_wpts_x.append(x)
            nextPossible_wpts_y.append(y)
            nextPossible_wpts_z.append(z)

    return [nextPossible_wpts_x, nextPossible_wpts_y, nextPossible_wpts_z]


# 1.2 代价函数
def cost(wpt_myself, wpt_next, enemy_points_list, omega, threat_pt, threat_radius, D_attack):
    # wpt_myself:当前点的坐标
    # wpt_next:下一点的坐标
    # enemy_points_list:目标点
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # D_attack:飞机可攻击距离

    # # 受敌机攻击的代价，视线角，单位：度
    # speed_vector_myself = get_vector(wpt_myself, wpt_next)
    # speed_vector_enemy = get_vector(enemy_points_list[-2,:], enemy_points_list[-1,:])
    # myself_to_enemy = get_vector(wpt_next, enemy_points_list[-1,:])
    # enemy_to_myself = get_vector(enemy_points_list[-1,:], wpt_next)

    # # 本机相对于敌机的视线角
    # alpha_myself = vectors_angle(speed_vector_enemy, enemy_to_myself)
    # # 敌机相对于本机的视线角
    # alpha_enemy = vectors_angle(speed_vector_myself, myself_to_enemy)

    # theta = (alpha_enemy - alpha_myself + np.pi)/(2*np.pi)

    # 受导弹攻击的代价
    # 下一点与导弹的距离
    l_threat = Euclid(wpt_next, threat_pt)

    if(threat_radius < l_threat):
        be_attack = 0
    else:
        be_attack = threat_radius/(threat_radius + l_threat)


    # 打击距离过远的代价,油量代价和被跟踪的代价

    # 油量代价主要与爬升高度有关
    dif_h = (wpt_myself[2] - wpt_next[2])
    d = Euclid(enemy_points_list[-1,:], wpt_next)

    if(d < D_attack):
        we_attack = 0.5
        be_follow = np.sqrt(wpt_myself[0]**2 + wpt_myself[1]**2)/np.sqrt(enemy_points_list[-1,:][0]**2 + enemy_points_list[-1,:][1]**2)
        if(dif_h < 0):
            fluel_cost = dif_h
        else:
            fluel_cost = 1
    else:
        we_attack = d/D_attack
        # 被敌机跟踪的代价
        be_follow = 0
        if(dif_h > 0):
            fluel_cost = -dif_h
        else:
            fluel_cost = 1

    cost_true = omega[0]*fluel_cost + omega[1]*be_attack**2 + omega[2]*we_attack**2 + omega[3]*be_follow**2

    return cost_true


# 第二部分
# 2.1 后继节点集中选择代价最小的点
def best_nextWpt(wpt_myself, enemy_points_list, nextPossible_wpts, omega, threat_pt, threat_radius, D_attack):
    # wpt_myself:当前点的坐标
    # enemy_points_list:目标点的坐标
    # nextPossible_wpts:当前点的后继节点集
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # D_attack:飞机可攻击距离

    cost_list = [cost(wpt_myself, [nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], 
                enemy_points_list, omega, threat_pt, threat_radius, D_attack) for i in range(len(nextPossible_wpts[0]))]

    best_Wpt_index = cost_list.index(min(cost_list))

    best_Wpt = [nextPossible_wpts[0][best_Wpt_index], nextPossible_wpts[1][best_Wpt_index], nextPossible_wpts[2][best_Wpt_index]]

    return best_Wpt, best_Wpt_index, min(cost_list)


# 第三部分
# 3.1 计算欧几里得距离
def Euclid(wpt, wpt_target):

    d = math.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    
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
    _range = np.max(data_list) - np.min(data_list)

    return (data_list - np.min(data_list))/_range


# 3.8 画威胁球体的函数
def ball(center, radius):

    u = np.linspace(0, 2*np.pi, 10)

    v = np.linspace(0, np.pi, 10)

    u, v = np.meshgrid(u, v)
    x = radius*np.cos(u)*np.sin(v) + center[0]
    y = radius*np.sin(u)*np.sin(v) + center[1]
    z = radius*np.cos(v) + center[2]

    return x, y, z


# 第四部分
# 4.1 规划解算函数
def simulate_pathPlanner(now_state, wpt_next, filename, tmax, step):
    # next waypoint
    waypoints = wpt_next

    ap = WaypointAutopilot(waypoints, stdout=True)

    # tmax = 150
    # step = 1/30
    extended_states = True
    res = run_f16_sim(now_state, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    if filename.endswith('.mp4'):
        skip_override = 4
    elif filename.endswith('.gif'):
        skip_override = 15
    else:
        skip_override = 30

    anim_lines = []
    modes = res['modes']
    modes = modes[0::skip_override]

    def init_extra(ax):
        'initialize plot extra shapes'

        l1, = ax.plot([], [], [], 'bo', ms=8, lw=0, zorder=50)
        anim_lines.append(l1)

        l2, = ax.plot([], [], [], 'lime', marker='o', ms=8, lw=0, zorder=50)
        anim_lines.append(l2)

        return anim_lines

    def update_extra(frame):
        'update plot extra shapes'

        # mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3']
        mode_names = ['Waypoint 1']

        done_xs = []
        done_ys = []
        done_zs = []

        blue_xs = []
        blue_ys = []
        blue_zs = []

        for i, mode_name in enumerate(mode_names):
            if modes[frame] == mode_name:
                blue_xs.append(waypoints[i][0])
                blue_ys.append(waypoints[i][1])
                blue_zs.append(waypoints[i][2])
                break

            done_xs.append(waypoints[i][0])
            done_ys.append(waypoints[i][1])
            done_zs.append(waypoints[i][2])

        anim_lines[0].set_data(blue_xs, blue_ys)
        anim_lines[0].set_3d_properties(blue_zs)

        anim_lines[1].set_data(done_xs, done_ys)
        anim_lines[1].set_3d_properties(done_zs)

    return res, init_extra, update_extra, skip_override, waypoints


# 第五部分
# 5.1 航迹的目标函数


# 第六部分
# 6.1 主函数
def main():
    # 若输入了文件名，则生成动态图
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")


    wpt_start = np.array([0, 0, 1000])   # 我方起始点
    wpt_target = np.array([-200000, -50000, 5000])   # 敌方起始点
    
    # SAS节点扩展算法参数设置
    wptPath_minLen = 15000
    psi_max = np.deg2rad(360)
    theta_max = np.deg2rad(180)
    M = 30
    N = 30

    # 航机代价权重系数
    omega = [0.1, 0.3, 0.5, 0.1]

    # 我方可攻击距离
    D = 30000

    # 威胁区的参数设置
    threat_pt = np.array([-200000, -50000, 12000])
    threat_radius = 20000
    
    # 我方飞机初始状态
    init = [500, deg2rad(2.15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]

    # 记录飞机实际到达坐标与期望坐标的误差
    error = 0
    error_list = [error]

    # 存放敌我双方点坐标，导弹坐标的数组
    # global myself_array, target_array, threat_array

    myself_array= [wpt_start]
    target_array = [wpt_target]
    target_array = np.append(target_array, [wpt_target], axis=0)
    threat_array = [threat_pt]

    # 循环次数
    iteration = 0

    # 解算时间和步长
    tmax = 30
    step = 1/20

    # 动态图参数记录
    res_list = []
    scale_list = []
    viewsize_list = []
    viewsize_z_list = []
    trail_pts_list = []
    elev_list = []
    azim_list = []
    skip_list = []
    chase_list = []
    fixed_floor_list = []
    init_extra_list = []
    update_extra_list = []

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # plt.ion()

    while(iteration <= 40):

        # 当前点下的后继点备选集
        nextPossible_wpts = sas(wpt_start, wptPath_minLen, psi_max, theta_max, M, N)

        # 获得后继点集中代价最小的点
        best_Wpt, best_Wpt_index, cost_next =  best_nextWpt(wpt_start, target_array, nextPossible_wpts, omega, threat_pt, threat_radius, D)
        
        print("最佳的下一点是：")
        print(best_Wpt)
        print("成本是：")
        print(cost_next)

        # 使飞机到达期望的点
        # res, init_extra, update_extra, skip_override, waypoints = simulate_pathPlanner(init, [best_Wpt], filename, tmax, step)

        # 更新本机当前坐标
        # init = res['states'][-1]
        # wpt_start = np.array([init[10], init[9], init[11]])
        wpt_start = np.array(best_Wpt)
        myself_array = np.append(myself_array, [wpt_start], axis=0)

        # 更新敌机坐标
        wpt_target = EnemyFly(wpt_target[0], wpt_target[1], wpt_target[2], [random.randint(-1000,15000), random.randint(-1000,15000), random.randint(-1000,2000)])
        target_array = np.append(target_array, [wpt_target], axis=0)

        # 更新导弹坐标
        threat_pt = Missile(threat_pt[0], threat_pt[1], threat_pt[2], wpt_start)
        threat_array = np.append(threat_array, [threat_pt], axis=0)

        # 动态三维图显示
        ax.plot3D(myself_array[:, 0], myself_array[:, 1], myself_array[:, 2], 'blue', linestyle='-', marker='o')
        ax.plot3D(target_array[:, 0], target_array[:, 1], target_array[:, 2], 'green', linestyle='-', marker='o')
        ax.plot3D(threat_array[:, 0], threat_array[:, 1], threat_array[:, 2], 'red', linestyle='-', marker='o')
        plt.pause(1)

        # 计算每次到达的误差
        error = Euclid(best_Wpt, wpt_start)
        error_list.append(error)
        print('到达误差：')
        print(error)

        # 计算与敌机的距离
        distance = Euclid(wpt_start, wpt_target)
        print('与敌机距离：')
        print(distance)

        # 迭代次数加一
        iteration = iteration + 1

        # 三维动态图需要记录每一段的动画参数
        # res_list.append(res)
        # scale_list.append(140)
        # viewsize_list.append(12000)
        # viewsize_z_list.append(10000)
        # trail_pts_list.append(np.inf)
        # elev_list.append(70)
        # azim_list.append(-200)
        # skip_list.append(skip_override)
        # chase_list.append(True)
        # fixed_floor_list.append(True)
        # init_extra_list.append(init_extra)
        # update_extra_list.append(update_extra)

    print("Finally reach:")
    print(wpt_start)
    plt.show()

    # 画三维动画
    # anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
    #                  trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
    #                  chase=chase_list, fixed_floor=fixed_floor_list,
    #                  init_extra=init_extra_list, update_extra=update_extra_list)

if __name__ == '__main__':
    main()
