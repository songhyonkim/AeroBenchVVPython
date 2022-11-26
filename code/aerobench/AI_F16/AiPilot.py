import math
import sys
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot


# 第一部分
# 1.1 航迹节点扩展函数--SAS搜索算法
# 算法目的：在规划空间中生成备选节点集
def sas(wpt_myself, wptPath_minLen, psi_max, theta_max, M, N):
    # wpt_myself:当前航迹点坐标
    # wpt_target:目标航迹点坐标
    # wptPath_minLen:最短航迹路径距离
    # psi_max:飞机最大侧偏角
    # theta_max:飞机最大爬升/俯冲角
    # M:俯仰区子扇形面个数
    # N:侧偏区子扇形面个数

    # 计算当前航迹点对目标的指向角度和俯仰角度
    # me2enemy_vector = get_vector(wpt_myself, wpt_target[-1])

    # vector_1 = [me2enemy_vector[0], me2enemy_vector[1], 0]
    # vector_2 = [me2enemy_vector[0], 0, me2enemy_vector[2]]
    # axis_x = [1, 0, 0]
    # axis_z = [-1, 0, 0]

    # me2enemy_angle = vectors_angle(vector_1, axis_x)
    # me2enemy_yaw = vectors_angle(vector_2, axis_z)
    # print(me2enemy_angle)
    # print(me2enemy_yaw)

    # 应当考虑航迹路径分段
    # epsilon = np.pi/20
    psi_list = [psi_max/N*i for i in range(N+1)]
    theta_list = [-theta_max/2 + theta_max/M*i for i in range(M+1)]
    # radius_list = [wptPath_minLen/K*(i+1) for i in range(K)]

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


# 1.2 航迹代价函数
def cost(wpt_myself, wpt_next, max_len, enemy_points_list, omega, threat_pt, threat_radius, threat_coef):
    # wpt_myself:当前航迹点的坐标
    # wpt_next:下一航迹点的坐标
    # enemy_points_list:目标航迹点
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数


    # 航段长度代价，油量代价主要与飞行航程相关
    l = Euclid(wpt_myself, wpt_next)/max_len  # 归一化
    
    # 受敌机攻击的代价，视线角，单位：度
    speed_vector_myself = get_vector(wpt_myself, wpt_next)
    speed_vector_enemy = get_vector(enemy_points_list[-2,:], enemy_points_list[-1,:])
    myself_to_enemy = get_vector(wpt_next, enemy_points_list[-1,:])
    enemy_to_myself = get_vector(enemy_points_list[-1,:], wpt_next)

    # 本机相对于敌机的视线角
    alpha_myself = vectors_angle(speed_vector_enemy, enemy_to_myself)
    # 敌机相对于本机的视线角
    alpha_enemy = vectors_angle(speed_vector_myself, myself_to_enemy)

    theta = (alpha_enemy - alpha_myself + np.pi)/(2*np.pi)

    # 受导弹攻击的代价
    # 下一航迹点与导弹的距离
    l_threat = Euclid(wpt_next, threat_pt)

    if(threat_coef[0]*threat_coef[1]*threat_radius < l_threat):
        f_attack = 0
    else:
        f_attack = threat_coef[0]*threat_coef[1]*threat_radius/l_threat

    cost_true = omega[0]*l**2 + omega[1]*theta**2 + omega[2]*f_attack**2

    return cost_true


# 第二部分
# 2.1 后继节点集中选择航迹代价最小的航迹点
def best_nextWpt(wpt_myself, min_len, enemy_points_list, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D):
    # wpt_myself:当前航迹点的坐标
    # enemy_points_list:目标航迹点的坐标
    # nextPossible_wpts:当前航迹点的后继节点集
    # omega:权重系数
    # threat_pt:威胁点的坐标信息
    # threat_radius:威胁半径
    # threat_coef:威胁系数

    cost_list = [cost(wpt_myself, [nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], min_len, 
                enemy_points_list, omega, threat_pt, threat_radius, threat_coef) for i in range(len(nextPossible_wpts[0]))]
    u_list  = [u([nextPossible_wpts[0][i], nextPossible_wpts[1][i], nextPossible_wpts[2][i]], enemy_points_list[-1,:], D)
                for i in range(len(nextPossible_wpts[0]))]

    f_list = np.sum([cost_list, u_list], axis=0).tolist()

    best_Wpt_index = f_list.index(min(f_list))

    best_Wpt = [nextPossible_wpts[0][best_Wpt_index], nextPossible_wpts[1][best_Wpt_index], nextPossible_wpts[2][best_Wpt_index]]

    return best_Wpt, best_Wpt_index, min(f_list)


# 第三部分
# 3.1 计算欧几里得距离
def Euclid(wpt, wpt_target):

    d = math.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    
    return d

# 3.2 启发函数
def u(wpt, wpt_target, D):

    # 我机与敌机距离
    d = Euclid(wpt, wpt_target)

    if(d>D):
        return 0.15*(d/D - 1)**2
    else:
        return 0.15*(D/d - 1)**2

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
# 4.1 航迹规划解算函数
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
# 5.1 主函数
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
    
    # SAS航迹节点扩展算法参数设置
    wptPath_minLen = 5000
    psi_max = np.deg2rad(360)
    theta_max = np.deg2rad(180)
    M = 30
    N = 30

    # 航机代价权重系数
    omega = [0.3, 0.5, 0.2]

    # 我方可攻击距离
    D = 30000

    # 威胁区的参数设置
    threat_pt = np.array([-200000, -50000, 12000])
    threat_radius = 5000
    threat_coef = [1.1, 1.2]
    
    # 我方飞机初始状态
    init = [500, deg2rad(2.15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]

    # 记录飞机实际到达坐标与期望坐标的误差
    error = 0
    error_list = [error]

    # 存放敌我双方航迹点坐标，导弹坐标的数组
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
    plt.ion()

    while(iteration <= 100):

        # 当前航迹点下的后继航迹点备选集
        nextPossible_wpts = sas(wpt_start, wptPath_minLen, psi_max, theta_max, M, N)

        # 获得后继航迹点集中航迹代价最小的航迹点
        best_Wpt, best_Wpt_index, cost_next =  best_nextWpt(wpt_start, wptPath_minLen, target_array, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D)
        
        print("最佳的下一航迹点是：")
        print(best_Wpt)

        # 使飞机到达期望的航迹点
        res, init_extra, update_extra, skip_override, waypoints = simulate_pathPlanner(init, [best_Wpt], filename, tmax, step)

        # 更新本机当前坐标
        init = res['states'][-1]
        wpt_start = np.array([init[10], init[9], init[11]])
        myself_array = np.append(myself_array, [wpt_start], axis=0)
        print("到达的坐标：")
        print(wpt_start)

        # 更新敌机坐标
        wpt_target = EnemyFly(wpt_target[0], wpt_target[1], wpt_target[2], [1000, 500, 0])
        target_array = np.append(target_array, [wpt_target], axis=0)

        # 更新导弹坐标
        threat_pt = Missile(threat_pt[0], threat_pt[1], threat_pt[2], wpt_start)
        threat_array = np.append(threat_array, [threat_pt], axis=0)

        # 动态三维图显示
        ax.plot3D(myself_array[:, 0], myself_array[:, 1], myself_array[:, 2], 'blue')
        ax.plot3D(target_array[:, 0], target_array[:, 1], target_array[:, 2], 'green')
        ax.plot3D(threat_array[:, 0], threat_array[:, 1], threat_array[:, 2], 'red')
        plt.pause(0.001)

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

        # 三维动态图需要记录每一段航迹的动画参数
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

    # 画三维动画
    # anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
    #                  trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
    #                  chase=chase_list, fixed_floor=fixed_floor_list,
    #                  init_extra=init_extra_list, update_extra=update_extra_list)

if __name__ == '__main__':
    main()
