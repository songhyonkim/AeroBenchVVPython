import math
import sys
import random
import numpy as np
from numpy import deg2rad
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import matplotlib.animation as animation
from aerobench.run_f16_sim import run_f16_sim
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
import GaTest
from FunFunctions import Euclid, Missile, EnemyFly
from CostFunctions import cost

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
    theta_list = [theta_max/M*i for i in range(M+1)]

    nextPossible_wpts = []

    
    for i in range(M):
        for j in range(N):
            x = wptPath_minLen*math.sin(theta_list[i])*math.cos(psi_list[j]) + wpt_myself[0]
            y = wptPath_minLen*math.sin(theta_list[i])*math.sin(psi_list[j]) + wpt_myself[1]
            z = wptPath_minLen*math.cos(theta_list[i]) + wpt_myself[2]

            nextPossible_wpts.append([x,y,z])

    return nextPossible_wpts

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

    cost_list = cost(wpt_myself, nextPossible_wpts, enemy_points_list, omega, threat_pt, threat_radius, D_attack)

    best_Wpt_index = cost_list.index(min(cost_list))

    best_Wpt = nextPossible_wpts[best_Wpt_index]

    return best_Wpt, best_Wpt_index, min(cost_list)


# 第三部分
# 3.1 规划解算函数
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


# 第四部分
# 4.1 主函数
def main():
    # 若输入了文件名，则生成动态图
    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")


    wpt_start = np.array([0, 0, 1000])   # 我方起始点
    wpt_target = np.array([200000, 50000, 5000])   # 敌方起始点
    
    # SAS节点扩展算法参数设置
    wptPath_minLen = 5000   
    psi_max = np.deg2rad(360)
    theta_max = np.deg2rad(180)
    M = 30
    N = 30

    # 路径节点代价权重系数
    omega1 = [0.3, 0.3, 0.2, 0.2]

    # 我方可攻击距离
    distance = D = 30000

    # 威胁区的参数设置
    threat_pt = np.array([200000, 50000, 12000])
    threat_radius = 20000
    
    # 我方飞机初始状态
    init = [1500, deg2rad(2.15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]

    # 记录飞机实际到达坐标与期望坐标的误差
    error = 0
    error_list = [error]

    # 存放敌我双方点坐标，导弹坐标的数组
    # global myself_array, target_array, threat_array

    myself_array= [wpt_start]
    target_array = 5000*np.ones((1,3))
    target_array = np.append(target_array, [wpt_target], axis=0)
    threat_array = np.array([threat_pt])
    

    # 循环次数
    iteration = 0

    # 解算时间和步长
    tmax = 30
    step = 1/10

    # 遗传算法部分固定参数
    CXPB, MUTPB, N_d, popsize, N_wpt = 0.9, 0.2, 2000, 50, 2
    omega2 = [0.3, 0.5, 0.1, 0.1]

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
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    # 添加图例
    y = np.arange(1, 4, 1)
    y_unique = np.unique(y-1)   # 可以看作图例类型个数
    color = ['b', 'g', 'r']  # 颜色集
    types = ('Myself', 'Enemy', 'Missle')  # 图例说明集
    legend_lines = [mpl.lines.Line2D([0], [0], linestyle="-", marker='o', c=color[y]) for y in y_unique]
    legend_labels = [types[y] for y in y_unique]
    ax.legend(legend_lines, legend_labels, numpoints=1, title='Type')


    # plt.ion()

    while(1):

        # 当前点下的后继点备选集
        nextPossible_wpts = sas(wpt_start, wptPath_minLen, psi_max, theta_max, M, N)

        # 获得后继点集中代价最小的点
        best_Wpt, best_Wpt_index, cost =  best_nextWpt(wpt_start, target_array, nextPossible_wpts, omega1, threat_pt, threat_radius, D)
        
        print("最佳的下一点是：")
        print(best_Wpt)
        print("成本是：")
        print(cost)

        # 用遗传算法规划航迹点
        parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, wpt_start, best_Wpt]
        outside_info = [D, [threat_array], target_array, omega2]

        ga = GaTest.GA(parameter, outside_info)
        ga.ga_main()

        best_wpts = ga.best_individual['Gene'].tolist()
        print("飞到下一点的路径是：")
        print(best_wpts)

        # 使飞机按照规划的航迹点飞行
        res, init_extra, update_extra, skip_override, waypoints = simulate_pathPlanner(init, best_wpts, filename, tmax, step)

        # 更新本机当前坐标
        init = res['states'][-1]
        wpt_start = np.array([init[10], init[9], init[11]])
        myself_array = np.append(myself_array, [wpt_start], axis=0)

        # 更新敌机坐标
        wpt_target = EnemyFly(wpt_target[0], wpt_target[1], wpt_target[2], [random.randint(-4000,1000), random.randint(-1000,4000), random.randint(-500,1000)])
        target_array = np.append(target_array, [wpt_target], axis=0)

        # 更新导弹坐标
        threat_pt = Missile(threat_pt[0], threat_pt[1], threat_pt[2], wpt_start)
        threat_array = np.append(threat_array, [threat_pt], axis=0)

        # 计算与敌机的距离
        distance = Euclid(wpt_start, wpt_target)
        print('与敌机距离：')
        print(distance)

        # 动态三维图显示
        ax.plot3D(myself_array[:, 0], myself_array[:, 1], myself_array[:, 2], 'blue', linestyle='-', marker='o')
        ax.plot3D(target_array[1:, 0], target_array[1:, 1], target_array[1:, 2], 'green', linestyle='-', marker='o')
        ax.plot3D(threat_array[:, 0], threat_array[:, 1], threat_array[:, 2], 'red', linestyle='-', marker='o')
        text = f'distance between my F16 and enemy : {distance:.3f}'
        ax.set_title(text)
        plt.pause(0.1)
       
        # 迭代次数加一
        iteration = iteration + 1

        # 计算每次到达的误差
        print("当前飞到第{}个点{}".format(iteration, wpt_start))
        error = Euclid(best_Wpt, wpt_start)
        error_list.append(error)
        print('到达误差：')
        print(error)

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

        if distance <= D and (wpt_start[2] >= wpt_target[2]):
            break

    print("Finally reach:")
    print(wpt_start)
    plt.show()

    # 初始线条
    # line_myself, = ax.plot3D(myself_array[:iteration, 0], myself_array[:iteration, 0], myself_array[:iteration, 0], 'blue', linestyle='-', marker='o', animated=True)
    # line_target = ax.plot(target_array[1:iteration, 0], target_array[1:iteration, 1], target_array[1:iteration, 2],'green', linestyle='-', marker='o', animated=True)[0]
    # line_threat = ax.plot(threat_array[:iteration, 0], threat_array[:iteration, 0], threat_array[:iteration, 0], 'red', linestyle='-', marker='o', animated=True)[0]

    # 动画更新函数
    def update(n):
        line_myself.set_xdata(myself_array[:n, 0])
        line_myself.set_ydata(myself_array[:n, 1])
        line_myself.set_3d_properties(myself_array[:n, 2])

        line_target.set_xdata(target_array[1:n, 0])
        line_target.set_ydata(target_array[1:n, 1])
        line_target.set_3d_properties(target_array[1:n, 2])

        line_threat.set_xdata(threat_array[:n, 0])
        line_threat.set_ydata(threat_array[:n, 1])
        line_threat.set_3d_properties(threat_array[:n, 2])

        return line_myself

    # 保存三维动态图
    # ani = animation.FuncAnimation(fig, update, iteration, interval=200, repeat=False)
    # ani.save("aipilot.gif",writer='pillow')
    

    # 画三维动画
    # anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
    #                  trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
    #                  chase=chase_list, fixed_floor=fixed_floor_list,
    #                  init_extra=init_extra_list, update_extra=update_extra_list)

if __name__ == '__main__':
    main()
