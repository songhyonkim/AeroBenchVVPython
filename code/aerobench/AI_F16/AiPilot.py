import sys
sys.path.append('./code')

import math
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
from SearchSpace import SearchSpace
from Sphere import sphere
from FunFunctions import Euclid
from aerobench.visualize import anim3d

# 规划解算函数
def simulate_pathPlanner(now_state, wpt_next, max_h, filename, tmax, step):
    # next waypoint
    waypoints = wpt_next

    ap = WaypointAutopilot(waypoints, max_h, stdout=True)

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

        # mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3', 'Waypoint 4', 'Waypoint 5', 'Waypoint 6','Waypoint 6']
        mode_names = ['Waypoint ' + str(i+1) for i in range(len(waypoints))]

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

# 主函数
def main():
    # 若输入了文件名，则生成动态图
    # if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
    #     filename = sys.argv[1]
    #     print(f"saving result to '{filename}'")
    # else:
    #     filename = ''
    #     print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")
    #     exit()

    filename = 'output.mp4'
    print(f"saving result to '{filename}'")    


    # 起点和终点
    start = np.array([2000, 5000, 3500])
    # end = np.array([42000, 40000, 5000])
    end = np.array([80000, 80000, 6000])

    # 我方飞机初始状态
    state = [400, deg2rad(2.15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 3000, 9]
    

    # 解算时间和步长
    tmax = 400
    step = 1/30

    # 威胁区
    # threat_pt1 = np.array([18000, 20000, 2000])
    # threat_pt2 = np.array([35000, 30000, 5000])
    threat_pt1 = np.array([38000, 40000, 2000])
    threat_pt2 = np.array([65000, 70000, 5000])
    threat_pt = [threat_pt1, threat_pt2]
    threat_radius = 10000
    threat_sphere1 = sphere(threat_pt1, threat_radius)
    threat_sphere2 = sphere(threat_pt2, threat_radius)
    threat_list = [threat_sphere1, threat_sphere2]

    # 遗传算法内部参数
    N = 2
    # CXPB, MUTPB, N_d, popsize, N_wpt = 0.9, 0.2, 2000, 80, 4
    CXPB, MUTPB, N_d, popsize, N_wpt = 0.9, 0.2, 2000, 80, 9
    ga_parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, start, end, N]

    # 飞行约束
    # fly_limits = [np.pi/6, np.pi/6, 6500, 400]
    fly_limits = [np.pi/6, np.pi/6, 6500, 400]

    # 外环境参数
    omega = [0.1, 0.6, 0.3]
    outside_info  = [threat_pt, threat_radius, omega, fly_limits]

    # 执行遗传算法
    ga = GaTest.GA(ga_parameter, outside_info)
    ga.ga_main()
    best_wpts = ga.best_individual['Gene']
    best_wptlist = best_wpts.tolist()
    print(best_wptlist)

    # 三维图显示并保存
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.scatter(start[0], start[1], start[2], 'g')
    ax.scatter(end[0], end[1], end[2], 'r')

    ax.plot_surface(threat_sphere1[0], threat_sphere1[1], threat_sphere1[2], linewidth=0.0)
    ax.plot_surface(threat_sphere2[0], threat_sphere2[1], threat_sphere2[2], linewidth=0.0)
    ax.plot3D(best_wpts[:, 0], best_wpts[:, 1], best_wpts[:, 2], 'blue', linestyle='-', marker='o')
    
    text = f'GA Waypoints'
    ax.set_title(text)

    plt.savefig("./GA_path.jpg")
    plt.show()
    

    # 距离尺度
    max_d = fly_limits[2]

    # 使飞机按照规划的航迹点飞行
    res, init_extra, update_extra, skip_override, waypoints = simulate_pathPlanner(state, best_wptlist, max_d, filename, tmax, step)

    # 画实际航迹图
    anim3d.make_anim(res, filename, threat_list, f16_scale=100, viewsize=10000, viewsize_z=10000, trail_pts=np.inf,
    elev=54, azim=-200, skip_frames=skip_override,
    chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)
    
    # # 迭代次数加一
    # iteration = iteration + 1


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

    # # 动态图保存
    # fig2 = plt.figure()
    # ax = fig2.add_subplot(projection='3d')
    # ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    # ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    # ax.scatter(end[0], end[1], end[2], 'r')
    # ax.plot_surface(threat_sphere[0], threat_sphere[1], threat_sphere[2], linewidth=0.0)

    # # 初始线条
    # line_myself, = ax.plot3D(myself_array[:iteration, 0], myself_array[:iteration, 0], myself_array[:iteration, 0], 'blue', linestyle='-', marker='o', animated=True)
    # # sphere_threat = ax.plot_surface(threat_sphere[0], threat_sphere[1], threat_sphere[2], linewidth=0.0)[0]

    # # 动画更新函数
    # def update(n):
    #     line_myself.set_xdata(myself_array[:n, 0])
    #     line_myself.set_ydata(myself_array[:n, 1])
    #     line_myself.set_3d_properties(myself_array[:n, 2])

    #     return line_myself

    # # 保存三维动态图
    # ani = animation.FuncAnimation(fig2, update, iteration+1, interval=200, repeat=False)
    # ani.save("aipilot.gif",writer='pillow')
    
    # 画三维动画
    # anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
    #                  trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
    #                  chase=chase_list, fixed_floor=fixed_floor_list,
    #                  init_extra=init_extra_list, update_extra=update_extra_list)

if __name__ == '__main__':
    main()
