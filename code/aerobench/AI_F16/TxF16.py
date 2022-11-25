'''
Stanley Bak

plots 3d animation for 'u_turn' scenario 
'''

from ipaddress import collapse_addresses
import sys
import csv
import pandas as pd
import numpy as np
from numpy import deg2rad

import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot
from aerobench.AI_F16.waypointPlanner import sas, u, best_nextWpt, ball, EnemyFly, Missile


def simulate(filename):
    'simulate the system, returning waypoints, res'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 1500        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = 0         # Pitch angle from nose level (rad)
    psi = 0           # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 150 # simulation time

    # make waypoint list
    waypoints = [[-5000, -7500, alt],
                 [-15000, -7500, alt-500],
                 [-15000, 5000, alt-200]]

    ap = WaypointAutopilot(waypoints, stdout=True)

    step = 1/30
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

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

        mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3']

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


# 一次节点扩展，一次生成一段航迹
def simulate_pathPlanner(now_state, wpt_next, filename):
    # next waypoint
    waypoints = wpt_next

    ap = WaypointAutopilot(waypoints, stdout=True)

    tmax = 150
    step = 1/30
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


def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")


    # 航迹规划算法实现

    # 初始条件设置
    wpt_start = [0, 0, 1000]
    wpt_target = [-200000, -50000, 12000]
    target_list = [wpt_target]

    wptPath_minLen = 15000
    psi_max = np.deg2rad(360)
    theta_max = np.deg2rad(180)
    M = 30
    N = 30

    omega = [0.25, 0.5, 0.25]
    threat_pt = [-200000, -50000, 12000]
    threat_radius = 1500
    threat_coef = [1.1, 1.2]
    D = 1.2

    init = [1500, deg2rad(2.15), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9]

    error = u(wpt_start, wpt_target, 1)
    error_list = [error]
    iteration = 0

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

    # 绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(wpt_start[0], wpt_start[1], wpt_start[2], c='b')
    ax.scatter(wpt_target[0], wpt_target[1], wpt_target[2], c='g')
    x, y, z = ball(threat_pt, threat_radius)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='black')

    while(iteration <= 100):

        # 当前航迹点下的后继航迹点集
        nextPossible_wpts = sas(wpt_start, wpt_target, wptPath_minLen, psi_max, theta_max, M, N)

        # 获得后继航迹点集中航迹代价最小的航迹点
        best_Wpt, best_Wpt_index, u_list =  best_nextWpt(wpt_start, target_list, nextPossible_wpts, omega, threat_pt, threat_radius, threat_coef, D)
        
        # print("下一航迹点：")
        # print(best_Wpt)

        res, init_extra, update_extra, skip_override, waypoints = simulate_pathPlanner(init, [best_Wpt], filename)

        # 更新航迹点，目标和导弹的坐标
        init = res['states'][-1]
        wpt_start = [init[10], init[9], init[11]]

        wpt_target = EnemyFly(wpt_target[0], wpt_target[1], wpt_target[2])
        target_list.append(wpt_target)

        threat_pt = Missile(threat_pt[0], threat_pt[1], threat_pt[2])
        x, y, z = ball(threat_pt, threat_radius)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color='black')
        ax.scatter(wpt_start[0], wpt_start[1], wpt_start[2], c='b')
        ax.scatter(wpt_target[0], wpt_target[1], wpt_target[2], c='g')

        # 记录每一段航迹的动画参数
        res_list.append(res)
        scale_list.append(140)
        viewsize_list.append(12000)
        viewsize_z_list.append(10000)
        trail_pts_list.append(np.inf)
        elev_list.append(70)
        azim_list.append(-200)
        skip_list.append(skip_override)
        chase_list.append(True)
        fixed_floor_list.append(True)
        init_extra_list.append(init_extra)
        update_extra_list.append(update_extra)
        
        error = u(wpt_start, wpt_target, 1)
        error_list.append(error)
        print(error)

        iteration = iteration + 1

    print("Finally reach:")
    print(wpt_start)
    
    # 添加坐标轴(顺序是X,Y,Z)
    ax.set_xlim(-210000, 0)
    ax.set_ylim(-210000, 0)
    ax.set_zlim(-30000, 210000)
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    plt.show()


    # 画三维动画
    # anim3d.make_anim(res_list, filename, f16_scale=scale_list, viewsize=viewsize_list, viewsize_z=viewsize_z_list,
    #                  trail_pts=trail_pts_list, elev=elev_list, azim=azim_list, skip_frames=skip_list,
    #                  chase=chase_list, fixed_floor=fixed_floor_list,
    #                  init_extra=init_extra_list, update_extra=update_extra_list)

   
if __name__ == '__main__':
    main()
