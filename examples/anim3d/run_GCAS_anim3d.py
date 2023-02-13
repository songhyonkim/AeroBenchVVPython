'''
Stanley Bak

run the gcas system, producing a 3d animation

pass a command line argument (ending with .mp4) to save a video instead of plotting to the screen
'''

import math
import sys

import numpy as np
import pandas as pd
from numpy import deg2rad

import matplotlib.pyplot as plt

from aerobench.run_f16_sim import run_f16_sim

from aerobench.visualize import anim3d, plot

from aerobench.examples.gcas.gcas_autopilot import GcasAutopilot
from aerobench.util import SafetyLimits, SafetyLimitsVerifier

def simulate():
    'sim system and return res'

    ### Initial Conditions ###
    power = 9 # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215) # Trim Angle of Attack (rad)
    beta = 0                # Side slip angle (rad)

    # Initial Attitude
    alt = 6200        # altitude (ft)
    vt = 540          # initial velocity (ft/sec)
    phi = 0           # Roll angle from wings level (rad)
    theta = (-math.pi/2)*0.7         # Pitch angle from nose level (rad)
    psi = 0.8 * math.pi   # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 15 # simulation time

    ap = GcasAutopilot(init_mode='waiting', stdout=True)

    ap.waiting_time = 5
    ap.waiting_cmd[1] = 2.2 # ps command

    # custom gains
    ap.cfg_k_prop = 1.4
    ap.cfg_k_der = 0
    ap.cfg_eps_p = deg2rad(20)
    ap.cfg_eps_phi = deg2rad(15)

    step = 1/30
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=True, integrator_str='rk45')

    print(f"Simulation Completed in {round(res['runtime'], 2)} seconds")

    # Determine whether the GCAS system kept the plane in a safe state
    # the entire time.
    safety_limits = SafetyLimits( \
        altitude=(0, 45000), #ft \
        Nz=(-5, 18), #G's \
        v=(300, 2500), # ft/s \
        alpha=(-10, 45), # deg \
        betaMaxDeg=30,# deg
        psMaxAccelDeg=500) # deg/s/s

    verifier = SafetyLimitsVerifier(safety_limits, ap.llc)
    verifier.verify(res)

    return res

def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res = simulate()

    plot.plot_attitude(res, figsize=(12, 10))
    plt.savefig('gcas_attitude.png')
    plt.close()
    
    v = [res['states'][i][0] for i in range(len(res['states']))]
    alt = [res['states'][i][11] for i in range(len(res['states']))]

    # 输出控制指令数据到csv
    cmd_list = [res['times'], v, alt, res['throttle_list'], res['ele_list'], res['ali_list'], res['rud_list'],
     res['moment_aileron'], res['power_aileron'], res['moment_elevator'], res['power_elevator'], res['moment_rudder'], res['power_rudder']]
    cmd_array = np.array(cmd_list).T
    column = ['time', 'velocity', 'altitude', 'throttle', 'elevator', 'aileron', 'rudder', 
    'moment_aileron', 'power_aileron', 'moment_elevator', 'power_elevator', 'moment_rudder', 'power_rudder']
    
    cmd_data = pd.DataFrame(columns=column, data=cmd_array)
    cmd_data.to_csv('dive_data.csv')
    anim3d.make_anim(res, filename, elev=15, azim=-150)

    # 画出副翼铰链力矩
    plot.plot_cmd(res, 'moment_aileron', 'moment/Nm', title='Aileron Moment')
    alt_filename = 'Aileron_Moment_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    # 画出副翼舵机功率
    plot.plot_cmd(res, 'power_aileron', 'power/W', title='Aileron Power')
    alt_filename = 'Aileron_Power_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

     # 画出升降舵铰链力矩
    plot.plot_cmd(res, 'moment_elevator', 'moment/Nm', title='Elevator Moment')
    alt_filename = 'Elevator_Moment_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    # 画出升降舵舵机功率
    plot.plot_cmd(res, 'power_elevator', 'power/W', title='Elevator Power')
    alt_filename = 'Elevator_Power_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

     # 画出方向舵铰链力矩
    plot.plot_cmd(res, 'moment_rudder', 'moment/Nm', title='Rudder Moment')
    alt_filename = 'Rudder_Moment_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    # 画出方向舵舵机功率
    plot.plot_cmd(res, 'power_rudder', 'power/W', title='Rudder Power')
    alt_filename = 'Rudder_Power_dive.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

    # 输出throttle控制指令
    plot.plot_cmd(res, 'throttle_list', 'throttle', title='Throttle')
    alt_filename = 'dive_throttle.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

     # 输出elevator控制指令
    plot.plot_cmd(res, 'ele_list', 'elevator', title='Elevator')
    alt_filename = 'dive_elevator.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

     # 输出aileron控制指令
    plot.plot_cmd(res, 'ali_list', 'aileron', title='Aileron')
    alt_filename = 'dive_aileron.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

     # 输出rudder控制指令
    plot.plot_cmd(res, 'rud_list', 'rudder', title='Rudder')
    alt_filename = 'dive_rudder.png'
    plt.savefig(alt_filename)
    print(f"Made {alt_filename}")
    plt.close()

if __name__ == '__main__':
    main()
