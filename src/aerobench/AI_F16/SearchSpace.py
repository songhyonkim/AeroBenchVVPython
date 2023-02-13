import numpy as np
from math import sin,cos
from FunFunctions import plot_point

def SearchSpace(max_len, psi_max, theta_max, current_state, slice_pra):
    # max_len:最大搜索半径
    # psi_max:偏转角度范围
    # theta_max:俯仰角度范围
    # current_state:飞机当前的状态
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    # slice_pra: 均匀划分参数

    # 地面坐标系下飞机当前的坐标
    current_pos = np.array([current_state[10], current_state[9], current_state[11]])
    # 飞机的欧拉角，完成地面坐标系与机体坐标系的转换
    Phi = current_state[3]
    Theta = current_state[4]
    Psi = current_state[5]

    # 机体坐标系下的备选路径点
    r_slices = [max_len*(i+1)/slice_pra[0] for i in range(slice_pra[0])]
    psi_slices = [-psi_max/2 + psi_max*i/slice_pra[1] for i in range(slice_pra[1]+1)]
    theta_slices = [-theta_max/2 + theta_max*i/slice_pra[1] for i in range(slice_pra[1]+1)]

    base_pos = []
    for r in r_slices:
        for psi in psi_slices:
            for theta in theta_slices:
                x = r*cos(theta)*cos(psi)
                y = r*cos(theta)*sin(psi)
                z = r*sin(theta)

                base_pos.append([x,y,z])

    # 地面坐标系下的备选路径点
    # 坐标变换矩阵，从机体坐标系到地面坐标系
    rotate_matrix = np.empty([3,3], dtype=float)
    
    rotate_matrix[0,:] = [cos(Theta)*cos(Psi), cos(Theta)*sin(Psi), -sin(Theta)]
    rotate_matrix[1,:] = [sin(Theta)*sin(Phi)*cos(Psi)-cos(Phi)*cos(Psi), sin(Theta)*sin(Phi)*sin(Psi)+cos(Phi)*cos(Psi), sin(Phi)*cos(Theta)]
    rotate_matrix[2,:] = [sin(Theta)*cos(Phi)*cos(Psi)+sin(Phi)*sin(Psi), sin(Theta)*cos(Phi)*sin(Psi)-cos(Phi)*cos(Psi), cos(Phi)*cos(Theta)]
    

    groud_pos = []
    pos_num = len(base_pos)
    for i in range(pos_num):
        pos = np.dot(rotate_matrix, np.array(base_pos[i])) + current_pos
        groud_pos.append(pos)

    return np.array(groud_pos)

def main():
    state = [10, 0, 0, np.pi/6, np.pi/6, np.pi/6, 0, 0, 0, 100, 50, 20, 0]
    max_len = 30
    psi_max = np.pi/2
    theta_max = np.pi/2
    slice_pra = [3, 3, 3]

    groud_pos = SearchSpace(max_len, psi_max, theta_max, state, slice_pra)
    plot_point(groud_pos)

if __name__ == '__main__':
    main()