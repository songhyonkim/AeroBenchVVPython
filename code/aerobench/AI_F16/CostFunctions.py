import numpy as np
from FunFunctions import Euclid, get_vector, vectors_angle

# 第二部分
# 实现航迹点评价函数和路径点评价函数

# 2.1 航迹点评价函数
def fitness_wpt(wpt_points, R_attack, Missles, Enemy, omega):

    N_wpt = len(wpt_points)
    N_missle = len(Missles)
    start_end_len = Euclid(wpt_points[0,:], wpt_points[N_wpt-1,:])

    # 1. 最小航迹长度代价
    f_l = sum([Euclid(wpt_points[i,:], wpt_points[i+1,:])/start_end_len for i in range(N_wpt-1)])

    # 2. 最小航迹段方差代价
    f_v = np.var([Euclid(wpt_points[i,:], wpt_points[i+1,:]) for i in range(N_wpt-1)])/start_end_len

    # 3. 最小被导弹攻击风险
    f_k = 0
    for i in range(1, N_wpt-1):
        for j in range(N_missle):
            d_ij = Euclid(wpt_points[i,:], Missles[j,:])
            if(d_ij > R_attack):
                RK_ij = 0
            else:
                RK_ij = R_attack**4/(R_attack**4 + d_ij**4)
            
            f_k = f_k + RK_ij
            
    # 4. 最小被敌机攻击劣势，受敌机攻击的代价，视线角，单位：弧度
    f_angle = 0
    for i in range(N_wpt-1):

        speed_vector_myself = get_vector(wpt_points[i,:], wpt_points[i+1,:])
        speed_vector_enemy = get_vector(Enemy[-2,:], Enemy[-1,:])
        myself_to_enemy = get_vector(wpt_points[i+1,:], Enemy[-1,:])
        enemy_to_myself = get_vector(Enemy[-1,:], wpt_points[i+1,:])

        # 本机相对于敌机的视线角
        alpha_myself = vectors_angle(speed_vector_enemy, enemy_to_myself)
        # 敌机相对于本机的视线角
        alpha_enemy = vectors_angle(speed_vector_myself, myself_to_enemy)

        theta = (alpha_enemy - alpha_myself + np.pi)/(2*np.pi)

        f_angle = f_angle + theta

    return omega[0]*f_l**2 + omega[1]*f_v**2 + omega[2]*f_k**2 + omega[3]*f_angle**2


# 2.2 路径点评价函数
def cost(wpt_myself, wpt_next, enemy_points_list, omega, threat_pt, threat_radius, D_attack):

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