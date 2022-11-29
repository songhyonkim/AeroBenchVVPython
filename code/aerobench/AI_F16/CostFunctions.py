import numpy as np
from FunFunctions import Euclid, get_vector, vectors_angle, normalization

# 第二部分
# 实现航迹点评价函数和路径点评价函数

# 2.1 航迹点评价函数
def fitness_wpt(wpt_points, R_attack, Missles, Enemy, omega):

    N_wpt = len(wpt_points)
    N_missle = len(Missles)
    start_end_len = Euclid(wpt_points[0,:], wpt_points[N_wpt-1,:])
    dl_list = [Euclid(wpt_points[i,:], wpt_points[i+1,:]) for i in range(N_wpt-1)]

    # 1. 最小航迹长度代价
    f_l = sum(dl_list)/start_end_len

    # 2. 最小航迹迂回代价
    f_v = sum([Euclid(wpt_points[i+1,:], Enemy[-1,:])/Euclid(wpt_points[i,:], Enemy[-1,:]) for i in range(N_wpt-1)])


    # 3. 最小被导弹攻击风险
    f_k = 0
    for i in range(1, N_wpt-1):
        for j in range(N_missle):
            d_ij = Euclid(wpt_points[i,:], Missles[j][-1,:])
            if(d_ij > R_attack):
                RK_ij = 0
            else:
                RK_ij = R_attack/(R_attack + d_ij)
            
            f_k = f_k + RK_ij

    # 4. 最小航迹段方差代价
    f_var = np.var(dl_list)/max(dl_list)**2
            
    # # 4. 最小被敌机攻击劣势，受敌机攻击的代价，视线角，单位：弧度
    # f_angle = 0
    # for i in range(N_wpt-1):

    #     speed_vector_myself = get_vector(wpt_points[i,:], wpt_points[i+1,:])
    #     speed_vector_enemy = get_vector(Enemy[-2,:], Enemy[-1,:])
    #     myself_to_enemy = get_vector(wpt_points[i+1,:], Enemy[-1,:])
    #     enemy_to_myself = get_vector(Enemy[-1,:], wpt_points[i+1,:])

    #     # 本机相对于敌机的视线角
    #     alpha_myself = vectors_angle(speed_vector_enemy, enemy_to_myself)
    #     # 敌机相对于本机的视线角
    #     alpha_enemy = vectors_angle(speed_vector_myself, myself_to_enemy)

    #     theta = (alpha_enemy - alpha_myself + np.pi)/(2*np.pi)

    #     f_angle = f_angle + theta


    return omega[0]*f_l**2 + omega[1]*f_v**2 + omega[2]*f_k**2 + omega[3]*f_var**2


# 2.2 路径点评价函数
def cost(wpt_myself, next_wpts, enemy_points_list, omega, threat_pt, threat_radius, D_attack):

    be_attacked = []   # 受导弹攻击的代价
    we_attack = []     # 我方打击距离过远的代价
    be_followed = []   # 被敌机跟踪的代价
    h_cost = []        # 飞行高度的代价

    for i in range(len(next_wpts)):

        # 1.受导弹攻击的代价
        l_threat = Euclid(next_wpts[i], threat_pt)

        if(threat_radius < l_threat):
            be_attack = 0
        else:
            be_attack = threat_radius/(threat_radius + l_threat)
        be_attacked.append(be_attack)


        # 2.我方打击距离过远的代价 && 3. 被敌机跟踪的代价
        d = Euclid(enemy_points_list[-1,:], next_wpts[i])

        if(d < D_attack):
            we_attack_pb = 0.5
            # 被敌机跟踪的代价
            be_follow = np.sqrt(next_wpts[i][0]**2 + next_wpts[i][1]**2)/np.sqrt(enemy_points_list[-1,:][0]**2 + enemy_points_list[-1,:][1]**2)
        else:
            we_attack_pb = d/D_attack
            be_follow = 0

        we_attack.append(we_attack_pb)
        be_followed.append(be_follow)
        
        # 4. 飞行高度的代价
        if wpt_myself[2] != next_wpts[i][2]:
            h_cost.append(abs(next_wpts[i][2] - enemy_points_list[-1,2])/abs(wpt_myself[2] - next_wpts[i][2]))
        else:
            h_cost.append(5)

    # 归一化处理
    be_attacked = normalization(be_attacked)
    we_attack = normalization(we_attack)
    be_followed = normalization(be_followed)
    h_cost = normalization(h_cost)

    cost_true = [omega[0]*a + omega[1]*b + omega[2]*c + omega[3]*d for a,b,c,d in zip(be_attacked, we_attack, be_followed, h_cost)]

    return cost_true