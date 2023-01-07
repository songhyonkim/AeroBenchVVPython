import numpy as np
from FunFunctions import Euclid, get_vector, vectors_angle, normalization

# 第二部分
# 实现航迹点评价函数和路径点评价函数

# 2.1 航迹点评价函数
def fitness_wpt(wpt_points, good_points, threat_p, threat_r, omega):

    # 航迹段中的点数
    N_wpt = len(wpt_points)
    # 航迹段的直线距离
    wpt_len = Euclid(wpt_points[0,:], wpt_points[N_wpt-1,:])
    # 航迹段中相邻点的距离
    dl_list = [Euclid(wpt_points[i,:], wpt_points[i+1,:])/wpt_len for i in range(N_wpt-1)]
    # 航迹段中各点受威胁程度
    dt_list = []
    for i in range(N_wpt-1):
        t = 0
        for j in range(len(threat_p)):
            t+=threat_r/(Euclid(wpt_points[i,:], threat_p[j])+threat_r)
        dt_list.append(t)

    # 1. 航迹长度代价
    f_l = sum(dl_list)

    # 2. 被威胁代价
    f_t = sum(dt_list)

    # 3. 约束条件代价
    f_c = good_points

    return omega[0]*f_l**2 + omega[1]*f_t**2 + omega[2]*f_c**2


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