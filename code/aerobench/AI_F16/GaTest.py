import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

#复现遗传算法

# 第一部分：评估函数的构造
def cost(wpt_points, R_attack, Missles, Enemy, omega):

    N_wpt = len(wpt_points)
    N_missle = len(Missles)
    start_end_len = Euclid(wpt_points[0,:], wpt_points[N_wpt-1,:])

    # 1. 最小航迹长度代价
    f_l = sum([Euclid(wpt_points[i,:], wpt_points[i+1,:])/start_end_len for i in range(N_wpt-1)])

    # 2. 最小被导弹攻击风险
    f_k = 0
    for i in range(1, N_wpt-1):
        for j in range(N_missle):
            d_ij = Euclid(wpt_points[i,:], Missles[j,:])
            if(d_ij > R_attack):
                RK_ij = 0
            else:
                RK_ij = R_attack**4/(R_attack**4 + d_ij**4)
            
            f_k = f_k + RK_ij
            
    # 3. 最小被敌机攻击劣势，受敌机攻击的代价，视线角，单位：弧度
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

    return omega[0]*f_l**2 + omega[1]*f_k**2 + omega[2]*f_angle**2


# 第三部分：遗传算法的实现
class GA:

    # 参数初始化
    def __init__(self, ga_parameter, outside_info) -> None:

        # ga_parameter = [CXPB, MUTPB, N_d, popsize, start, end]
        self.parameter = ga_parameter
        # 约束条件
        self.bound = [[0,0],[0,0],[0,0]]
        start = self.parameter[4]
        end = self.parameter[5]

        for i in range(3):
            self.bound[i][0] = min(start[i], end[i])
            self.bound[i][1] = max(start[i], end[i])

        # 随机生成初始种群
        pop = []
        for i in range(self.parameter[3]):

            # 个体
            gen_path = [np.array(start)]
            for j in range(self.parameter[2]):
                x = random.randint(self.bound[0][0], self.bound[0][1])
                y = random.randint(self.bound[1][0], self.bound[1][1])
                z = random.randint(self.bound[2][0], self.bound[2][1])
                gen_path = np.append(gen_path, [np.array([x,y,z])], axis=0)
            
            # 计算每个个体的适应度
            fitness = self.evaluate(gen_path, outside_info)

            # 以字典形式存储个体信息，形成种群
            pop.append({'个体': gen_path, '适应度': fitness})
            # pop.append(gen_path)

        self.pop = pop
        self.best_individual = self.selectBest(self.pop)
        pass

    # 适应度评价函数
    def evaluate(self, path, outside_info):
        # outside_info = [R_attack, Missles, Enemy, omega]
        return cost(path, outside_info[0], outside_info[1], outside_info[2], outside_info[3])

    # 从种群中选择最好的个体
    def selectBest(self, pop):
        return 

    # 选择
    def select(self, individuals, k):
        return

    # 交叉
    def cross(self, offspring):
        return 

    # 变异
    def mutation(self, cross_offspring, bound):
        return 

    # 遗传算法执行函数
    def ga_main(self):
        return

# 第四部分：功能函数
# 4.1 计算欧几里得距离
def Euclid(wpt, wpt_target):

    d = math.sqrt((wpt[0] - wpt_target[0])**2 + (wpt[1] - wpt_target[1])**2 + (wpt[2] - wpt_target[2])**2)
    
    return d

# 4.2 速度矢量计算函数
def get_vector(point_a, point_b):

    vector_x = point_b[0] - point_a[0]
    vector_y = point_b[1] - point_a[1]
    vector_z = point_b[2] - point_a[2]

    return [vector_x, vector_y, vector_z]


# 4.3 矢量夹角计算函数
def vectors_angle(vector_a, vector_b):

    length_a = math.sqrt(vector_a[0]**2 + vector_a[1]**2 + vector_a[2]**2)
    length_b = math.sqrt(vector_b[0]**2 + vector_b[1]**2 + vector_b[2]**2)
    angle = sum(np.multiply(vector_a, vector_b))/(length_a * length_b)

    return angle


# 4.4 敌机飞行函数
def EnemyFly(x, y, z, speed):
    # x,y,z：敌机当前的坐标
    # 输出：下一时刻敌机的坐标

    x_next = x + speed[0]
    y_next = y + speed[1]
    z_next = z + speed[2]

    return np.array([x_next, y_next, z_next])


# 4.5 导弹飞行函数
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


# 第五部分：测试
def main():
    CXPB, MUTPB, N_d, popsize = 0.8, 0.1, 5, 50

    start = np.array([0, 10, 200])
    end = np.array([-100, 30, 70])

    ga_parameter = [CXPB, MUTPB, N_d, popsize, start, end]
    outside_info  = 0

    ga = GA(ga_parameter, outside_info)

    pop = ga.pop

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    for i in range(popsize):
        ax.plot3D(pop[i][:, 0], pop[i][:, 1], pop[i][:, 2], 'blue', linestyle='-', marker='o')
        plt.pause(1)

    plt.show()

if __name__ == '__main__':
    main()