import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from operator import itemgetter
from CostFunctions import fitness_wpt
from math import acos, sqrt, atan
from SearchSpace import SearchSpace
from Sphere import sphere

# 遗传算法的实现
class GA:

    # 参数初始化
    def __init__(self, ga_parameter, outside_info):

        # ga_parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, start, end]
        self.parameter = ga_parameter
        self.outside_info = outside_info

        # 外环境约束条件
        start = self.parameter[5]
        end = self.parameter[6]
        self.bound = [50, 50]
        

        # 随机生成初始种群，初始种群应该尽量满足约束条件
        pop = []
        for i in range(self.parameter[3]):

            # 个体
            gen_path = [start]
            for j in range(self.parameter[4]):
                # x = random.randint(gen_path[-1][0]+0.9*self.bound[0], gen_path[-1][0]+self.bound[0])
                # y = random.randint(gen_path[-1][1]+0.9*self.bound[0], gen_path[-1][1]+self.bound[0])
                # z = random.randint(max(gen_path[-1][2]-self.bound[1], 0), gen_path[-1][2]+self.bound[1])
               
                point = self.valid_point(np.array([gen_path[-1][0], gen_path[-1][1], gen_path[-1][2]]))
                gen_path = np.append(gen_path, [point], axis=0)
            gen_path = np.append(gen_path, [end], axis=0)
            gen_path = self.smooth(gen_path)
            # print(gen_path)

            # 计算每个个体的适应度
            limitness = self.wpt_limit(gen_path)
            fitness = self.evaluate(gen_path, limitness)
            
            # 以字典形式存储个体信息，形成种群
            pop.append({'Gene': gen_path, 'fitness': fitness})
            # pop.append(gen_path)

        self.pop = pop
        self.best_individual = self.selectBest(self.pop)
        

    # 适应度评价函数
    def evaluate(self, path, good_points):
        outside_info = self.outside_info
        fitness = fitness_wpt(path, good_points, outside_info[0], outside_info[1], outside_info[2])
        return fitness

    # 从种群中选择最好的个体
    def selectBest(self, population):
        # 对整个种群按照成本从小到大排序，返回成本最小的个体
        pop_sorted = sorted(population, key=itemgetter("fitness"), reverse=False)  
        return pop_sorted[0]

    # 选择
    def select(self, individuals, k):
        # 按照概率从上一代种群选择个体，直到形成新的一代

        individuals_sorted = sorted(individuals, key=itemgetter("fitness"), reverse=False)

        # 累加适应度
        sum_fitness = sum(1/individual['fitness'] for individual in individuals)

        chosen = []
        for i in range(k):
            # 随机选取一个在[0,sum_fitness]区间上的值作为判断是否选择的条件
            threshold = random.random()*sum_fitness

            ind_fitness_sum = 0
            for ind in individuals_sorted:
                ind_fitness_sum += 1/ind['fitness']

                if(ind_fitness_sum > threshold):
                    chosen.append(ind)
                    break
        
        # 成本从大到小排序，方便后面的价交叉操作
        chosen = sorted(chosen, key=itemgetter('fitness'), reverse=True)
        
        return chosen

    # 交叉
    def cross(self, offspring):
        # 实现双点交叉

        dim = len(offspring[0]['Gene'])

        # 要进行交叉的两个个体
        gen_path1 = offspring[0]['Gene']
        gen_path2 = offspring[0]['Gene']

        # 设置交叉点位
        if(dim == 1):
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randint(1,dim-2)
            pos2 = random.randint(1,dim-2)

        # 交叉后的新后代
        newoff1 = gen_path1.copy()
        newoff2 = gen_path2.copy()

        for i in range(dim):
            if min(pos1, pos2)<= i < max(pos1, pos2):
                newoff2[i,:] = gen_path2[i,:]
                newoff1[i,:] = gen_path1[i,:]
            else:
                newoff2[i,:] = gen_path1[i,:]
                newoff1[i,:] = gen_path2[i,:]
        
        return newoff1, newoff2

    # 变异
    def mutation(self, cross_offspring):
        # 变异在遗传过程中属于小概率事件此处实现单点变异

        dim = len(cross_offspring)

        if dim == 1:
            pos = 0
        else:
            pos = random.randint(1, dim-2)

        x = random.uniform(cross_offspring[pos,0]-self.bound[0], cross_offspring[pos,0]+self.bound[0])
        y = random.uniform(cross_offspring[pos,1]-self.bound[0], cross_offspring[pos,1]+self.bound[0])
        z = random.uniform(max(cross_offspring[pos,2]-self.bound[1],0), cross_offspring[pos,2]+self.bound[1])
        
        cross_offspring[pos,:] = np.array([x,y,z])

        return cross_offspring


    # 评价航迹段对约束条件的满足程度
    def wpt_limit(self, path):
        end = self.parameter[6]
        end_xy = np.array([end[0], end[1]])

        flag = False
        fly_limits = self.outside_info[3]
        counter = 0

        # 偏转角约束
        alpha_max = fly_limits[0]
        # 俯仰角约束
        beta_max = fly_limits[1]

        for i in range(1, path.shape[0]-1):
            wpt_xy = np.array([path[i,0], path[i,1]])
            wpt_back_xy = np.array([path[i-1,0], path[i-1,1]])

            vector_1 = wpt_xy - wpt_back_xy
            vector_2 = end_xy - wpt_xy
            vector = path[i+1,:] - path[i,:]

            # 计算第i个点的偏转角
            alpha_i = acos(np.dot(vector_1, vector_2)/(sqrt(vector_1[0]**2+vector_1[1]**2)*sqrt(vector_2[0]**2+vector_2[1]**2)))
            # 计算第i个点的俯仰角
            beta_i = atan(abs(path[i,2]-path[i-1,2])/sqrt(vector_1[0]**2+vector_1[1]**2))

            counter += (alpha_i/alpha_max + beta_i/beta_max)

        return counter/(path.shape[0]-1)

    # 生成满足约束的航迹点
    def valid_point(self, ex_point):

        end = self.parameter[6]
        end_xy = np.array([end[0], end[1]])

        # 约束条件
        fly_limits = self.outside_info[3]
        # 偏转角约束
        alpha_max = fly_limits[0]
        # 俯仰角约束
        beta_max = fly_limits[1]
        # 最小航迹段长度约束
        L_min = fly_limits[2]
        # 最小航迹段高度约束
        h_min = fly_limits[3]

        # 基于前一点生成备选点
        posible_points = sphere(ex_point, 1.1*L_min)
        x = posible_points[0]
        y = posible_points[1]
        z = posible_points[2]

        points = []

        # 在备选点中选择满足约束的点
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                point = np.array([x[i,j],y[i,j],z[i,j]])
                # print(point)
                point_xy = np.array([x[i,j],y[i,j]])
                wpt_back_xy = np.array([ex_point[0],ex_point[1]])
                vector_1 = point_xy - wpt_back_xy
                vector_2 = end_xy - point_xy

                if(vector_1[0]==0 and vector_1[1]==0):
                    alpha_i = 0
                    beta_i = np.pi/2
                else:
                    # 计算第i个点的偏转角
                    alpha_i = acos(np.dot(vector_1, vector_2)/(sqrt(vector_1[0]**2+vector_1[1]**2)*sqrt(vector_2[0]**2+vector_2[1]**2)))
                    # 计算第i个点的俯仰角
                    beta_i = atan(abs(point[2]-ex_point[2])/sqrt(vector_1[0]**2+vector_1[1]**2))
                    # 计算第i个点与前一点的高度差
                    dh_i = abs(point[2] - ex_point[2])
                if(alpha_i<alpha_max and beta_i<beta_max and point[2]>0 and dh_i<h_min):
                    points.append(point)
        return points[random.randint(0,len(points)-1)]

    # 航迹平滑
    def smooth(self, path):
        
        # 邻域点数
        N = self.parameter[7]
        # 总点数
        N_wpts = path.shape[0]

        # N点滑动平均平滑
        for i in range(N_wpts):
            if i == 0 or i >= N_wpts-1:
                path[i,:] =  path[i,:]
            elif i == 1:
                path[i,:] = (path[i-1,:] + path[i,:] + path[i+1,:])/3
            else:
                for j in range(i-N, i+N):
                    path[i,:] += path[j,:]
                path[i,:] = path[i,:]/(2*N+1)
        return path

    # 遗传算法执行函数
    def ga_main(self):
        N_d = self.parameter[2]
        CXPB = self.parameter[0]
        MUTPB = self.parameter[1]
        k = 10

        # print('开始进化')

        gen_best = []   # 记录每一代最好的个体
        for g in range(N_d):
            # print("目前进化到第{}代".format(g))

            # 首先在当前种群中按成本从小到大选择个体构成一个种群
            select_pop = self.select(self.pop, k)

            # 对该种群进行交叉变异操作，产生下一代种群
            nextoff = []

            while len(nextoff) != k:
                
                offspring = [select_pop.pop() for _ in range(2)]

                # 首先是交叉操作
                if random.random() < CXPB:
                    cross_off1, cross_off2 = self.cross(offspring)
                    # 变异操作
                    if random.random() < MUTPB:
                        mute_off1 = self.mutation(cross_off1)
                        mute_off2 = self.mutation(cross_off2)
                        mo1_limitness = self.wpt_limit(mute_off1)
                        mo2_limitness = self.wpt_limit(mute_off2)
                        mo1_fitness = self.evaluate(mute_off1, mo1_limitness)
                        mo2_fitness = self.evaluate(mute_off2, mo2_limitness)
                        nextoff.append({'Gene': mute_off1, 'fitness': mo1_fitness})
                        nextoff.append({'Gene': mute_off2, 'fitness': mo2_fitness})
                    else:
                        co1_limitness = self.wpt_limit(cross_off1)
                        co2_limitness = self.wpt_limit(cross_off2)
                        co1_fitness = self.evaluate(cross_off1, co1_limitness)
                        co2_fitness = self.evaluate(cross_off2, co2_limitness)
                        nextoff.append({'Gene': cross_off1, 'fitness': co1_fitness})
                        nextoff.append({'Gene': cross_off2, 'fitness': co2_fitness})
                else:
                    nextoff.extend(offspring)

            # 令新生成的种群为当代种群
            self.pop = nextoff

            # 当前成本最小的个体
            best_ind = self.selectBest(self.pop)

            if((best_ind['fitness'] < self.best_individual['fitness'])):
                self.best_individual = best_ind

            gen_best.append(self.best_individual)

            # print("当前最好的路径是：{}".format(self.best_individual['Gene']))
            # print("当前最优路径的成本：{}".format(self.best_individual['fitness']))
        
        self.gen_best = gen_best     

# 测试
def main():
    
    # 起点和终点
    start = np.array([0, 0, 0])
    end = np.array([20000, 18000, 2000])

    # 威胁区
    threat_pt1 = np.array([10000, 6000, 1000])
    threat_pt2 = np.array([15000, 17000, 1500])
    threat_pt = [threat_pt1, threat_pt2]
    threat_radius = 5000
    threat_sphere1 = sphere(threat_pt1, threat_radius)
    threat_sphere2 = sphere(threat_pt2, threat_radius)

    # 遗传算法内部参数
    N = 2
    CXPB, MUTPB, N_d, popsize, N_wpt = 0.9, 0.2, 2000, 80, 4
    ga_parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, start, end, N]

    # 飞行约束
    fly_limits = [np.pi/6, np.pi/6, 5000, 500]

    # 外环境参数
    omega = [0.1, 0.4, 0.5]
    outside_info  = [threat_pt, threat_radius, omega, fly_limits]

    # 执行遗传算法
    ga = GA(ga_parameter, outside_info)
    ga.ga_main()
    gen_best = ga.gen_best

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    

    for i in range(len(gen_best)):
        plt.cla()
        ax.scatter(start[0], start[1], start[2], 'red')
        ax.scatter(end[0], end[1], end[2], 'green')
        ax.plot3D(gen_best[i]['Gene'][:, 0], gen_best[i]['Gene'][:, 1], gen_best[i]['Gene'][:, 2], 'blue', linestyle='-', marker='o')
        ax.plot_surface(threat_sphere1[0], threat_sphere1[1], threat_sphere1[2], linewidth=0.0)
        ax.plot_surface(threat_sphere2[0], threat_sphere2[1], threat_sphere2[2], linewidth=0.0)
        cost = gen_best[i]['fitness']
        text = f'pop = {i:d}\ncost = {cost:.3f}'
        ax.set_title(text)
        plt.pause(0.0005)

    plt.show()

    

if __name__ == '__main__':
    main()