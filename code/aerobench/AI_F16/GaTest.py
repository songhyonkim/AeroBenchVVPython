import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from operator import itemgetter
from CostFunctions import fitness_wpt

# 遗传算法的实现
class GA:

    # 参数初始化
    def __init__(self, ga_parameter, outside_info):

        # ga_parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, start, end]
        self.parameter = ga_parameter
        self.outside_info = outside_info

        # 约束条件(有待优化)
        self.bound = [[0,0],[0,0],[0,0]]
        start = self.parameter[5]
        end = self.parameter[6]

        for i in range(3):
            self.bound[i][0] = round(min(start[i], end[i]))
            self.bound[i][1] = round(max(start[i], end[i]))

        # 随机生成初始种群
        pop = []
        for i in range(self.parameter[3]):

            # 个体
            gen_path = [start]
            for j in range(self.parameter[4]):
                
                x = random.randint(self.bound[0][0], self.bound[0][1])
                y = random.randint(self.bound[1][0], self.bound[1][1])
                z = random.randint(self.bound[2][0], self.bound[2][1])

                gen_path = np.append(gen_path, [np.array([x,y,z])], axis=0)
            
            gen_path = np.append(gen_path, [end], axis=0)
            
            # 计算每个个体的适应度
            fitness = self.evaluate(gen_path)
            
            # 以字典形式存储个体信息，形成种群
            pop.append({'Gene': gen_path, 'fitness': fitness})
            # pop.append(gen_path)

        self.pop = pop
        self.best_individual = self.selectBest(self.pop)
        

    # 适应度评价函数
    def evaluate(self, path):
        # outside_info = [R_attack, Missles, Enemy, omega]
        outside_info = self.outside_info
        fitness = fitness_wpt(path, outside_info[0], outside_info[1], outside_info[2], outside_info[3])
        return fitness

    # 从种群中选择最好的个体
    def selectBest(self, population):
        # 对整个种群按照成本从小到大排序，返回成本最小的个体
        pop_sorted = sorted(population, key=itemgetter("fitness"), reverse=False)  
        # print(pop_sorted)
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
    def mutation(self, cross_offspring, bound):
        # 变异在遗传过程中属于小概率事件此处实现单点变异

        dim = len(cross_offspring)

        if dim == 1:
            pos = 0
        else:
            pos = random.randint(1, dim-2)

        x = random.randint(bound[0][0], bound[0][1])
        y = random.randint(bound[1][0], bound[1][1])
        z = random.randint(bound[2][0], bound[2][1])

        cross_offspring[pos,:] = np.array([x,y,z])

        return cross_offspring

    # 遗传算法执行函数
    def ga_main(self):
        popsize = self.parameter[3]
        N_d = self.parameter[2]
        CXPB = self.parameter[0]
        MUTPB = self.parameter[1]
        k = 40

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
                    if random.random() < MUTPB:
                        mute_off1 = self.mutation(cross_off1, self.bound)
                        mute_off2 = self.mutation(cross_off2, self.bound)
                        mo1_fitness = self.evaluate(mute_off1)
                        mo2_fitness = self.evaluate(mute_off2)
                        nextoff.append({'Gene': mute_off1, 'fitness': mo1_fitness})
                        nextoff.append({'Gene': mute_off2, 'fitness': mo2_fitness})
                    else:
                        co1_fitness = self.evaluate(cross_off1)
                        co2_fitness = self.evaluate(cross_off2)
                        nextoff.append({'Gene': cross_off1, 'fitness': co1_fitness})
                        nextoff.append({'Gene': cross_off2, 'fitness': co2_fitness})
                else:
                    nextoff.extend(offspring)

            # 令新生成的种群为当代种群
            self.pop = nextoff

            # 当前成本最小的个体
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.best_individual['fitness']:
                self.best_individual = best_ind

            gen_best.append(self.best_individual)

            # print("当前最好的路径是：{}".format(self.best_individual['Gene']))
            # print("当前最优路径的成本：{}".format(self.best_individual['fitness']))
        
        self.gen_best = gen_best     

# 测试
def main():
    CXPB, MUTPB, N_d, popsize, N_wpt = 0.9, 0.2, 1000, 80, 5
    start = np.array([0, 1000, 20000])
    end = np.array([-1000, 3000, 30000])
    ga_parameter = [CXPB, MUTPB, N_d, popsize, N_wpt, start, end]

    R_attack = 50000
    Missles = [np.array([0,0,0])]
    Missles = np.append(Missles, [np.array([-500,5000,12000])], axis=0)
    Enemy = [end]
    Enemy = np.append(Enemy, [np.array([-5000,5000,12000])], axis=0)
    omega = [0.2, 0.4, 0.2, 0.2]
    outside_info  = [R_attack, Missles, Enemy, omega]

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
        ax.scatter(Missles[0,0], Missles[0,1], Missles[0,2], 'black')
        ax.scatter(Missles[1,0], Missles[1,1], Missles[1,2], 'black')
        ax.plot3D(gen_best[i]['Gene'][:, 0], gen_best[i]['Gene'][:, 1], gen_best[i]['Gene'][:, 2], 'blue', linestyle='-', marker='o')
        plt.pause(0.005)

    plt.show()


if __name__ == '__main__':
    main()