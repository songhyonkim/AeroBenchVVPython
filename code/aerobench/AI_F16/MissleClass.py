import random
import numpy as np
import matplotlib.pyplot as plt
from FunFunctions import Euclid
from mpl_toolkits.mplot3d import Axes3D

class Missle:

    # 初始化
    def __init__(self, ID, Pos, Captured, Target_Pos, D):

        self.id = ID
        self.pos = Pos
        self.captured = Captured
        self.target_pos = Target_Pos
        self.d = D


    # 控制导弹运动
    def control(self, step):
        delta_x = self.target_pos[0] - self.pos[0]
        delta_y = self.target_pos[1] - self.pos[1]
        delta_z = self.target_pos[2] - self.pos[2]

        if pow(delta_x,2)+pow(delta_y,2)+pow(delta_z,2) > pow(self.d,2):
            self.vel_x = delta_x
            self.vel_y = delta_y
            self.vel_z = delta_z
        else:
            self.vel_x = self.d
            self.vel_y = self.d
            self.vel_z = self.d

        self.pos[0]+=self.vel_x*step
        self.pos[1]+=self.vel_y*step
        self.pos[2]+=self.vel_z*step
 

def main():
    id = 0
    pos = np.array([10,10,10])
    captured = False
    target_pos = np.array([100,100,100])
    d = 80
    step = 1/10

    missle = Missle(id, pos, captured, target_pos, d)

    missle_pos = [pos]
    target_poss = [target_pos]

    # 画图显示
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X', fontdict={'size': 20, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})

    while(1):
        distance = Euclid(target_pos, missle.pos)

        if random.random() > 0.8 and ~missle.captured:
            missle.captured = True

        target_pos[0] += random.randint(-100,200)*step
        target_pos[1] += random.randint(-100,200)*step
        target_pos[2] += random.randint(-100,200)*step

        if missle.captured:
            missle.target_pos = target_pos
        else:
            missle.target_pos = missle.target_pos

        missle.control(step)

        missle_pos = np.append(missle_pos, [np.array(missle.pos)], axis=0)
        target_poss = np.append(target_poss, [target_pos], axis=0)

        ax.plot3D(missle_pos[:, 0], missle_pos[:, 1], missle_pos[:, 2], 'red', linestyle='-', marker='o')
        ax.plot3D(target_poss[:, 0], target_poss[:, 1], target_poss[:, 2], 'green', linestyle='-', marker='o')
        text = f'distance between missle and target : {distance:.3f}'
        ax.set_title(text)
        plt.pause(0.5)

 
if __name__ == '__main__':
    main()