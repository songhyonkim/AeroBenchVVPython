#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from mpl_toolkits.mplot3d import Axes3D

x_track = np.zeros((1, 3))
x_track_s = np.array([.0,.0,.0])
theta = 0
def gen_path(): # 生成螺旋
    global x_track_s,x_track,theta
    theta += 10*np.pi/180
    x = 6*np.sin(theta)
    y = 6*np.cos(theta)
    x_track_s +=[x,y,0.1]
    x_track = np.append(x_track, [x_track_s],axis=0)
    return x_track

ax = plt.axes(projection = '3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('3d_mobile_obs')
#ax.set_xlim([0,70])
#ax.set_ylim([-60,-70])

plt.grid(True)
plt.ion()  # interactive mode on!!!! 很重要，有了他就不需要plt.show()了

for t in count():
    if t == 2500:
        break
    # plt.cla() # 此命令是每次清空画布，所以就不会有前序的效果
    ax.plot3D(x_track[:, 0], x_track[:, 1], x_track[:, 2], 'blue')
    x_track = gen_path()
    print(type(x_track_s))
    plt.pause(0.001)

