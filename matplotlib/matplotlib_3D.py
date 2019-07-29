# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 定义一个图像窗口
fig = plt.figure()
# 在窗口上添加3D坐标轴
ax = Axes3D(fig)
# X, Y value
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)  # x-y 平面的网格
R = np.sqrt(X ** 2 + Y ** 2)
# height value
Z = np.sin(R)
# 绘制3D图形表面(rstride 和 cstride 分别代表 row 和 column 的跨度)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# 添加 XY 平面的等高线
# zdir 选择了x，那么效果将会在x平面产生投影
ax.contourf(X, Y, Z, zdir='x', offset=-2, cmap=plt.get_cmap('rainbow'))

plt.show()
