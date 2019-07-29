# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def f(x, y):
    # the height function
    return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
# meshgrid在二维平面中将每一个x和每一个y分别对应起来，编织成栅格
X, Y = np.meshgrid(x, y)

# 使用函数plt.contourf把颜色加进去，位置参数分别为：X, Y, f(X,Y)。透明度0.75，并将 f(X,Y) 的值对应到color map的暖色组中寻找对应颜色
# use plt.contourf to filling contours
# X, Y and value for (X,Y) point，8代表等高线的密集程度,这里被分为10个部分。如果是0，则图像被一分为二
plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap=plt.cm.hot)

# use plt.contour to add contour lines
C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidths=.5)
# inline控制是否将Label画在线里面，字体大小为10。并将坐标轴隐藏
plt.clabel(C, inline=True, fontsize=10)
plt.xticks(())
plt.yticks(())

plt.show()
