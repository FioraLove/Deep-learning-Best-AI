# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np


def f(t):
    # 'A damped exponential'
    s1 = np.cos(2 * np.pi * t)
    e1 = np.exp(-t)
    return s1 * e1


# create data
t1 = np.arange(.0, 5., .2)

l = plt.plot(t1, f(t1), 'ro', label='test')

# plt.setp函数表示设置刚才绘图的属性，设置maker大小为30
plt.setp(l, markersize=10)
plt.setp(l, markerfacecolor='g')
plt.legend()
plt.show()
"""
拓：‘ro’中的’r’代表颜色为红色，’o’表示“圆圈绘制—marker为圆圈”，以下是可选的其他颜色列表。具体请查看plot的帮助信息

    ``'b'``          blue
    ``'g'``          green
    ``'r'``          red
    ``'c'``          cyan
    ``'m'``          magenta
    ``'y'``          yellow
    ``'k'``          black
    ``'w'``          white
 ————————————————
LineWidth——指定线宽

MarkerEdgeColor——指定标识符的边缘颜色

MarkerFaceColor——指定标识符填充颜色

MarkerSize——指定标识符的大小
"""
#
"""
样例二：这是使用条形创建带有误差线的堆积条形图的示例。
注意yerr用于误差条的参数，并且底部用于将女人的条形堆叠在男人条形的顶部
"""
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)  # the x locations for the groups
width = 0.15  # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, yerr=menStd)
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans, yerr=womenStd)  # bottom=menMeans参数是将其女人的条形堆叠在男人条形的顶部

plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()
"""
绘制柱状图，我们主要用到bar()函数。我们先看下bar()的构造函数：bar(x，height， width，*，align='center'，**kwargs)
x：包含所有柱子的下标的列表
height：包含所有柱子的高度值的列表
width：每个柱子的宽度。可以指定一个固定值，那么所有的柱子都是一样的宽。
       或者设置一个列表，这样可以分别对每个柱子设定不同的宽度。
align：柱子对齐方式，有两个可选值：center和edge。
       center表示每根柱子是根据下标来对齐, edge则表示每根柱子全部以下标为起点，然后显示到下标的右边。
       如果不指定该参数，默认值是center。除了以上几个重要的参数，
       还有几个样式参数：
color，设置颜色；
edgecolor设置边框颜色；
linewidth设置柱子的边框宽
tick_label，柱子上显示的标签
"""

# 样例三：这是简单的条形图，在单个条形图上带有误差条形图和高度标签
import numpy as np
import matplotlib.pyplot as plt

men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

ind = np.arange(len(men_means))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
# yerr用于误差条的参数，为图形增加参数条
rects1 = ax.bar(ind - width / 2, men_means, width, yerr=men_std,
                color='SkyBlue', label='Men')
rects2 = ax.bar(ind + width / 2, women_means, width, yerr=women_std,
                color='IndianRed', label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend()


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                '{}'.format(height), ha=ha[xpos], va='bottom')


autolabel(rects1, "left")
autolabel(rects2, "right")

plt.show()

import numpy as np
import matplotlib.pyplot as plt

"""
样例四：简单的水平条形图
matplotlib.pyplot.barh(bottom, width, height=0.8, left=None, hold=None, **kwargs)

Make a horizontal bar plot.

Make a horizontal bar plot with rectangles bounded by:

    left, left + width, bottom, bottom + height
        (left, right, bottom and top edges)
"""
data = np.random.randint(10, 100, size=10)

plt.barh(np.arange(len(data)), data, color='g', label='xiaoye', xerr=10 * np.random.random(size=10))
plt.legend()
plt.show()

# 样例五：3D图像
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
# 在窗口上添加3D坐标轴
ax = Axes3D(fig)

# Plot a sin curve using the x and y axes.
x = np.linspace(0, 1, 100)
y = np.sin(x * 2 * np.pi) / 2 + 0.5
ax.plot(x, y, zs=0, zdir='z', label='curve in (x,y)')

# Plot scatterplot data (20 2D points per colour) on the x and z axes.
colors = ('r', 'g', 'b', 'k')

# Fixing random state for reproducibility
np.random.seed(19680801)

x = np.random.sample(20 * len(colors))
y = np.random.sample(20 * len(colors))
c_list = []
for c in colors:
    c_list.extend([c] * 20)
# By using zdir='y', the y value of these points is fixed to the zs value 0
# and the (x,y) points are plotted on the x and z axes.
ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x,z)')

# Make legend, set axes limits and labels
ax.legend()
plt.xlim(0, 1)
# ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35)

plt.show()

# 样例六：如何使用和不使用着色绘制3D条形图
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# make fake data
x = np.arange(4)
y = np.arange(5)
# meshgrid(x,y):x-y 平面的网格
_x, _y = np.meshgrid(x, y)
# 数据扁平化
x, y = _x.ravel(), _y.ravel()

top = x + y
# zeros_like(a):依据给定数组(a)的形状和类型返回一个新的元素全部为0的数组
# bottom:每个柱的起始位置
bottom = np.zeros_like(top)
# x,y方向的宽厚
width = depth = 1
#
ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('shaded')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
ax2.set_title('Not shaded')

plt.legend()
plt.show()

# 样例七、条形码示例
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# the bar
x = np.where(np.random.rand(500) > 0.7, 1.0, 0.0)

axprops = dict(xticks=[], yticks=[])
barprops = dict(aspect='auto', cmap=plt.cm.binary, interpolation='nearest')

fig = plt.figure()

# a vertical barcode
ax1 = fig.add_axes([0.1, 0.3, 0.1, 0.6], **axprops)
ax1.imshow(x.reshape((-1, 1)), **barprops)

# a horizontal barcode
ax2 = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
ax2.imshow(x.reshape((1, -1)), **barprops)

plt.show()
