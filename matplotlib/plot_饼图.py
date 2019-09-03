# -*- coding:utf-8 -*-
# 饼图一
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 绘制饼图
plt.pie([10, 20, 2, 18, 50],
        # 图例说明，设置标签
        labels=['england', 'japan', 'usa', 'usk', 'china'],
        # 显示百分比(%.2f%%即保留小数点后两位)
        autopct='%.2f%%',
        # 设置突出程度
        explode=[0, 0.1, 0.1, 0, 0],
        # 设置阴影
        shadow=False,
        )
# 设置图片朝向
plt.axis('equal')
# plt.axis('square')
plt.show()


# 饼图二：
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [17, 35, 45, 13]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes,
        explode=explode,
        labels=labels,
        autopct='%.3f%%',
        shadow=True,
        startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# 饼图三：
from matplotlib import pyplot as plt
import numpy as np

# some data
labels = ['Frogs', 'Hogs', 'Dogs', 'Logs', 'bad']
data = [13, 32, 50, 3, 2]
explode = [0, 0.1, 0.1, 0, .2]

plt.subplot(2, 2, 1)
plt.pie(data, labels=labels, explode=explode, autopct='%.3f%%', shadow=True, startangle=90)
plt.subplot(2, 2, 2)
plt.pie(data, labels=labels, explode=explode, autopct='%.2f%%', textprops={"size": "smaller"}, shadow=True, radius=1)

plt.show()

# 样例四：嵌套饼图（空心饼图形状效果是通过wedgeprops参数设置馅饼的宽度来实现）
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots()

size = 0.3
vals = np.array([[60, 32], [37, 40], [29, 10]])
labels = ['i', 'love', 'you']
cmap = plt.get_cmap('tab20c')
outer_colors = cmap(np.arange(3) * 4)
inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))
explode = [0, 0.1, 0.1, 0, .2, .2]
label = ['Frogs', 'Hogs', 'Dogs', 'Logs', 'bad', 'woman']

ax.pie(
    # 矩阵的sum函数，axis=1：垂直求和
    vals.sum(axis=1),
    labels=labels,
    radius=1,
    colors=outer_colors,
    wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(vals.flatten(), labels=label, explode=explode, radius=1 - size, colors=inner_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set_title('Pie plot with badwoman')
plt.axis('equal')
plt.show()

# 样例五：极轴上的饼图
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = np.pi / 4 * np.random.rand(N)

fig, ax = plt.subplot(projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()

# 样例七、极坐标上绘制线段
import numpy as np
import matplotlib.pyplot as plt

r = np.arange(0, 8, 0.01)
theta = 2 * np.pi * r

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r)
# 设置最大半径长度
ax.set_rmax(8)
# 设置每一圈的刻度
ax.set_rticks(np.arange(0, 8, 1))  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()

# 八、极坐标上的图例
import matplotlib.pyplot as plt
import numpy as np

# radar green, solid grid lines
plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)

# force square figure and square axes looks better for polar, IMO
fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],
                  projection='polar', facecolor='#d5de9c')

r = np.arange(0, 3.0, 0.01)
theta = 2 * np.pi * r
ax.plot(theta, r, color='red', lw=3, label='a line')
ax.plot(0.5 * theta, r, color='blue', ls='--', lw=3, label='another line')
ax.legend()

plt.show()
