# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

x = np.linspace(-10, 10, 200)
y1 = x
y2 = x ** 2
y3 = 3 * (x ** 3) + 5 * (x ** 2) + 2 * x + 1
plt.plot(x, y1, color='red', label='y=x')
plt.plot(x, y2, color='green', label='y=x^2')
plt.plot(x, y3, color='pink', label='y=y3')
plt.xlim((-3, 3))
plt.ylim(-100, 100)
plt.legend()
# 坐标轴中移
ax = plt.gca()
# 隐藏上边和右边
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
# 移动另外两个轴
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.show()


# 绘制柱形图与饼图
data = [32, 48, 21, 100]
labels = ['tian', 'xia', 'di', 'yi']
plt.bar(np.arange(len(data)), data, facecolor='#9999ff', width=.1)
# 即用labels的值替换np.arange(len(data))的值作为x轴的刻度
plt.xticks(np.arange(len(data)), labels)
plt.show()

# 绘制饼图
plt.pie([10, 20, 2, 18, 50],
        # 图例说明，设置标签
        labels=['england', 'japan', 'usa', 'usk', 'china'],
        # 显示百分比
        autopct='%.2f%%',
        # 设置突出程度
        explode=[0, 0.1, 0.1, 0, 0])
# 设置图片朝向
plt.axis('equal')
plt.show()
