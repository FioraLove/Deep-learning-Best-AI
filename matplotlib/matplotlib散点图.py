# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

# 数据集
n = 1024
# 每一个点的x值
x = np.random.normal(0, 1, n)
# 每一个点的y值
y = np.random.normal(0, 1, n)
# 颜色集
t = np.arctan2(x, y)
"""
输入X和Y作为location，size=75，颜色为T，color map用默认值，透明度alpha 为 50%。
x轴显示范围定位(-1.5，1.5)，并用xtick()函数来隐藏x坐标轴，y轴同理
"""
plt.scatter(x, y, s=75, c=t, alpha=.5)

plt.xlim(-1.5, 1.5)
plt.xticks(())

plt.ylim(-1.5, 1.5)
plt.yticks(())

plt.show()
