# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

n = 3
"""
np.arange()用法：np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1。
#一个参数 默认起点0，步长为1 输出：[0 1 2]
a = np.arange(3)

#两个参数 默认步长为1 输出[3 4 5 6 7 8]
a = np.arange(3,9)

#三个参数 起点为0，终点为4，步长为0.1 输出[ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.   1.1  1.2  1.3  1.4 1.5  1.6  1.7  1.8  1.9  2.   2.1  2.2  2.3  2.4  2.5  2.6  2.7  2.8  2.9]
a = np.arange(0, 3, 0.1)
"""

# 生成数据集[0,1,2...11]
X = np.arange(n)
# np.random.uniform从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)  # # 产生n个[0.5,1.0)的数
Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)

# 用facecolor设置主体颜色，edgecolor设置边框颜色为白色，
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

# plt.text分别在柱体上方（下方）加上数值，用%.2f保留两位小数，横向居中对齐ha='center'，纵向底部（顶部）对齐va='bottom'
for x, y in zip(X, Y1):  # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, y + 0.05, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
    # ha: horizontal alignment
    # va: vertical alignment
    plt.text(x + 0.4, -y - 0.05, '%.2f' % y, ha='center', va='top')

# 设置x，y轴的坐标轴范围以及刻度
plt.xlim(-1.25, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())

plt.show()
