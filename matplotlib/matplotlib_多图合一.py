# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np

# 规则图形：

# 创建一个图形窗口
plt.figure()
# subplot(m,n,x):代表创建m*n个小图，x代表第i个图
plt.subplot(2, 2, 1)
# 绘制折线图
x = [1, 2, 3, 4, 5, 6]
y = [3, 5, 7, 8, 1, 2]
plt.plot(x, y)

# plt.subplot(2,2,2)表示将整个图像窗口分为2行2列, 当前位置为2
plt.subplot(2, 2, 2)
data = np.arange(1, 4, .25)
plt.scatter(data, data)

# plt.subplot(2,2,3)表示将整个图像窗口分为2行2列,当前位置为3
plt.subplot(223)
plt.plot([0, 1], [0, 3])

# plt.subplot(2,2,4)表示将整个图像窗口分为2行2列,当前位置为4
plt.subplot(2, 2, 4)
plt.plot([0, 1], [3, 3])

plt.show()
