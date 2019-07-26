# -*- coding:utf-8 -*-
"""关键词：
np.array：创建数组
np.dtype：指定数据类型
np.zeros：创建数据全为0
np.ones：创建数据全为1
np.empty：创建数据接近0
np.arrange：按指定范围创建数据
np.linspace：创建线段
"""
import numpy as np

# 创建一个数组，指定数据类型dtype=int32
data = np.array([1, 23, 5], dtype=np.int)

# 创建一个数组，指定数据类型dtype=float64
data1 = np.array([1, 23, 5], dtype=np.float)

# 创建特定数据:2d矩阵 2行3
data2 = np.array([[1, 2, 3], [4, 5, 6]])
print(data2.ndim)

# 创建全为零的数组:数据全为零，3行4列
data3 = np.zeros((3, 4))
print(data3)

# 创建全一数组：数据全为一，3行4列
data4 = np.ones((3, 4), dtype=int)
print(data4)

# 创建全空数组, 其实每个值都是接近于零的数
data5 = np.empty((3, 4))
print(data5)

# arange创建连续数组
data6 = np.arange(10, 20, .5)
print(data6)

# .reshape()改变数据形状
data6 = np.arange(10, 20, .5).reshape(4, 5)
print(data6)

# np.linspance()创建线段型数据
#  # 开始端1，结束端10，且分割成20个数据，生成线段
data7 = np.linspace(1, 10, 20).reshape(5, 4)
print(data7)
