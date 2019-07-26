# -*- coding:utf-8 -*-
import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(4)

# a 和 b 是两个属性为 array 也就是矩阵的变量，而且二者都是1行4列的矩阵，
# 其中b矩阵中的元素分别是从0到3

# 矩阵的加法，减法
c = a - b
print(c)
c1 = a + b
print(c1)

# 矩阵中的各元素的乘方
c2 = b ** 2
print(c2)

# 数学函数
c3 = 10 * np.sin(b)
print(c3)

# 标准的矩阵乘法np.dot(a,b)
a1 = np.array([[1, 2], [0, 1]])
b1 = np.arange(4).reshape(2, 2)
c4 = np.dot(a1, b1)
print(c4)

# sum(), min(), max()的使用
a2 = np.random.random((2, 4))
print(a2)
# 对矩阵求和
print(np.sum(a2))
# 对矩阵求最大值
print(np.max(a2))
# 对矩阵求最小值
print(np.min(a2))
# argmin() 和 argmax() 两个函数分别对应着求矩阵中最小元素和最大元素的索引
print(np.argmax(a2))
print(np.argmin(a2))

# 矩阵的均值np.mean()
c5 = np.mean(b1)
print(c5)

# 矩阵的累加
A = np.arange(2, 14).reshape(3, 4)
print(np.cumsum(A))
print(np.sum(A))

# 矩阵的排序：仅针对每一行排序,仿照列表一样的排序操作
A1 = np.arange(14, 2, -1).reshape(3, 4)
print(np.sort(A1))

# 矩阵的转置
print(np.transpose(A1))

# 特殊函数clip()
"""
格式是clip(Array,Array_min,Array_max)，顾名思义，Array指的是将要被执行用的矩阵，
而后面的最小值最大值则用于让函数判断矩阵中元素是否有比最小值小的或者比最大值大的元素，
并将这些指定的元素转换为最小值或者最大值
"""
print(A1)
print(np.clip(A1, 5, 9))
