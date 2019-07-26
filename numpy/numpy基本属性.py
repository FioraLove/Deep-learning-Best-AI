# -*- coding:utf-8 -*-
import numpy as np

"""
numpy的三种属性：ndim：维度  ； shape：行数和列数  ； size：元素个数
"""
# 列表转化为矩阵
array = np.array([[1, 2, 3], [4, 5, 6]])
print(type(array))
print('number of dim:', array.ndim)  # 维度
# number of dim: 2

print('shape :', array.shape)  # 行数和列数
# shape : (2, 3)

print('size:', array.size)  # 元素个数
# size: 6
