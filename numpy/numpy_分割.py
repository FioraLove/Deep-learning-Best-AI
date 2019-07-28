# -*- coding:utf-8 -*-
import numpy as np
# 创建数据 3行4列
a= np.arange(12).reshape(3,4)
print(a)

# 等量分割格式：np.split(矩阵，选择要分成n个部分，axis=0或者1)
# 纵向分割(axis = 1)，
b = np.split(a,2,axis=1) # 纵向等量分割成两个部分
print(b)
# 取其中第一部分
print(b[0])

# 横向分割（axis = 0）
c= np.split(a,3,axis=0)
print(c)
# 取其中第二部分
print(c[1])

# 不等量分割：np.array_split(矩阵，n，axis=0|1)
d = np.array_split(a,3,axis = 1)
print(d)
