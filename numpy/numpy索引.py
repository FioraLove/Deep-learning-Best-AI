# -*- coding:utf-8 -*-
import numpy as np

# 生成一维的3至15的序列
a = np.arange(3, 15)
print(a)
# 元素列表或者数组中，我们可以用如同a[2]一样的表示方法,索引从0开始
print(a[3])

# 二维数组的一维索引
b = np.arange(3, 15).reshape(3, 4)
print(b)
# 实际上这时的A[2]对应的就是矩阵A中第三行(从0开始算第一行)的所有元素
print(b[2])
print('*' * 15)

# 二维索引
# 1.b[1][1]与b[1,1]等效，均表示第二行第二列的值
print(b[1][1])
print(b[1, 1])

# 2.类似于列表，利用‘:’对一定范围内的元素进行切片操作
# 即针对第二行中第2到第4列元素进行切片输出（不包含第4列）
print(b[1, 1:3])
print('-' * 20)

# 逐行打印
for row in b:
    print(row)

# 逐列打印：把原矩阵转置后在逐行打印
for column in b.T:
    print(column)
print('^' * 20)

# 迭代输出:
# flatten是一个展开性质的函数，将多维的矩阵进行展开成1行的数列
print(b.flatten())
# flat是一个迭代器，本身是一个object属性
for item in b.flat:
    print(item)
