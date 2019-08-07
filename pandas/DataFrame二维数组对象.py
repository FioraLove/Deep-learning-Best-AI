# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

"""
    DataFrame是一个表格式的数据结构，含有一组有序的列（即：好几列）。
　　DataFrame可以被看做是由Series组成的字典，并且共用一个索引。
"""
# 创建方式1：通过一个字典来创建
a = pd.DataFrame({'name': ['chd', 'zyq', 'fqt'], 'age': [18, 12, 15]})
print(a)
"""
>>>a  结果为
  name  age
0  chd   18
1  zyq   12
2  fqt   15
"""
# index指定行索引
a1 = pd.DataFrame({'name': ['chd', 'zyq', 'fqt'], 'age': [18, 12, 15]},index=['a','b','c'])
print(a1)
print('-------------------------')

# 用Series来组成字典
a2 = pd.DataFrame({'one':pd.Series([1,2,3],index=['a','b','c']),'two':pd.Series([5,6,7,8],index=['a','b','c','d'])})
print(a2)

#　CSV文件的读写
# CSV文件的写入
print(a2.to_csv('demo.csv'))

# CSV文件的读取
print(pd.read_csv('demo.csv'))


# DataFrame常用属性
# var.index获取行索引
print(a2.index)

# var.columns:获取列索引
print(a2.columns)

# values 获取取值数组（一般是二维数组）
print(a2.values)

# T:装置
print(a2.T)

# descriibe():获取统计数据
print(a2.describe())
