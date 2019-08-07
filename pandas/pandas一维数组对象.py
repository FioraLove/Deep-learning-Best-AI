# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

# Series——一维数据对象
# Series是一种类似于一维数组的对象，由一组数据和一组与之相关的数据标签（索引）组成
a = pd.Series([2,3,4,5])
# print(a.shape)

# pandas的index指定索引目录
a1 = pd.Series([2, 3, 4, 5], index=['a', 'b', 'c', 'd'])
# print(a1)

# series支持array特性:
# 1.从ndarray创建Series
a2 = pd.Series(np.arange(1, 9, 2))
# print(a2)

# 2.与标量的运算
a3 = pd.Series(np.array([6, 3, 2, 4]))
a3 *= 2
# print(a3)

# 3.两个series运算
b = a2 + a3
print(b)

# 4.Series的索引:下角标从零开始
print(b[0]) # 单个字符的索引
print(b[[1,2,3]]) # 数组组合的索引所对应的具体值

# 5.切片操作
c = b[1:4]
print(c)

# 6.通用函数（最大值，绝对值等）
data = 10
c1 = pd.Series(np.linspace(-1,1,data))
print(c1)
print(c1.min())

# Series支持字典的特性：
# 1.从字典创建series
sr = pd.Series({'name': 'chd', 'age': 18, 'gender': 'man', 'capacity': 10,'is_love_girl':True})
print(type(sr))

# 2.in运算：‘a' in sr:只能判断键名是否在'pandas.core.series.Series'类中，不能判断values值
print('name' in sr)
print('age' in sr)

# 3.遍历运算:只遍历打印值，不会打印键名
for i in sr:
    print(i)

# 4.获取索引以及对应值
print(sr.index) # 获取索引
print(sr.values) # 获取对应的值

# 整数索引问题：
str = pd.Series(np.arange(4.))
print(str)
# 浅拷贝，a，b都指向内存地址，若其中一个修改值，则另一个也必会修改值
str2 = str[1:].copy()
print(str2)

"""
series数据对齐，pandas在进行两个Series对象运算时，会按索引对齐然后运算
若两series对象的index长度不一样，则会当做数据缺失值NaN处理
"""
# 1.Series对象运算
sr1 = pd.Series([12, 23, 34, 34], index=['c', 'a', 'b', 'd'])
sr2 = pd.Series([11, 20, 10], index=['a', 'b', 'c'])
# print(sr1 + sr2)
#
# # 2.Series灵活算术方法
# print(sr1.add(sr2,fill_value = 0))
# print(sr1.sub(sr2,fill_value = 0))
# print(sr1.div(sr2,fill_value = 0))

# 3.缺失值处理方式：
# 缺失值处理方式一：过滤缺失数据str.dropna()：直接删除缺值的键值
d = pd.Series(np.array([1, 3, 4., None, 8, None, 99]))
# print(d.dropna())
# print(d.isnull())  # 判断每一键值对是否缺失数据

# 缺失值处理方式二：填充缺失数据：str.fillna(填充值)
print(d.fillna(1))

# 提出NaN后求得平均值
print(d.mean())
"""
Series数据对象小结:
　　Series是数组和字典的结合体，可以通过下标和标签来访问。
　　当索引值为整数时，索引一定会解释为标签。可以使用loc和iloc来明确指明索引被解释为标签还是下标。
　　如果两个Series对象的索引不完全相同，则结果的索引是两个操作数索引的并集。
　　如果只有一个对象在某索引下有值，则结果中该索引的值为nan(缺失值)。
　　缺失数据处理方法：dropna（过滤）、fillna（填充）。
"""


