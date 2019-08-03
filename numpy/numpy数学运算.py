# -*- coding:utf-8 -*-
import numpy as np
import math

# reshape修改形状
a = np.arange(6).reshape(2, 3)
print('a:\n',a)

# 数组元素迭代器
for i in a.flat:
    print(i, end=' ')
print('0000000000')

# numpy.ravel(),numpy.flatten():均按行展平数列
print(a.ravel()) # [0 1 2 3 4 5]
print('___________')
print(type(a.flatten()))  # <class 'numpy.ndarray'>
print((a.flatten()).shape) # (6,)

# np.transpose(a),a.T数组转置
b = np.transpose(a)
print(a.T)

# numpy.append(arr, values, axis=None)
# axis：默认为 None。当axis无定义时，是横向加成，返回总是为一维数组！当axis有定义的时候，分别为0和1的时候。
# 当axis有定义的时候，分别为0和1的时候（列数要相同）。当axis为1时，数组是加在右边（行数要相同）
c = np.append(a, [[0, 0, 0]])
print(c)
print(np.append(a, [[0, 0, 0]], axis=0))
print(np.append(a, [[0, 0],[1,1]], axis=1))

# numpy字符串函数：
m = ['chd']
n = ['python']
# numpy.char.add()：对两个数组字符串的元素的连接
print(np.char.add(m,n))

# numpy.char.capitalize():字符串的首字母大写
print(np.char.capitalize(m))
# 类似于字符串的str.capitalize()操作
print('chd'.capitalize())

# numpy.char.title():字符串每一个单词的首字母全部大写
print(np.char.title('hello python'))

# numpy.char.lower():对数组每一个元素转换为小写。它对每个元素调用 str.lower
print(np.char.lower('i love bad woman'))

# # numpy.char.upper():对数组每一个元素转换为大写。它对每个元素调用 str.upper
# print(np.char.upper('hello python'))

# numpy.char.split('str',sep='str') 通过指定分隔符对字符串进行分割，并返回数组。默认情况下，分隔符为空格。
print(np.char.split('www.baidu.com', sep='.'))

# numpy.char.strip('str','str') 函数用于移除开头或结尾处的特定字符。
print(np.char.strip('hello python', 'he'))

# numpy.char.join() 函数通过指定分隔符来连接数组中的元素或字符串
print(np.char.join(':','runoob'))

# 指定多个分隔符操作数组元素
print (np.char.join([':','-'],['runoob','google']))

# numpy.char.replace() 函数使用新字符串替换字符串中的所有子字符串
print(np.char.replace('i love bad woman','o','h'))

# numpy的三角函数sin(),cos(),tan()与Python的数学函数（import math）
array = np.array([0, math.pi/2, math.pi/6, math.pi*2/3, 40])

# 正弦值
print(np.sin(array))

# 余弦值
print(np.cos(array))

# 正切函数
print(np.tan(array))

# 四舍五入函数numpy.around() 函数返回指定数字的四舍五入值。
"""
# numpy.around(a,decimals) 参数说明：
a: 数组
decimals: 舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
"""

print(np.around(3.1415927,decimals=3)) # 对单个数值
print(np.around(np.cos(array),decimals=4)) # 对数组数值

# numpy的算术函数add(),subtract(),multiply()和divide()
a1 = np.arange(1,10, dtype=np.float_).reshape(3, 3)
print('a1:\n',a1)
a2 = np.array([1, 1, 2])
# 算术求和
print(np.add(a1, a2))

# 数组相减
print(np.subtract(a1, a2))

# 数组相乘:点乘
print(np.multiply(a1, a2))

# 数组除法
print(np.divide(a1, a2))

# numpy.power('数组‘，幂次/数组) 函数将第一个输入数组中的元素作为底数，计算它与第二个输入数组中相应元素的幂(幂函数)
print(np.power(a1,2)) # 幂次
print(np.power(a1,a2)) # 数组

# 取余：numpy.mod() 计算输入数组中相应元素的相除后的余数
print(np.mod(a1,3)) # 单个数值的求余
print('\n')
print(np.mod(a1,[1,2,3])) # 对数组类的求余
