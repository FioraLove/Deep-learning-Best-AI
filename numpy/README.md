# NumPy

## 一、Python之路——numpy各函数简介之生成数组函数（Array creation routines）

    网站：https://www.cnblogs.com/fortran/archive/2010/09/01/1814773.html
    
##### 1.ones(shape[, dtype, order])：依据给定形状和类型(shape[, dtype, order])返回一个新的元素全部为1的数组。

    >>> np.ones(5)
    array([ 1., 1., 1., 1., 1.])

    >>> np.ones((5,), dtype=np.int)
    array([1, 1, 1, 1, 1])

    >>> np.ones((2, 1))
    array([[ 1.],
           [ 1.]])

    >>> s = (2,2)
    >>> np.ones(s)
    array([[ 1., 1.],
           [ 1., 1.]])
           
#####  2.ones_like():依据给定数组(a)的形状和类型返回一个新的元素全部为1的数组。等同于a.copy().fill(1)

    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ones_like(a)
    array([[1, 1, 1],
          [1, 1, 1]])
          
#####  3、zeros(shape[, dtype, order])

        依据给定形状和类型(shape[, dtype, order])返回一个新的元素全部为0的数组。

        shape：int或者ints元组；

        定义返回数组的形状，形如：(2, 3)或2。

        dtype：数据类型，可选。

        返回数组的数据类型，例如：numpy.int8、默认为numpy.float64。

        order:{‘C’, ‘F’},可选,返回数组为多维时，元素在内存的排列方式是按C语言还是Fortran语言顺序(row- or columnwise)。

        输出：ndarray

        给定形状，数据类型的数组。

    >>> np.zeros(5)
    array([ 0., 0., 0., 0., 0.])

    >>> np.zeros((5,), dtype=numpy.int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0., 0.],
           [ 0., 0.]])

    >>> np.zeros((2,), dtype=[(’x’, ’i4’), (’y’, ’i4’)]) # custom dtype
    array([(0, 0), (0, 0)],
    dtype=[(’x’, ’<i4’), (’y’, ’<i4’)])
    
##### 4、zeros_like(a)

    依据给定数组(a)的形状和类型返回一个新的元素全部为0的数组。等同于a.copy().fill(0)。

    >>> x = np.arange(6)
    >>> x = x.reshape((2, 3))
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.zeros_like(x)
    array([[0, 0, 0],
           [0, 0, 0]])
    >>> y = np.arange(3, dtype=np.float)
    >>> y
    array([ 0., 1., 2.])
    >>> np.zeros_like(y)
    array([ 0., 0., 0.])

## 二、numpy数据分析练习
    # -*- coding:utf-8 -*-
    import numpy as np
    import pandas as pd

    用numpy求解方程组
    a = np.array([[2, 1, -1],
                  [3, 0, 1],
                  [1, 1, -1]])
    b = np.transpose(np.array([-3, 5, 2]))
    x = np.linalg.solve(a, b)
    print(x)

    # 运用：多元线性回归
    # numpy数据分析问答
##### 1.导入numpy，并打印版本号？
    print(np.__version__)

##### 2.如何创建一个布尔数组?
    arr = np.full((3, 3), True, dtype=bool)  # 法一
    print(arr)

    # 法二：利用np.ones()
    arr1 = np.ones((3, 3), dtype=bool)
    print(arr1)

##### 3.如何从一堆一维1数组中提取满足指定条件的元素？
    arr2 = np.arange(0, 10)
    print(arr2, arr2.ndim, arr2.shape, type(arr2))
    # 提取指定条件的元素
    b = arr2[arr2 % 3 == 1]
    print(b)

##### 4.numpy数组中的另一个值替换满足条件的元素项？
    arr2[arr2 % 2 == 0] = 4396
    print(arr2)

##### 5.人如何改变数组的形状？
    ar = np.arange(3, 15)
    # ps；设置为-1会自动决定列数
    ar1 = ar.reshape(2, -1)
    print(ar1)

##### 6.数组的合并
    a = np.arange(10).reshape(2, -1)
    # np.repeat: Repeat elements of an array
    """
    Examples：
    >>>np.repeat(3,4)
    array([3,3,3,3])
    """
    b = np.repeat(1, 10).reshape(2, -1)
    # 上下合并
    print(np.vstack([a, b]))
    """
    the result like this:
    [[0 1 2 3 4]
     [5 6 7 8 9]
     [1 1 1 1 1]
     [1 1 1 1 1]]
    """
    # 左右合并:Stack arrays in sequence horizontally (column wise)
    print(np.hstack([a, b]))
    """
    the result:
    [[0 1 2 3 4 1 1 1 1 1]
     [5 6 7 8 9 1 1 1 1 1]]
    """

##### 7.How to get the public item between two numpy arrays
    a = np.arange(1, 7)
    print(a)
    b = np.array([7, 2, 3, 8, 9, 0])
    c = np.intersect1d(a, b)
    print('the public item is:', c)

##### 8.How to remove items that existing in another array from an array
    # from 'a' remove all of 'b'
    d = np.setdiff1d(a, b)
    print('the result:', d)
    # >>>array([4,5,6])

##### 9.How to get the position(or index) where two array elements match
    e = np.where(a == b)
    print('the index of the same position is :', e)

##### 10.how to get the numbers that between 'a' and 'b'
    new_array = np.array([2, 9, 3, .1, .5, 8, .0, 1.25])
    print(type(new_array))
    print(new_array)
    # method_1：
    index = np.where(new_array >= .0) & (new_array <= 4)
    print(new_array[index])
    # method_2:
    aa = new_array[(new_array >= .0) & (new_array <= 4)]
    print(aa)


##### 11.如何创建一个Python函数来处理scalars并在numpy数组上工作
    def maxx(x, y):
        if x > y:
            return x
        else:
            return y


    # pair_max实际为原maxx函数向量化后的一个新函数
    pair_max = np.vectorize(maxx, otypes=[float])  # 将函数向量化，变为一个向量函数，otypes指定数据类型

    a = np.array([5, 7, 9, 8, 6, 4, 5])
    b = np.array([6, 3, 4, 8, 9, 7, 1])

    ab = pair_max(a, b)
    print(ab)

##### 12.如何创建包含5到10之间随机浮动的二维数组
    # Input
    array = np.arange(9).reshape(3, 3)

    # solution method 1:
    # random.random生成随机小数
    rand_arr = np.random.randint(low=5, high=10, size=(5, 3)) + np.random.random((5, 3))
    print(rand_arr)

    # Solution method 2:
    # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    rand_arr1 = np.random.uniform(low=5, high=10, size=(5, 3))
    print(rand_arr1)

##### 13.如何在numpy数组中只打印小数点后三位
    ar = np.random.random((3, 3))
    # limit to 3 decimal places
    np.set_printoptions(precision=3)
    print(ar)
    print('--------------')
    # 切片操作
    print(ar[1:2])

##### 14.限制数组打印的项数
    ab = np.arange(9)
    np.set_printoptions(threshold=6)
    print(ab)
    # >>>[0 1 2 ... 6 7 8]

##### 15. np.itemsize:每个元素的字节大小。例如float类型的数组的itemsize为8(64/8),而complex32类型的数组的itemsize为4(32/8)
    array = np.arange(12, dtype=np.float).reshape(3, 2, 2)
    print(array)
    print(array.shape)
    print(array.dtype)
    print(array.itemsize)

### 16.numpy.ravel() vs numpy.flatten()的辨异：
    """
    首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），
    numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
    而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵
     ———————————————— 
     >>> x = np.array([[1, 2], [3, 4]])
    >>> x
    array([[1, 2],
           [3, 4]])
    >>> x.flatten()
    array([1, 2, 3, 4])
    >>> x.ravel()
    array([1, 2, 3, 4])
                        两者默认均是行序优先
    >>> x.flatten('F')
    array([1, 3, 2, 4])
    >>> x.ravel('F')
    array([1, 3, 2, 4])

    >>> x.reshape(-1)
    array([1, 2, 3, 4])
    >>> x.T.reshape(-1)
    array([1, 3, 2, 4])

    2. 两者的区别
    >>> x = np.array([[1, 2], [3, 4]])
    >>> x.flatten()[1] = 100
    >>> x
    array([[1, 2],
           [3, 4]])            # flatten：返回的是拷贝
    >>> x.ravel()[1] = 100
    >>> x
    array([[  1, 100],
           [  3,   4]])
     ———————————————— 
    axis参数表示轴，用来定义超过一维数组的属性，二维数据拥有两个轴，
    matplotlib,pandas, numpy的axis：
    axis=1代表行的运算；
    axis=0代表列的运算;
    - 使用0值表示沿着每一列或行标签\索引值向下执行方法 
    - 使用1值表示沿着每一行或者列标签模向执行对应的方法

    拓：
    numpy中的axis的设置参数与数组的shape有关 
    例如一个shape（3，2，4）的数组，代表一个三维数组，要注意的是这里的维度与物理学的维度的理解是不太一样的 
    axis = 0时，就相当于所求的数组的结果变成shape（2，4） 
    axis = 1时，数组的结果shape（3，4） 
    axis = 2时，数组的结果shape（3，2） 
    这里应该看出来了，当axis=n的时候shape中相应的索引就会被去除，数组发生了降维，那么是如何降维的呢？首先要清楚shape里的数字都代表什么意义： 
    3代表这个numpy数组里嵌套着3个数组（有三层）， 2代表其中每个数组的行数，3代表其中每个数组的列数。
    """


##### 17-广播Broadcasting规则：
    1.如果所有输入数组不具有相同数量的维度，则‘1’将被重复地添加到较小数组的形状，直到所有的数组具有相同数量的维度
    2.确保沿着特定维度具有大小为1的数组表现得好像它们具有沿着该维度具有最大形状的数组的大小

###### 除了通过整数和切片进行索引之外，还可以使用整数数组和布尔数组进行索引


##### 18-numpy通用函数
    """
    ceil：向上取整 3.6>>4 , 4.1>>5, -3.3>>-3
    floor:向下取整
    round：四舍五入
    trunc（number）:舍去小数点后数字
    """
    aa = np.array([.23, 3.14159, 1, 3, 4, -3])
    print(np.ceil(aa))
    print(np.floor(aa))
    print(np.round(aa))
    print(np.trunc(aa))
## 三、数组的创建

##### 1.random随机模块

    创建随机一维数组(N为数组个数)
    np.random.xxxx(N)

    创建随机二维数组（或称为矩阵）
    np.random.randint(low= 5,high =10,size=(5,3)) # 生成随机从low到high的整数，数组型为5行3列(也可以生成高维矩阵size=(3,2,5))
    np.random.random((5, 3))  # 5行3列的随机小数
    np.random.uniform(low=1,high=4,size=(10,1)) # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    np.random.rand(2,2) # 随机0-1的数，但参数至少2个，即np.random.rand((2,2))会出现报错
    
##### 2.np.[array,zero,ones,empty,arrange,linspace....]
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
    data = np.ones((3, 3), dtype=np.float)
    print(data)
    # 创建一个数组，指定数据类型dtype=float64
    data1 = np.array([1, 23, 5], dtype=np.float)

    # 创建特定数据:2d矩阵 2行3列
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
    # 开始端1，结束端10，且分割成20个数据，生成线段
    data7 = np.linspace(1, 10, 20).reshape(5, 4)
    print(data7)
