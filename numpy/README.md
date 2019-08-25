# NumPy

### Python之路——numpy各函数简介之生成数组函数（Array creation routines）

    网站：https://www.cnblogs.com/fortran/archive/2010/09/01/1814773.html
    
    1.ones(shape[, dtype, order])：依据给定形状和类型(shape[, dtype, order])返回一个新的元素全部为1的数组。

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
           
    2.ones_like():依据给定数组(a)的形状和类型返回一个新的元素全部为1的数组。等同于a.copy().fill(1)

    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ones_like(a)
    array([[1, 1, 1],
          [1, 1, 1]])
          
    3、zeros(shape[, dtype, order])
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
    
    4、zeros_like(a)
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
