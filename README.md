# Deep-learning-Best-AI
TensorFlow神经网络，深度学习
环境：

### Anaconda3(numpy+Tensorflow) + Python3.7 + Pycharm2019.1.2


### Main technique：


      线性代数、概率和信息论
      欠拟合、过拟合、正则化
      最大似然估计和贝叶斯统计
      随机梯度下降
      监督学习和无监督学习
      深度前馈网络、代价函数和反向传播

      自适应学习算法
      卷积神经网络
      循环神经网络
      递归神经网络
      深度神经网络和深度堆叠网络
      主成分分析
      softmax回归、决策树和聚类算法

      KNN和SVM
      生成对抗网络和有向生成网络
      机器视觉和图像识别
      自然语言处理
      语音识别和机器翻译
      动态规划
      梯度策略算法
      增强学习（Q-learning）

      Application field：
      机器视觉
      语音处理
      语言信号处理

### 学习网址：
      https://www.cnblogs.com/xingshansi/p/6777945.html
      https://www.matplotlib.org.cn/
## Matplotlib

### 3D图绘制：

#### 拉弧测试
   
    # -*- coding:utf-8 -*-
    from matplotlib import pyplot as plt
    import numpy as np


    def f(t):
        # 'A damped exponential'
        s1 = np.cos(2 * np.pi * t)
        e1 = np.exp(-t)
        return s1 * e1


    # create data
    t1 = np.arange(.0, 5., .2)

    l = plt.plot(t1, f(t1), 'ro', label='test')

    # plt.setp函数表示设置刚才绘图的属性，设置maker大小为30
    plt.setp(l, markersize=10)
    plt.setp(l, markerfacecolor='g')
    plt.legend()
    plt.show()
    """
    拓：‘ro’中的’r’代表颜色为红色，’o’表示“圆圈绘制—marker为圆圈”，以下是可选的其他颜色列表。具体请查看plot的帮助信息

        ``'b'``          blue
        ``'g'``          green
        ``'r'``          red
        ``'c'``          cyan
        ``'m'``          magenta
        ``'y'``          yellow
        ``'k'``          black
        ``'w'``          white
     ————————————————
    LineWidth——指定线宽

    MarkerEdgeColor——指定标识符的边缘颜色

    MarkerFaceColor——指定标识符填充颜色

    MarkerSize——指定标识符的大小
    """
#### 使用条形创建带有误差线的堆积条形图

    """
    样例二：这是使用条形创建带有误差线的堆积条形图的示例。
    注意yerr用于误差条的参数，并且底部用于将女人的条形堆叠在男人条形的顶部
    """
    import numpy as np
    import matplotlib.pyplot as plt

    N = 5
    menMeans = (20, 35, 30, 35, 27)
    womenMeans = (25, 32, 34, 20, 25)
    menStd = (2, 3, 4, 1, 2)
    womenStd = (3, 5, 2, 3, 3)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, menMeans, width, yerr=menStd)
    p2 = plt.bar(ind, womenMeans, width,
                 bottom=menMeans, yerr=womenStd) # bottom=menMeans参数是将其女人的条形堆叠在男人条形的顶部

    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    plt.show()
    """
    绘制柱状图，我们主要用到bar()函数。我们先看下bar()的构造函数：bar(x，height， width，*，align='center'，**kwargs)
    x：包含所有柱子的下标的列表
    height：包含所有柱子的高度值的列表
    width：每个柱子的宽度。可以指定一个固定值，那么所有的柱子都是一样的宽。
           或者设置一个列表，这样可以分别对每个柱子设定不同的宽度。
    align：柱子对齐方式，有两个可选值：center和edge。
           center表示每根柱子是根据下标来对齐, edge则表示每根柱子全部以下标为起点，然后显示到下标的右边。
           如果不指定该参数，默认值是center。除了以上几个重要的参数，
           还有几个样式参数：
    color，设置颜色；
    edgecolor设置边框颜色；
    linewidth设置柱子的边框宽
    tick_label，柱子上显示的标签
    """
    
#### 单个条形图上带有误差条形图和高度标签

    # 样例三：这是简单的条形图，在单个条形图上带有误差条形图和高度标签
    import numpy as np
    import matplotlib.pyplot as plt

    men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
    women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

    ind = np.arange(len(men_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    # yerr用于误差条的参数，为图形增加参数条
    rects1 = ax.bar(ind - width / 2, men_means, width, yerr=men_std,
                    color='SkyBlue', label='Men')
    rects2 = ax.bar(ind + width / 2, women_means, width, yerr=women_std,
                    color='IndianRed', label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind)
    ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()

    # autolabel函数为条形图上增加高度标签（即显示此柱所表示的具体数值）
    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')


    autolabel(rects1, "left")
    autolabel(rects2, "right")

    plt.show()
    
#### 简单的水平条形图
    """
    样例四：简单的水平条形图
    matplotlib.pyplot.barh(bottom, width, height=0.8, left=None, hold=None, **kwargs)

    Make a horizontal bar plot.

    Make a horizontal bar plot with rectangles bounded by:

        left, left + width, bottom, bottom + height
            (left, right, bottom and top edges)
    """
    data = np.random.randint(10, 100, size=10)

    plt.barh(np.arange(len(data)), data, color='g', label='xiaoye', xerr=10 * np.random.random(size=10))
    plt.legend()
    plt.show()
    
#### 简单3D图像

    # 样例五：3D图像
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    # 在窗口上添加3D坐标轴
    ax = Axes3D(fig)

    # Plot a sin curve using the x and y axes.
    x = np.linspace(0, 1, 100)
    y = np.sin(x * 2 * np.pi) / 2 + 0.5
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x,y)')

    # Plot scatterplot data (20 2D points per colour) on the x and z axes.
    colors = ('r', 'g', 'b', 'k')

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    x = np.random.sample(20 * len(colors))
    y = np.random.sample(20 * len(colors))
    c_list = []
    for c in colors:
        c_list.extend([c] * 20)
    # By using zdir='y', the y value of these points is fixed to the zs value 0
    # and the (x,y) points are plotted on the x and z axes.
    ax.scatter(x, y, zs=0, zdir='y', c=c_list, label='points in (x,z)')

    # Make legend, set axes limits and labels
    ax.legend()
    plt.xlim(0, 1)
    # ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    ax.view_init(elev=20., azim=-35)

    plt.show()
    
#### 如何使用和不使用着色绘制3D条形图
    # 样例六：如何使用和不使用着色绘制3D条形图
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # make fake data
    x = np.arange(4)
    y = np.arange(5)
    # meshgrid(x,y):x-y 平面的网格
    _x, _y = np.meshgrid(x, y)
    # 数据扁平化
    x, y = _x.ravel(), _y.ravel()

    top = x + y
    # zeros_like(a):依据给定数组(a)的形状和类型返回一个新的元素全部为0的数组
    # bottom:每个柱的起始位置
    bottom = np.zeros_like(top)
    # x,y方向的宽厚
    width = depth = 1
    #
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
    ax1.set_title('shaded')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
    ax2.set_title('Not shaded')

    plt.legend()
    plt.show()
    
      最后总结一下，绘制和显示图片常用到的函数有：

      函数名	      功能	                        调用格式
      figure	创建一个显示窗口	        plt.figure(num=1,figsize=(8,8)
      imshow	绘制图片	                  plt.imshow(image)
      show	      显示窗口	                  plt.show()
      subplot	划分子图	                  plt.subplot(2,2,1)
      title	      设置子图标题(与subplot结合使用）	plt.title('origin image')
      axis	      是否显示坐标尺	         plt.axis('off')
      subplots	创建带有多个子图的窗口	     fig,axes=plt.subplots(2,2,figsize=(8,8))
      ravel	      为每个子图设置变量	       ax0,ax1,ax2,ax3=axes.ravel()
      set_title	设置子图标题（与axes结合使用）	ax0.set_title('first window')
      tight_layout自动调整子图显示布局	      plt.tight_layout()

## Numpy

##### numpy.ravel() vs numpy.flatten()的辨异：
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
###### axis=1代表行的运算；
###### axis=0代表列的运算;
###### 使用0值表示沿着每一列或行标签\索引值向下执行方法 
###### 使用1值表示沿着每一行或者列标签模向执行对应的方法

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


##### numpy通用函数
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

##### random随机模块

    创建随机一维数组(N为数组个数)
    np.random.xxxx(N)

    创建随机二维数组（或称为矩阵）
    np.random.randint(low= 5,high =10,size=(5,3)) # 生成随机从low到high的整数，数组型为5行3列(也可以生成高维矩阵size=(3,2,5))
    np.random.random((5, 3))  # 5行3列的随机小数
    np.random.seed(n) # 设置随机算法的初始值
   
    np.random.uniform(low=1,high=4,size=(10,1)) # 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    np.random.rand(2,2) # 随机0-1的数，但参数至少2个，即np.random.rand((2,2))会出现报错
