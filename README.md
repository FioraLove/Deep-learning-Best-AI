# Deep-learning-Best-AI
TensorFlow神经网络，深度学习
环境：

Anaconda3(numpy+Tensorflow) + Python3.7 + Pycharm2019.1.2


Main technique：

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
    
  ## 1.要点：
    
    1.1 随机生成2行100列矩阵,并设置为数据格式
    x = random.rand(2, 100)
    
    1.2 创建一个常量 op, 产生一个 1x2 矩阵. 这个 op 被作为一个节
    matrix1 = tf.constant([[3., 3.]])
    
    1.3　关于矩阵乘法与点乘的探讨
    tf.matmul() 为矩阵乘法
    tf.multiply() 为矩阵点乘
    np.dot() 为矩阵乘法
    np.multiply() 为矩阵点乘
    点乘：矩阵对应元素相乘
    矩阵乘法：用于矩阵相乘，表示为C=A*B，A的列数与B的行数必须相同，C也是矩阵，C的行数等于A的行数，C的列数等于B的列数
    
    1.4 
    
    
