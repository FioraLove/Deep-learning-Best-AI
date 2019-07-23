import tensorflow as tf
import numpy as np


# 线性回归

def liner_regression():
    # 1.准备数据，x为随机数
    X = tf.random_normal(shape=[100, 1])
    # y真实值，利用矩阵乘法
    y_true = tf.matmul(X, [[0.8]]) + 0.7

    # 2.构造模型
    # 权重为随机数
    weight = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    # 损失值为随机数
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    # 利用矩阵乘法，y = weight * X + bias
    y_predict = tf.matmul(X, weight) + bias

    # 构造损失函数（预测值与真实值做差）
    error = tf.reduce_mean(tf.square(y_predict - y_true))
    # 3.梯度下降自适应优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
    # 显式初始化变量
    init = tf.global_variables_initializer()

    # 收集变量
    tf.summary.scalar('error', error)
    tf.summary.histogram('weight', weight)
    tf.summary.histogram('bias', bias)

    # 合并变量
    merged = tf.summary.merge_all()
    # 创建会话
    with tf.Session() as sess:
        sess.run(init)

        print('训练前的 权重:%s  偏差:%s  损失值:%s' % (weight.eval(), bias.eval(), error.eval()))
        # 开始训练
        for i in range(1000):
            # if i % 50 == 0:
            sess.run(optimizer)

            # 每50次打印当前结果
            if i % 50 == 0:
                print('训练后的第%s次的 权重:%s  偏差:%s  损失值:%s' % (i, weight.eval(), bias.eval(), error.eval()))

    return None


if __name__ == '__main__':
    liner_regression()
