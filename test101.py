import tensorflow as tf
import numpy as np
import time

start_time = time.time()
# 创建一些数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 开始创建TensorFlow结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + Biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)  # 激活init
    for step in range(200):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(Biases))

end_time = time.time()
print('it cost %s seconds!' % (end_time - start_time))


# 例二：矩阵乘法的处理
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)
# method 1
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
