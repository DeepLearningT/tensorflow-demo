# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 定义添加神经层函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_uniform([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outpus = Wx_plus_b
    else:
        outpus = activation_function(Wx_plus_b)
    return outpus

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)  # 定义噪点
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 定义隐藏层 激励方程采用 relu
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 定义输出层
predition = add_layer(l1, 10, 1, activation_function=None)

# 误差
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition), reduction_indices=[1]))
# 优化器以0.1的学习效率对误差进行更正，下一次会有更好的结果
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始所有的变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# 打印视图数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# 学习1000步
for i in range(1000):
    # 训练
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # 打印误差
        print (sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predition_value = sess.run(predition, feed_dict={xs: x_data})
        lines = ax.plot(x_data, predition_value, 'r-', lw=5)

        plt.pause(0.1)




