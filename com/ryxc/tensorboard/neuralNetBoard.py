# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# 可视化训练过程
# 定义添加神经层函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_uniform([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights) #可视化观看变量
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name+"/biases", biases) #可视化观看变量
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            tf.summary.histogram(layer_name+"/Wx_plus_b", Wx_plus_b) #可视化观看变量
        if activation_function is None:
            outpus = Wx_plus_b
        else:
            outpus = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name+"/outputs", outpus)
        return outpus

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)  # 定义噪点
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 定义隐藏层 激励方程采用 relu
#三层神经，输入层（1个神经元），隐藏层（10神经元），输出层（1个神经元）
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu) # 隐藏层
predition = add_layer(l1, 10, 1, n_layer=2, activation_function=None) # 输出层

# 误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition), reduction_indices=[1])) #square()平方,sum()求和,mean()平均值
    tf.summary.scalar('loss', loss) # 可视化观看常量
# 优化器以0.1的学习效率对误差进行更正，下一次会有更好的结果
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始所有的变量
init = tf.global_variables_initializer()

sess = tf.Session()
# 合并到Summary中
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

# 训练1000步
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # 打印误差
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        result = sess.run(merged,feed_dict={xs: x_data, ys: y_data}) #merged也是需要run的
        writer.add_summary(result, i) #result是summary类型的，需要放入writer中，i步数（x轴）
