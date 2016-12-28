# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 返回一个tensor
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义卷积网络神经层
# x 图片的所有信息
def conv2d(x, W):
    # strides[1,x_movement,y_movement,1] strides 定义步长 四个长度的列表 0位和3位固定为1， 1位和2位分别表示x,y中的跨度
    # padding VALID抽取的长和宽比原图小但是类别抽取是在里面的  SAME抽取的长和宽和原图一样但是类别抽取有一部分是在外面的
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 为了防止跨步太大，图片信息都是太多。中间加了一个pooling的处理，将跨度减小，然后用pooling处理的时候将跨度变大的
# 用pooling处理的跨度大的，这里的处理可以保留更多的图片信息
# 需要把2x2窗子里面那个最大的拿走
def max_pool_2x2(x):
    # 在pooling的阶段把长和宽减小了
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1表示所有的sample  1表示图片的厚度(黑白厚度为1个单位，彩色GRB为3)

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5X5， in size(图片厚度)1， out size(高度) 32
b_conv1 = bias_variable([32])  # 高度32
# relu 激励函数转为非线性
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32 (SAMPLE)
h_pool1 = max_pool_2x2(h_conv1)                           # output size 14x14x32 (跨步为2)
## conv2 layer ##
W_conv2 = weight_variable([5, 5, 32, 64])  # path 5X5, in size 32, out size 64 特征
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                           # output size 7x7x64
## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])  # 定义1024 变的更高
b_fc1 = bias_variable([1024])
# [n_samples,7,7,64] ->> [n_sample,7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
# 输出为10，输出的每一维都是图片属于该类别的概率。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax 计算概率

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
# adamOptimizer 学习效率是0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))