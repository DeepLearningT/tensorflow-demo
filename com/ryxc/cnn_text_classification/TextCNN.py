# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 返回一个tensor
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#  定义卷积网络神经层
def conv2d(x, W):
    # strides[1,x_movement,y_movement,1] strides 定义步长 四个长度的列表 0位和3位固定为1， 1位和2位分别表示x,y中的跨度
    # padding VALID抽取的长和宽比原图小但是类别抽取是在里面的  SAME抽取的长和宽和原图一样但是类别抽取有一部分是在外面的
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID")


#  定义池化层
def max_pool(h, x_span):
    return tf.nn.max_pool(h, ksize=[1, x_span, 1, 1], strides=[1, 1, 1, 1], padding="VALID")





class TextCNN(object):
    '''
    sequence_length: x_train.shape[1]  56
    num_classes: y_train.shape[1]   2
    vocab_size=len(vocab_processor.vocabulary_)分类变量的词汇类   18758
    embedding_size: 128
    filter_sizes: 3  3,4,5
    num_filters: 128
    l2_reg_lambda:
    '''
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda):
        # Placeholder for input and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")  # dropout解决过拟合

        # 计算L2范数的一半张量没有 开根号
        l2_loss = tf.constant(0.0)

        # random_uniform返回形状为shape的tensor，其中的元素服从minval和maxval之间的均匀分布
        with tf.device('/cpu:0'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))   # 18758x128
            # 需要对批数据中的单词建立嵌套向量 根据input_x中的id，寻找W中的对应元素。
            # 比如，input_x=[1,3,5]，则找出W中下标为1,3,5的向量组成一个矩阵返回。
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)  # ?x56x128  image:56x128
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)  # ?x56x128x1 在 tensor的最后 插入1维

        # 循环创建滤波器（一个滤波器包括一个卷积层和一个池化层）
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # 创建卷积层
            # 1: patch(3x128)  特征图像 in size:1  out size:128
            # 2: patch(4x128)  特征图像 in size:1  out size:128
            # 3: patch(5x128)  特征图像 in size:1  out size:128
            filter_shape = [int(filter_size), embedding_size, 1, num_filters]
            W = weight_variable(filter_shape)
            b = bias_variable([num_filters])
            conv = conv2d(self.embedded_chars_expanded, W)
            # 1: img:56x128  out size: 54x1x128  由于VALID所以56-3+1=54
            # 2: img:56x128  out size: 53x1x128  由于VALID所以56-4+1=53
            # 3: img:56x128  out size: 52x1x128  由于VALID所以56-5+1=52

            # 使用rele激励函数转为非线性
            h = tf.nn.relu(tf.nn.bias_add(conv, b))

            # 创建池化层 使用maxpooling
            pooled = max_pool(h, sequence_length-int(filter_size)+1)
            # 1. image:54x1x128  范围:56-3+1=54,1  maxpooling：54x1范围内取最大的    out size:1x1x128
            # 2. image:53x1x128  范围:56-4+1=53,1  maxpooling：53x1范围内取最大的    out size:1x1x128
            # 3. image:52x1x128  范围:56-5+1=52,1  maxpooling：52x1范围内取最大的    out size:1x1x128
            # 因为patch的宽度和image的宽度一致，所以在滤波的时候只有1x1维数据 厚度为128

            pooled_outputs.append(pooled)

        # 归约所有的池化特征
        num_filters_total = num_filters * len(filter_sizes)  # 384
        self.h_pool = tf.concat(3, pooled_outputs)  # ?x1x1x384 按照3个维度合并
        # 重塑一个张量  [n_samples,1,1,384] ->> [n_sample,1*1*384]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # 添加过拟化处理
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # output 分数和预测
        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        l2_loss += tf.nn.l2_loss(W)  #  计算L2范数的一半张量没有 开根号
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_drop, W, b)
        self.predictions = tf.argmax(self.scores, 1)

        # 计算平均 cross-entropy loss
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy 准确度计算
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

























