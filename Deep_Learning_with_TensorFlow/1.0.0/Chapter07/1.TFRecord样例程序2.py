# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 2. 读取TFRecord文件
# 读取文件。
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["Records/output.tfrecords"])

# 从文件中读出一个样例。也可以使用read_up_to函数一次性读取多个样例。
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })


# tf.decode_raw 可以将字符串解析成图像对应的像素数组。
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行可以读取TFRecord文件的一个样本，当所有样例都读完之后，在此样例中程序会在重头读取
for i in range(10):
     image, label, pixel = sess.run([images, labels, pixels])
     print label

