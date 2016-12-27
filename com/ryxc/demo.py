# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
sess = tf.Session()
a = tf.constant(100)
b = tf.constant(50)
print(sess.run(a+b))