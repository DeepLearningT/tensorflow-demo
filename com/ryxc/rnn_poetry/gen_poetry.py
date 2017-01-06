# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-

# 使用训练好的模型生成古诗
from com.ryxc.rnn_poetry.RNN import RNN
from com.ryxc.rnn_poetry import DataHelpers
import tensorflow as tf
import numpy as np



def gen_poetry():
    words, poetrys, word_num_map = DataHelpers.preData()

    def to_word(weight):
        t = np.cumsum(weight)
        s = np.sum(weight)
        sample = int(np.searchsorted(t, np.random.rand(1) * s))
        return words[sample]

    rnn = RNN(words, model='lstm', run_size=128, num_layers=2, batch_size=64)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables)
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, "poetry.model")

        state_ = sess.run(rnn.cell.zero_state(1, tf.float32))
        x = np.array([list(map(word_num_map.get, '['))])
        [probs_, state_] = sess.run([rnn.probs, rnn.last_state],
                                    feed_dict={rnn.input_data: x, rnn.initial_state: state_})

        word = to_word(probs_)
        poem = ''
        while word != ']':
            poem += word
    return poem







