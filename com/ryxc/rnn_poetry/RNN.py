# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf

class RNN(object):
    '''

    '''
    def __init__(self, words, model='lstm', run_size=128, num_layers=2, batch_size=64):
        self.input_data = tf.placeholder(tf.int32, [batch_size, None])
        self.output_targets = tf.placeholder(tf.int32, [batch_size, None])
        # neurl_network
        if model == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fun = tf.nn.rnn_cell.LSTMCell

        cell = cell_fun(run_size, forget_bias=1.0, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        # 创建c_state主线和m_state分线
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        softmax_w = tf.get_variable("softmax_w", [run_size, len(words)+1])
        softmax_b = tf.get_variable("softmax_b", [len(words)+1])

        embedding = tf.get_variable("embedding", [len(words)+1, run_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        outputs, self.last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state)
        output = tf.reshape(outputs, [-1, run_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)










