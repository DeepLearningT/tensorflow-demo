# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from com.ryxc.rnn_poetry.RNN import RNN
from com.ryxc.rnn_poetry import DataHelpers

# ------------------------- 数据预处理 -----------------------------------#
words, poetrys, word_num_map = DataHelpers.preData()

# 把诗转换为向量形式
to_num = lambda word: word_num_map.get(word, len(words))
poetry_vector = [list(map(to_num, poetry)) for poetry in poetrys]
# print(poetry_vector)

# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetry_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batches = poetry_vector[start_index:end_index]
    # print(batches)
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_num_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    x_batches.append(xdata)
    y_batches.append(ydata)



print(x_batches)
print(x_batches[0])
print(y_batches)



# 训练
def train_neural_network():
    rnn = RNN(words, model='lstm', run_size=128, num_layers=2, batch_size=batch_size)
    input_data = rnn.input_data
    output_targets = rnn.output_targets
    logits = rnn.logits
    last_state = rnn.last_state
    targets = tf.reshape(output_targets, [-1])
    loss = seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
    cost = tf.reduce_mean(loss)
    learning_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())
        for epoch in range(50):
            sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
            for n in range(n_chunk):
                train_loss, _, _ = sess.run([cost, last_state, train_op],
                                            feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
                print(epoch, n, train_loss)
            if epoch % 7 == 0:
                saver.save(sess, 'poetry.module', global_step=epoch)

train_neural_network()

