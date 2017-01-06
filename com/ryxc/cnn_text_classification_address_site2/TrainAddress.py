# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import datetime
import os

import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib import learn
from com.ryxc.cnn_text_classification_address_site2.TextCNN import TextCNN
from com.ryxc.cnn_text_classification_address_site2 import DataHelpers

# Data loading params
tf.flags.DEFINE_float("test_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_path", "data/address-info-suzhou-789-sample", "地址-网店数据文件目录")
tf.flags.DEFINE_string("runs_path", "runs", "模型存储目录")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")  # 滤波器尺寸
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")  # 每个过滤器的过滤器数量大小
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print(" ")

# Data Preparation
# =====================================================================================================
# Load data
print("Loading data...")
x_text, y = DataHelpers.load_data_and_labels(FLAGS.data_path)
# print("y:", y)
print("-------------------------------------- Build vocabulary---------------------------------------------")
max_document_length = max([len(x.split('\t')) for x in x_text])
print("max_document_length:", max_document_length)


vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# 计算tf-idf
# 返回每行的词id, 其中词id是该词在x_text的_tokenizer(分词去重)后的位置
fit_transform = vocab_processor.fit_transform(x_text)
# for a in fit_transform:
#     print(a)
fit_transform_list = list(fit_transform)
x = np.array(list(fit_transform_list))
# 文档分词索引集合
# print("x:", x)
# 分类变量的词汇类
print("Vocabulary Size:{:d}".format(len(vocab_processor.vocabulary_)))

print("-------------------------------------- Randomly shuffle data--------------------------------")
np.random.seed(10)  # 设置相同的seed种子值，返回的随机数据是一样的
# print("Random number with seed 10 : ", np.random.permutation([1, 4, 9, 12, 15]))
# np.random.seed(10)
# print("Random number with seed 10 : ", np.random.permutation([1, 4, 9, 12, 15]))
# # Random number with seed 10 :  [ 9 12  1 15  4]
# # Random number with seed 10 :  [ 9 12  1 15  4]
# permutation 返回与集合size相同的随机集合
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffle = x[shuffle_indices]
y_shuffle = y[shuffle_indices]
# print("shuffle_indices", shuffle_indices)

print("-------------------------------------- Split tran/test set-----------------------------------------")
test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
# print("len(y):", len(y))
# print("test_sample_index:", test_sample_index)
x_train, x_test = x_shuffle[:test_sample_index], x_shuffle[test_sample_index:]
y_train, y_text = y_shuffle[:test_sample_index], y_shuffle[test_sample_index:]
print("x_train/x_test split:{:d}/{:d}".format(len(x_train), len(x_test)))
print("y_train/y_text split:{:d}/{:d}".format(len(y_train), len(y_text)))


# Training
print("============================================================================")
print("Training starting...........")

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)

    cnn = TextCNN(sequence_length=x_train.shape[1],
                  num_classes=y_train.shape[1],
                  vocab_size=len(vocab_processor.vocabulary_),
                  embedding_size=FLAGS.embedding_dim,
                  filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                  num_filters=FLAGS.num_filters,
                  l2_reg_lambda=FLAGS.l2_reg_lambda)


    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # Adam 适应性动量估计法 是另一种能对不同参数计算适应性学习率的方法
    optimizer = tf.train.AdamOptimizer(1e-3)
    # 根据loss损失的变量值计算梯度变量
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    # 定义训练器
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.runs_path, timestamp))
    # print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    # Generate batches
    batches = DataHelpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

    sess.run(tf.global_variables_initializer())


    def train_step(x_batch, y_batch):
        feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
        _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
        time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


    def dev_step(x_batch, y_batch):
        feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
        step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    DataHelpers.writeFile(FLAGS.runs_path+'/train_status', 'running')

    for batch in batches:
        x_batch, y_batch = zip(*batch)

        train_step(x_batch, y_batch)

        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(x_batch, y_batch)
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to{}\n".format(path))

    DataHelpers.writeFile(FLAGS.runs_path+'/train_status', 'ending')


