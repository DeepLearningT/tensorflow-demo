# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import tensorflow as tf
from com.ryxc.cnn_text_classification import data_helpers
from tensorflow.contrib import learn
import numpy as np

# Data loading params
tf.flags.DEFINE_float("test_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

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
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

print("-------------------------------------- Build vocabulary---------------------------------------------")
max_document_length = max([len(x.split(' ')) for x in x_text])
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
print("x:", x)
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
print("shuffle_indices", shuffle_indices)

print("-------------------------------------- Split tran/test set-----------------------------------------")
test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))
print("len(y):", len(y))
print("dev_sample_index:", test_sample_index)
x_train, x_test = x_shuffle[:test_sample_index], x_shuffle[test_sample_index:]
y_train, y_text = y_shuffle[:test_sample_index], y_shuffle[test_sample_index:]
print("Train/Test split:{:d}/{:d}".format(len(x_train), len(x_test)))

# Training
# ============================================================================



