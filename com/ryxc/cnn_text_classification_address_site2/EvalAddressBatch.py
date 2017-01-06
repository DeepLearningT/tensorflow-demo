# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from com.ryxc.cnn_text_classification_address_site2 import DataHelpers


path = DataHelpers.getModelPath('runs')
print(path)
# if not path.isdigit():
#     os._exit(0)
path = "./runs/"+"1483693439"+"/checkpoints/"
print("读取模型目录:", path)


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", path, "Checkpoint directory from training run")
#tf.flags.DEFINE_string("data_path", "./data/address-info-suzhou-789-sample", "地址-网店数据文件目录")
tf.flags.DEFINE_string("data_path", "./data", "地址-网店数据文件目录")
#tf.flags.DEFINE_string("eval_path", "./eval/address-info-10-sample", "评估地址-网店数据文件目录")
tf.flags.DEFINE_string("eval_path", "./eval", "评估地址-网店数据文件目录")
tf.flags.DEFINE_boolean("eval_train", True, "评估批量地址预测")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 数据准备
if FLAGS.eval_train:  # 批量测试
    print("Loading data...")
    x_raw, y = DataHelpers.load_data_and_labels_eval(FLAGS.eval_path)
else:  # 单个测试
    x_raw = ["江苏省	苏州市	常熟市	江苏省 苏州市 常熟市 东南开发区常昆路5888号 靠路边厂房南侧3楼 淘女装 拒收邮政平邮和到付件(张孟 收) 18936103170"]
    print("地址:", x_raw)
    x_raw = DataHelpers.splitWord(x_raw)

# 将数据映射词汇表
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))


print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches =DataHelpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        result_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            print("1.batch_predictions:", batch_predictions)
            # if not FLAGS.eval_train:
            #     print("batch_predictions:", DataHelpers.findSite(FLAGS.data_path, batch_predictions[0]))
            result_predictions = np.concatenate([result_predictions, batch_predictions])
            #print("2.result_predictions:", batch_predictions)

        #print("--all_predictions:", all_predictions)
        all_predictions = [DataHelpers.findSite(FLAGS.data_path, s) for s in result_predictions]
        #print("all_predictions:", all_predictions)

if FLAGS.eval_train:
     correct_predictions = float(sum(all_predictions == y))
     print("测试样本数量: {}".format(len(y)))
     print("准确率: {:g}".format(correct_predictions/float(len(y))))

