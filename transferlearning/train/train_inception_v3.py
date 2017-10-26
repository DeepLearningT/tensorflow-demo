# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import datetime
import glob
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

# Inception-v3模型瓶颈层结果的张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3:0'    # (1, 1, 1, 2048)
#BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape'  # (1, 2048)

# 图像输入张量对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


# 谷歌训练好的Inception-v3模型目录
MODEL_DIR = "../model/inception_pretrain"

# 谷歌训练好的Inception-v3模型名称
MODEL_FILE = "classify_image_graph_def.pb"

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得到的特征向量保存在文件中，
# 免去重复计算
CACHE_DIR = "tmp/bottleneck"



# 验证数据百分比
VALIDATION_PERCENTAGE = 10
# 测试数据百分比
TEST_PERCENTAGE = 10

# 定义神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

# 图片路径
INPUT_DATA = '../data/flower_photos'

# 从数据文件夹中读取所有的图片列表并按训练，验证， 测试数据分开
# testing_percentage和validation_percentage指定测试数据集和验证数据集的占比
def create_iamge_lists(test_percentage, validation_percentage):
    # 得到的所有图都存在restult这个字典 key为类型 value也是也一个字典
    result = {}
    # 获取图片路径下的所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前的目录， 不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名称获取类别的名称
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集，测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 随机将数据分到训练数据集，测试数据和验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (test_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing' : testing_images,
            'validation' : validation_images
        }
    return result

def get_image_path(image_lists, image_dir, label_name, image_index, category):
    # 获取给定类别中所有图片的信息
    label_image_lists = image_lists[label_name]
    # 根据所属的数据集的名称获取集合中的全部图片信息
    category_image_list = label_image_lists[category]
    # 取模操作  将随机的image_index通过对类别下所属的数据集的图片数量取摸转换成在所属的数据集图取随机
    mod_index = image_index % len(category_image_list)
    # 获取图片的文件名
    base_name = category_image_list[mod_index]
    sub_dir = label_image_lists['dir']
    # 最终的地址
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path




# 通过类别名称，所属的数据集和图片编号获取经过Inception-v3模型处理后的特征向量文件地址
def get_bottleneck_feature_file_path(image_lists, image_dir, label_name, image_index, category):
    # 获取一张图片在Inception-v3瓶颈层处理后的特征向量文件路径
    feature_file_path = get_image_path(image_lists, image_dir, label_name, image_index, category) + ".feature"
    #print("获取一张图片在Inception-v3瓶颈层处理后的特征向量文件路径:", feature_file_path)
    return feature_file_path


# 使用加载的训练好的Inception-v3模型处理一张图片，得到这张图片的特征向量
def run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {jpeg_data_tensor: image_data})
    #print("获取pool_3:0 四维向量:", bottleneck_values.shape)
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    #print("压缩成一维特征向量:", bottleneck_values.shape)
    return bottleneck_values



# 获取一张图片经过Inception-v3模型处理后的特征向量，
# 先视图寻找已经计算并且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name,
                                 image_index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_image_lists = image_lists[label_name]
    sub_dir = label_image_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    # 获取一张图片在Inception-v3瓶颈层处理后的特征向量文件路径
    bottleneck_feature_file_path = get_bottleneck_feature_file_path(
        image_lists, CACHE_DIR, label_name, image_index, category)

    # 如果这个特征向量文件不存在， 则通过Inception-v3模型来计算特征向量，并将计算结果存入文件
    if not os.path.exists(bottleneck_feature_file_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, image_index, category)
        #print("获取原始的图片路径", image_path)

        # 获取原始图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        #print("获取原始图片内容")

        # 通过Inception-v3模型计算特征向量
        bottleneck_feature_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor
        )
        #print("通过Inception-v3模型计算原始图片特征向量:", bottleneck_feature_values)

        # 将计算得到的特征向量存入文件
        bottleneck_string = ",".join(str(x) for x in bottleneck_feature_values)
        with open(bottleneck_feature_file_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
        #print("将计算得到的特征向量存入文件:", bottleneck_feature_file_path)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_feature_file_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_feature_values = [float(x) for x in bottleneck_string.split(",")]
        #print("直接从文件中获取图片相应的特征向量:", bottleneck_feature_values)
    #print()
    return bottleneck_feature_values



# 获取瓶颈层的特征集 构件一个batch
def get_random_cached_bottlenecks(sess, n_classes, image_lists,
                                  batch, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(batch):
        # print("构建batch:", _)
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        # 获取或者创建瓶颈层特征向量
        bottleneck_feature_values = get_or_create_bottleneck(sess, image_lists, label_name,
                                 image_index, category, jpeg_data_tensor, bottleneck_tensor)
        # 构建新的分类标准数据
        new_label_ground_truth = np.zeros(n_classes, dtype=np.float32)
        new_label_ground_truth[label_index] = 1.0

        bottlenecks.append(bottleneck_feature_values)
        ground_truths.append(new_label_ground_truth)
    return bottlenecks, ground_truths


# 获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 迭代所有的类别和每个类别中的测试图片
    for label_index,label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终的数据的列表
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor
            )
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths



def main(_):
    # 读取所有图片
    image_lists = create_iamge_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    # 读取有训练好的Inception-v3模型。谷歌训练好的模型保存在了GraphDef Protocol Buffer中
    # 里面保存了每一个节点取值的计算方法以及变量的取值
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载读取Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME]
    )


    # 定义新的神经网络输入
    # 这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层的节点取值
    # 可以将这个过程类似的理解为一种特征提取
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder'
    )

    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes], name='GroundTruthInput'
    )

    # 定义一个全连接层来解决新的图片分类问题
    # 因为训练好的Inception-v3模型以及将原始的图片抽象为更加容易分类的特征向量了
    # 所以不需要在训练那么复杂的神经网络来完成这个新的分类任务
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001
        ))

        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=ground_truth_input
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1),
                                      tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32)
        )

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        # 训练过程
        for step in range(STEPS):
            # 每次获取一个batch的训练数据
            print("训练步数", step)
            train_bottlenecks, train_ground_truths = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, "training", jpeg_data_tensor, bottleneck_tensor)
            _, loss, train_acc = sess.run([train_step, cross_entropy_mean, evaluation_step],
                                          feed_dict={bottleneck_input: train_bottlenecks,
                                                     ground_truth_input: train_ground_truths})
            time_str = datetime.datetime.now().isoformat()
            print("######### 训练 {}: step {}, loss {:g}, acc {:g}% \n".format(time_str, step, loss, train_acc*100))

            # 在验证数据上测试正确率
            if step % 100 or step + 1 == STEPS:
                validation_bottlenecks, validation_ground_truths = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, "validation", jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={bottleneck_input: validation_bottlenecks,
                                                          ground_truth_input: validation_ground_truths})
                print("$$$$$$$$$$$$$$$$ 验证 {}: Step Validation accuracy on random sampled {} examples = {:g}% "
                      "".format(time_str, step, BATCH, validation_accuracy*100))

            # 在最后的测试数据上测试正确率
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(
                sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor
            )
            test_accuracy = sess.run(evaluation_step,
                                     feed_dict={bottleneck_input: test_bottlenecks,
                                                ground_truth_input: test_ground_truth})

            print("^^^^^^^^^^^^^ 测试 {}:  Final test accuracy = {:g}% ".format(time_str, validation_accuracy*100))











if __name__ == '__main__':
    tf.app.run()
