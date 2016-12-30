# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import os
import re
import numpy as np


def findSite(data_path, index):
    """
    根据索引返回网店site
    :param data_path:
    :param index:
    :return:
    """
    files = os.listdir(data_path)
    site = ""
    for i, file in enumerate(files):
        if index == i:
            site = os.path.splitext(file)[len(os.path.splitext(file))-1][1:]
            break
        else:
            site = "none"
    return site



def splitWord(examples):
    x_text = []
    examples = [clean_str(s) for s in examples]
    for example in examples:
        tmp = ""
        for s in example:
            tmp +=(s+"\t")
        x_text.append(tmp)
    return x_text


def load_data_and_labels(data_path):
    """
    从文件加载数据,将数据分为词汇和生成标签。返回分割句子和标签
    :param postive_data_file:
    :param negative_data_file:
    :return:
    """
    files = os.listdir(data_path)
    x_text = []
    labels = []
    for i, file in enumerate(files):
        print(file)
        fullname = os.path.join(data_path, file)
        examples = list(open(fullname, 'r', encoding='utf-8').readlines())
        x_text.extend(splitWord(examples))
        arr = np.zeros(len(files))
        arr[i] = 1
        labels.append([arr for _ in examples])
    y = np.concatenate(labels)
    print("x_text:", x_text)
    print("y:", y)
    return [x_text, y]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\t", "", string)
    string = re.sub(r"\,", "", string)
    string = re.sub(r"\，", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\（", "", string)
    string = re.sub(r"\）", "", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



if __name__ == '__main__':
    print(findSite("data", 1))