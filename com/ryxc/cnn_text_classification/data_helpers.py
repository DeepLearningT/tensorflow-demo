# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import re
import numpy as np

def load_data_and_labels(postive_data_file, negative_data_file):
    """
    从文件加载数据,将数据分为词汇和生成标签。返回分割句子和标签
    :param postive_data_file:
    :param negative_data_file:
    :return:
    """
    # load data from files
    postive_examples = list(open(postive_data_file, 'r', encoding="utf-8").readlines())
    postive_examples = [s.strip() for s in postive_examples]  # 删除 字符串中开头、结尾处的 默认删除空白符（包括'\n', '\r',  '\t',  ' ')
    # for s in postive_examples:
    #     print(s.encode("utf-8", 'ignore'))
    negative_examples = list(open(negative_data_file, 'r', encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = postive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # for s in x_text:
    #     print(s.encode("utf-8", 'ignore'))

    # Generate labels
    postive_labels = [[0, 1] for _ in postive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([postive_labels, negative_labels])
    # print(y)
    return [x_text, y]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    a = "for 我 more than two decades mr . ";
    print(a)

    zeros = np.zeros(2, np.int64)
    print(zeros)

    a = [1,2,3,4,5]
    print(a[:-2])
    print(a[-2:])





