# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import collections


def preData():
    poetry_file = 'data/poetry.txt'
    # 诗集
    poetrys = []
    with open(poetry_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                split = line.strip().split(':')
                title = split[0]
                content = split[1].replace(' ', '')
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                pass

    print(poetrys)

    # 按诗的字数排序
    poetrys = sorted(poetrys, key=lambda line: len(line))
    print("唐诗总数:", len(poetrys))

    # 统计每个字出现的次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)


    # 取前多少个常用字
    words = words[:len(words)] + (' ',)

    # 每个字映射为一个数字ID
    word_num_map = dict(zip(words, range(len(words))))
    print(word_num_map)
    return words, poetrys, word_num_map

