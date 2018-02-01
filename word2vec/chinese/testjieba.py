# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import codecs

import jieba


def cut_words(sentence):
    return "".join(jieba.cut(sentence)).encode("utf-8")

f = codecs.open('E:\\tensorflow\\word2vec\\chinese\\data\\wiki.zh.tran.text', 'r', encoding="utf8")
target = codecs.open('E:\\tensorflow\\word2vec\\chinese\\data\\wiki.zh.tran.seq.txt', 'w', encoding="utf8")
print('open files')
line_num = 1
line = f.readline()
while line:
    print('------ processing ', line_num, 'article -------------')
    line_seq = " ".join(jieba.cut(line))
    target.writelines(line_seq)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()
exit()