# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
import codecs

f = codecs.open('E:\\tensorflow\\word2vec\\chinese\\data\\wiki.zh.tran.seq.txt', 'r', encoding="utf8")
line = f.readline()
print(line)