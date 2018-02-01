# __author__ = 'tonye0115'
# -*- coding: utf-8 -*-
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s %(message)s', level=logging.INFO)
raw_sentences = ["the quick brown fox jumps over the lazy dogs", "youyou you do home now to sleep"]
sentences = [s.split() for s in raw_sentences]
print(sentences)

model = word2vec.Word2Vec(sentences, min_count=1)

print(model.similarity('dogs', 'you'))