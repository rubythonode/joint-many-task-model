import pandas as pd
from gensim.models import Word2Vec
import multiprocessing
import numpy as np
from itertools import chain
from collections import namedtuple
import json


word_level = dict()
sent_level = dict()
train_chunk = open('data/train.txt').readlines()
test_chunk = open('data/test.txt').readlines()
sent, t1, t2, t3, pos, chunk = [], [], [], [], [], []
for i in train_chunk:
    if i == '\n':
        sent.append(t1)
        pos.append(t2)
        chunk.append(t3)
        t1, t2, t3 = [], [], []
    else:
        t1.append(i.split()[0].lower())
        t2.append(i.split()[1])
        t3.append(i.split()[2])

for i in test_chunk:
    if i == '\n':
        sent.append(t1)
        pos.append(t2)
        chunk.append(t3)
        t1, t2, t3 = [], [], []
    else:
        t1.append(i.split()[0].lower())
        t2.append(i.split()[1])
        t3.append(i.split()[2])
sent = pd.Series(sent)
word_level['pos'], word_level['chunk'] = np.array(pos), np.array(chunk)
word_level['sent'] = sent
i2w = dict(enumerate(set(chain(*sent))))
word_level['i2p'] = dict(enumerate(set(chain(*pos))))
word_level['i2c'] = dict(enumerate(set(chain(*chunk))))
word_level['p2i'] = {v: k for k, v in word_level['i2p'].items()}
word_level['c2i'] = {v: k for k, v in word_level['i2c'].items()}
w2i = {v: k for k, v in i2w.items()}
for i, _ in enumerate(word_level['pos']):
    for j, _ in enumerate(word_level['pos'][i]):
        word_level['pos'][i][j] = word_level[
            'p2i'][word_level['pos'][i][j]]
for i, _ in enumerate(word_level['chunk']):
    for j, _ in enumerate(word_level['chunk'][i]):
        word_level['chunk'][i][j] = word_level[
            'c2i'][word_level['chunk'][i][j]]

sick = pd.read_csv('data/SICK.txt', sep='\t')
sent = pd.concat(
    [pd.concat([sick.sentence_A, sick.sentence_B]).apply(split), word_level['sent']])
i2w = dict(enumerate(set(chain(*sent))))
w2i = {v: k for k, v in i2w.items()}

model = Word2Vec(sent, min_count=1, size=300, sg=1, iter=25, negative=128, workers=multiprocessing.cpu_count(),
                 window=2, batch_words=500)

for i, _ in enumerate(word_level['sent']):
for j, _ in enumerate(word_level['sent'][i]):
    word_level['sent'][i][j] = w2i[word_level['sent'][i][j]]

sent_level['sent1'], sent_level['sent2'] = sick.sentence_A.apply(
    split), sick.sentence_B.apply(split)
sent_level['i2e'] = dict(enumerate(set(sick.entailment_label)))
sent_level['e2i'] = {v: k for k, v in sent_level['i2e'].items()}
sent_level['rel'] = sick.relatedness_score

for i, _ in enumerate(sent_level['sent1']):
    for j, _ in enumerate(sent_level['sent1'][i]):
        sent_level['sent1'][i][j] = w2i[sent_level['sent1'][i][j]]

for i, _ in enumerate(sent_level['sent2']):
    for j, _ in enumerate(sent_level['sent2'][i]):
        sent_level['sent2'][i][j] = w2i[sent_level['sent2'][i][j]]

sent_level['entailment'] = sick.entailment_label.apply(
    lambda x: sent_level['e2i'][x])


vec = []
for i in range(len(i2w)):
    vec.append(model[i2w[i]])

data = {'i2w': i2w, 'w2i': w2i, 'word_level': word_level,
        'vec': vec, 'sent_level': sent_level}
np.savez('data.npz', data=data)
