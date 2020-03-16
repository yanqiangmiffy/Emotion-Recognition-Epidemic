#!usr/bin/env python
# -*- coding:utf-8 _*-
"""
@version:
author:yanqiang
@time: 2019/04/09
@file: main.py
@description: 
"""
from sklearn.metrics import *
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow_text

from tqdm import tqdm
import numpy as np
import pandas as pd
import re

tqdm.pandas()
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
print(embed)
train = pd.read_csv('data/train.csv', )  # (100000, 7)

train = train[~train['情感倾向'].isnull()]  # (99919,7)
train = train[train['情感倾向'].isin(['-1', '0', '1'])]  # (99913,7)
test = pd.read_csv('data/test.csv', )
df = pd.concat([train, test])
df['微博中文内容'] = df['微博中文内容'].astype(str)
train_size = len(train)

label_map = {'-1': 0, '0': 1, '1': 2}
label_map_reverse = {0: '-1', 1: '0', 2: '1'}


def remove_punctuation(line):
    rule = re.compile("[^\u4e00-\u9fa5]")
    # print(line)
    line = rule.sub('', str(line))
    return line


train['微博中文内容'] = train['微博中文内容'].progress_apply(remove_punctuation)
test['微博中文内容'] = test['微博中文内容'].progress_apply(remove_punctuation)


X_train = []
for r in tqdm(train['微博中文内容'].values):
    review_emb = tf.reshape(embed, [-1]).numpy()
    X_train.append(review_emb)

X_train = np.array(X_train)
y_train = train['情感倾向'].apply(lambda x: label_map[x])

X_test = []
for r in tqdm(test['微博中文内容'].values):
    review_emb = tf.reshape(embed, [-1]).numpy()
    X_test.append(review_emb)

X_test = np.array(X_test)

train_arrays, test_arrays, train_labels, test_labels = train_test_split(X_train, y_train, test_size=0.05)


def svc_param_selection(X, y, nfolds):
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    Cs = [1.070, 1.074, 1.075, 1.1, 1.125]
    # gammas = [0.001, 0.01, 0.1, 1]
    gammas = [2.065, 2.075, 2.08]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds, n_jobs=8)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search


model = svc_param_selection(train_arrays, train_labels, 5)
pred = model.predict(test_arrays)

cm = confusion_matrix(test_labels,pred)
print(cm)