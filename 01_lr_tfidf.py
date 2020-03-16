#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: 01_lr_tfidf.py
@time: 2020-03-01 23:56
@description:
"""
import re
import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tqdm.pandas()

train = pd.read_csv('data/train.csv', )#  (100000, 7)

train = train[~train['情感倾向'].isnull()] #(99919,7)
train = train[train['情感倾向'].isin(['-1','0','1'])] # (99913,7)
test = pd.read_csv('data/test.csv', )
df = pd.concat([train, test])
df['微博中文内容'] = df['微博中文内容'].astype(str)
train_size = len(train)

label_map = {'-1': 0, '0': 1, '1': 2}
label_map_reverse = {0: '-1', 1: '0', 2: '1'}


# Out[3]: Index(['微博id', '微博发布时间', '发布人账号', '微博中文内容', '微博图片', '微博视频', '情感倾向'], dtype='object')

# print(df.head())
# print(train.shape[0]-train.count())
# print(test.shape[0]-test.count())
# train_none=train[train['微博中文内容'].isnull()]
def remove_punctuation(line):
    rule = re.compile("[^\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line

def token(s):
    """
    实现分词
    :param s:
    :return:
    """
    return " ".join(jieba.cut(remove_punctuation(s)))


corpus = df['微博中文内容'].progress_apply(lambda x: token(x)).values.tolist()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.95,max_features=100000 )
X = vectorizer.fit_transform(corpus)
x_train, x_test = X[:train_size], X[train_size:]

y_train = train['情感倾向'].apply(lambda x: label_map[x])

lr = LogisticRegression()
lr.fit(x_train, y_train)
prob = lr.predict_proba(x_test)

test['情感倾向'] = np.argmax(prob, axis=1)
test['y'] = test['情感倾向'].apply(lambda x: label_map_reverse[x])

test[['微博id','y']].to_csv('result/lr.csv',header=['id','y'],index=None)