#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: 01_lr_tfidf.py
@time: 2020-03-01 23:56
@description:
"""
import jieba
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer


train=pd.read_csv('data/train.csv',)
test=pd.read_csv('data/test.csv',)
df=pd.concat([train,test])
print(df.head())

print(train.shape[0]-train.count())
print(test.shape[0]-test.count())



train_none=train[train['微博中文内容'].isnull()]