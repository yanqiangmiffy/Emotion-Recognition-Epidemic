#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincyqiang
@license: Apache Licence
@file: 01_lr_tfidf.py
@time: 2020-03-01 23:56
@description:
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer

train=pd.read_csv('data/nCoV_100k_train.labled.csv',encoding='iso-8859-1')
test=pd.read_csv('data/nCov_10k_test.csv',encoding='iso-8859-1')
df=pd.concat([train,test])
print(df.head())

