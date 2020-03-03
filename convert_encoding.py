#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: convert_encoding.py 
@time: 2020-03-02 22:07
@description:
"""
import pandas as pd
import csv

with open('data/nCoV_100k_train.labled.csv', encoding='gb18030', errors='ignore') as f:
    data_list = []
    csv_data = csv.reader(f)
    for item in csv_data:
        data_list.append(item)
    df=pd.DataFrame(data_list[1:],columns=data_list[0])
    df.to_csv('data/train.csv',index=False)

with open('data/nCov_10k_test.csv', encoding='gb18030', errors='ignore') as f:
    data_list = []
    csv_data = csv.reader(f)
    for item in csv_data:
        data_list.append(item)
    df=pd.DataFrame(data_list[1:],columns=data_list[0])
    df.to_csv('data/test.csv',index=False)