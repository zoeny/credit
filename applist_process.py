# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/3 19:07
@month : 八月
@Author : mhm
@FileName: applist_process.py
@Software: PyCharm
'''
'''处理applist'''
import pandas as pd
import numpy as np
import os
from word_similer_values import *

import warnings
warnings.filterwarnings("ignore")

applist = pd.read_excel('data_info.xlsx',sheetname='applist')
print(applist.shape)

# 1.删除applist重复的数据
applist.dropna(axis=0,how='any',inplace=True)
applist = applist.sort_values(by='createtime', ascending=False).drop_duplicates('userid').reset_index(drop=True)
print(applist.shape)

# 4.统计app个数
applist['app'] = applist.applist.apply(lambda x:x.split('##'))
applist['app_count'] = applist.app.apply(lambda x:len(x))

def count_similer(word_arr,word):
    values = []
    for i in range(len(word_arr)):
        values.append(SIM().Baidu_simi(word_arr[i],word))
    return values

def mean(arr):
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
    return sum/len(arr)

applist['app_sim'] = applist.app.apply(lambda x:count_similer(x,'拍拍贷'))
applist['app_sim'] = applist.app_sim.apply(lambda x:mean(x))

applist.to_csv('./new1_applist.csv',index=False)