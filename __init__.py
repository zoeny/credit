# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/3 0:40
@month : 八月
@Author : mhm
@FileName: __init__.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np
import os
from word_similer_values import *

import warnings
warnings.filterwarnings("ignore")

applist = pd.read_excel('data_info.xlsx',sheetname='applist')
applist.dropna(axis=0,how='any',inplace=True) # 删除缺失值

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

a = pd.DataFrame({'id':[1,2,3],'applist':[['爱奇艺','点融网','现金贷','美颜相机'],['饿了吗','极速贷','暴风影音'],['百度地图','融360','捕鱼神器','王者荣耀','吃鸡']],'appcount':[4,6,9]})
print(a)

applist.drop(['userid','applist','createtime'],axis=1,inplace=True)
print(applist.head())

a['app_sim'] = a.applist.apply(lambda x:count_similer(x,'拍拍贷'))
a['app_sim'] = a.app_sim.apply(lambda x:mean(x))
# v = count_similer(a,'拍拍贷')
# print(v)
print(a)




