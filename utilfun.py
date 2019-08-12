# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 8:50
@month : 八月
@Author : mhm
@FileName: utils.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np

def fillna_mean(data, cols):
    for col in cols:
        data[col] = data[col].fillna(np.mean(data[col]))
    return data

def fillna_min(data, cols):
    for col in cols:
        data[col] = data[col].fillna(np.min(data[col]))
    return data

def neg_to_zero(data, cols):
    for col in cols:
        data[col] = data[col].apply(lambda x: 0 if x < 0 else x)
    return data

def neg_to_mean(data, cols):
    for col in cols:
        mean_col = np.mean(data[col][data[col] > 0])
        data[col] = data[col].apply(lambda x: mean_col if x < 0 else x)
    return data