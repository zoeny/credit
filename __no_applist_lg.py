# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 9:23
@month : 八月
@Author : mhm
@FileName: __no_applist_lg.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('no_missing_data.csv')

train_df = data[data.y_bad.isnull() == False]
test_df = data[data.y_bad.isnull() == True]
target = train_df['y_bad']
features = np.array([c for c in train_df.columns if c not in ['y_bad','loanID','uuid']])

X = train_df[features]

def modelfit(dtrain,dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.25, random_state=10)
        lr = LogisticRegression(C=0.01,penalty='l2',class_weight='balanced')
        lr.fit(X_train, y_train)
        dtrain_predictions = lr.predict(X_test)
        dtrain_predprob = lr.predict_proba(X_test)[:, 1]
        prob = lr.predict_proba(test_df[features])[:, 1]
        print("Accuracy : %.4g" % metrics.accuracy_score(y_test, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, dtrain_predprob))
    return prob

prob = modelfit(train_df,test_df, features)
submit = pd.DataFrame({'loanID':test_df.loanID,'score':prob})
submit.score = submit.score.apply(lambda x:round(x,6))
submit.to_csv('submit/_no_applist_submit_lg.csv',index=False)