# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 9:51
@month : 八月
@Author : mhm
@FileName: __no_applist_xgb.py
@Software: PyCharm
'''
import pandas as pd
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate,GridSearchCV


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('no_missing_data.csv')

train_df = data[data.y_bad.isnull() == False]
test_df = data[data.y_bad.isnull() == True]

target = train_df['y_bad']
features = np.array([c for c in train_df.columns if c not in ['y_bad','loanID','uuid']])


def modelfit(alg, dtrain,dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        print(xgb_param)
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], target, eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
    print(alg)
    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(target.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(target, dtrain_predprob))
    return dtest_predprob

xgb1 = XGBClassifier(
                learning_rate =0.1,
                n_estimators=1000,
                max_depth=18,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27)
pred = modelfit(xgb1, train_df,test_df, features)
submit = pd.DataFrame({'loanID':test_df.loanID,'score':pred})
submit.score = submit.score.apply(lambda x:round(x,6))

import os
if not os.path.exists('./submit'):
    os.mkdir('./submit')

submit.to_csv('submit/_no_applist_submit_xgb.csv',index=False)

