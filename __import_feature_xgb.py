# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 16:56
@month : 八月
@Author : mhm
@FileName: __import_feature_xgb.py
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
import pickle
warnings.filterwarnings('ignore')

import os
if not os.path.exists('./model'):
    os.mkdir('./model')

data  = pd.read_csv('./train_test_applist.csv')
import_feature = ['re_pas','app_count','cell_relateCellRatio','tel_maxCPay','cell_callno','cell_micallR16',
                  'fst_apply_day','telRemain','repay_amt_sum','last_loan_day',
                  'cell_miflowR25','cell_allFlow','cellDate','contactsXyqbRegisteredUserNumRct',
                  'delq_days_max','na_count','cell_meanMonCallno','contactsInBlack_blackOrgNum']

train_df = data[data.y_bad.isnull() == False]
test_df = data[data.y_bad.isnull() == True]

target = train_df['y_bad']
features = np.array([c for c in train_df.columns if c in import_feature])

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
    pickle.dump(alg,open('./model/xgb.pickle.dat','wb'))
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

submit.to_csv('submit/_importfeat_submit_xgb.csv',index=False)













