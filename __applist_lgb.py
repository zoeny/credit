# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/4 16:16
@month : 八月
@Author : mhm
@FileName: __applist_lgb.py
@Software: PyCharm
'''
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np

import warnings
warnings.filterwarnings('ignore')

data2  = pd.read_csv('./train_test_applist.csv')
train_df = data2[data2.y_bad.isnull() == False]
test_df = data2[data2.y_bad.isnull() == True]
data2.to_csv('./no_missing_data.csv',index=False)

target = train_df['y_bad']
features = np.array([c for c in train_df.columns if c not in ['y_bad','loanID','uuid']])

# 数据拆分(训练集+验证集+测试集)
print('拆分数据集')
train, val = train_test_split(train_df, test_size=0.2, random_state=21)

# 训练集
y_train = train.y_bad
X_train = train[features]

# 验证集
y_val = val.y_bad
X_val = val[features]

# 测试集
test_X = test_df[features]

# 数据转换

# lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,free_raw_data=False)

def modelfit(alg, dtrain,dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        lgb_param = {'learning_rate':0.01,
                'n_estimators':1000,
                'max_depth':6,
                'num_leaves':50,
                'min_child_weight':1,
                'gamma':0,
                'subsample':0.8,
                'colsample_bytree':0.8,
                'objective=':'binary',
                'scale_pos_weight':1,
                'metrics':'auc',
                'seed':27}
        lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
        cvresult = lgb.cv(lgb_param, lgb_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        # alg.set_params(n_estimators=cvresult.shape[0])
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

lgb1 = LGBMClassifier(
                learning_rate =0.01,
                n_estimators=1000,
                max_depth=6,
                num_leaves=50,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary',
                nthread=4,
                scale_pos_weight=1,
                metrics='auc',
                seed=27)

# 预测
pred = modelfit(lgb1, train_df,test_df, features)

submit = pd.DataFrame({'loanID':test_df.loanID,'score':pred})
submit.score = submit.score.apply(lambda x:round(x,6))
submit.to_csv('submit/_applist_submit_lgb.csv',index=False)


