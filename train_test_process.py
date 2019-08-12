# -*- coding: utf-8 -*-
'''
@project: PY_project
@Time : 2019/8/2 22:21
@month : 八月
@Author : mhm
@FileName: try1.py
@Software: PyCharm
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读入数据
risk_train = pd.read_excel('data_info.xlsx',sheetname='risk_train')
risk_test = pd.read_excel('data_info.xlsx',sheetname='risk_test')

print("训练数据集大小:",risk_train.shape)
print("测试数据集大小:",risk_test.shape)

'''一、对训练集和测试集进行预处理'''
# 1.将训练数据和测试数据拼接
all_data = pd.concat((risk_train,risk_test),axis=0,ignore_index=True,sort=False)

# 2.将applied_from、applied_type类别变量哑变量化
all_data = pd.concat([all_data.drop(['applied_from','applied_type'],axis=1),pd.get_dummies(all_data[['applied_from','applied_type']])],axis=1)

## 缺失值的处理
# 1.将缺失值个数作为一列统计特征
all_data.replace(-99999,np.nan,inplace=True)
all_data['na_count'] = all_data.shape[1] - all_data.drop('y_bad',axis=1).count(axis=1)-1

# 2.查看每维特征缺失值分布情况
'''根据分布图可看出'app1254','app1176','app0329','app0973','active0476'的缺失值高达80%以上'''
y = [i/all_data.shape[0] for i in np.array(all_data.drop('y_bad',axis=1).isnull().sum())]
x = [i for i in range(len(all_data.drop('y_bad',axis=1).columns))]
plt.bar(x,y,width=0.5,color='r')
_xticks_labels = [str(index + 1) + " " + value for index, value in enumerate(np.array(all_data.drop('y_bad',axis=1).columns))]
plt.xticks(x, _xticks_labels, rotation=100, fontsize=8)
plt.title("各列缺失值情况分布")
plt.xlabel("特征名", color='b')
plt.ylabel("频率", color='black')
plt.savefig('./特征缺失值分布图.jpg')
# plt.show()

# 对这几列进行操作：空值用0填充，几列相加之后归一化
sub_df = all_data[['app1254','app1176','app0329','app0973','active0476']]
sub_df = sub_df.fillna(0)
sub_df['app_sum'] = sub_df.apply(lambda x:x.sum(),axis=1)
max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
all_data['app_sum'] = sub_df[['app_sum']].apply(max_min_scaler)

def drop_col(data,col):
    data = data.drop(col,1)
    return data
all_data = drop_col(all_data,['app1254','app1176','app0329','app0973','active0476'])

# 删除取值唯一的特征cell_micall123
all_data.drop('cell_micall123',axis=1,inplace=True)

# 对'cell_allFlow','cell_callno','cell_meanMonCallno','cellDate','tel_maxCPay','telRemain','contactsInBlack_blackOrgNum''contactsXyqbRegisteredUserNumRct','delq_days_max','fst_apply_day','repay_amt_sum','re_pas','last_loan_day'归一化
sub_df1 = all_data[['cell_allFlow','cell_callno','cell_meanMonCallno','cellDate','tel_maxCPay','telRemain','contactsInBlack_blackOrgNum','contactsXyqbRegisteredUserNumRct','delq_days_max','fst_apply_day','repay_amt_sum','re_pas','last_loan_day']]
all_data[['cell_allFlow','cell_callno','cell_meanMonCallno','cellDate','tel_maxCPay','telRemain','contactsInBlack_blackOrgNum','contactsXyqbRegisteredUserNumRct','delq_days_max','fst_apply_day','repay_amt_sum','re_pas','last_loan_day']] = sub_df1.apply(max_min_scaler)

# 保存处理后的结果
all_data.to_csv('./train_test.csv',index=False)

# # 2.合并applist和all_data
# df = pd.merge(all_data,applist,left_on='uuid',right_on='userid',how='left')
# df = drop_col(df,['userid','createtime','applied_at'])
# print(df.shape)

# # 3.缺失值用-99999进行填充
# new_df = df.fillna(-99999)
# # train_df = new_df[new_df.y_bad != -99999]
# # test_df = new_df[new_df.y_bad == -99999]