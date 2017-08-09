#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:58:08 2017

@author: zxf-pc
"""
### 数据预处理部分
import  pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
### 分类器部分
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression##LR二分类
from sklearn.ensemble import BaggingClassifier##bagging提升算法
from sklearn.ensemble import RandomForestClassifier##随机森林
from sklearn.ensemble import GradientBoostingClassifier##GBDT
###评价体系
from sklearn.metrics import accuracy_score##准确率
from sklearn.metrics import recall_score##召回率
from sklearn.metrics import precision_score##精确率
import matplotlib.pyplot as plt
### 分区函数
def split(file,input_list):
    total_list=[]
    for i in range(len(input_list)):
        temp_list=[]
        temp_file=file.loc[input_list[i]]
        print(i)
        temp_list.append(temp_file)
        total_list.append(temp_list)
    return total_list
### 补值函数有
def get_mean(file):
    file=pd.concat(file)
    imp=Imputer(missing_values=np.nan, strategy='mean', axis=0)###这种转换函数必须训
    imp.fit(file)
    new_file=imp.transform(file)
    return new_file
### 筛选出时间列，并暂时做删除处理,返回不含有时间项
def del_time(file,input_list):
    temp_list=[]
    for i in range(len(input_list)):
        if '时间' in  input_list[i]:
            temp_list.append(input_list[i])
    for l in range(len(temp_list)):
        file=file.drop(temp_list[l],axis=1)###删除时间项目
    return file
### 查看字段类型并暂时去除字符字段
def search_type(file):
    type_list=[]
    for i in range(len(file.columns)):
        type_list.append(type(file.iloc[0,i]))
    for l in range(len(type_list)):
        if type_list[l]==str:
            file=file.drop(type_list[l],axis=1)
    return file
#######缩放函数
def change(probility,train_file):
    decision_prb=(probility[:,1]/probility[:,0])*(train_file.count(0)/train_file.count(1))
    return decision_prb
############结果函数
def Print_answer(train_predict_prb,name):
    temp_train_decision_label=[]
    name=str(name)
    for i in range(len(train_predict_prb)):
        if train_predict_prb[i]>0.9:
            temp_train_decision_label.append(1)
        if train_predict_prb[i]<0.9:
            temp_train_decision_label.append(0)
    print('%s answer'%name)
    recall=recall_score(y_test,temp_train_decision_label)
    print('recall is %.4f'%recall)
    accuracy=accuracy_score(y_test,temp_train_decision_label)
    print('accuracy is %.4f'%accuracy)
    precision=precision_score(y_test,temp_train_decision_label)
    print('precision is %.4f'%precision)
    F1=2*precision*recall/(precision+recall)
    print('F1 is %.4f'%F1)
    #return temp_train_decision_label
###########
df_train_file=pd.read_csv('/Users/zxf-pc/Desktop/天池文件/df_train.csv'
                          ,index_col='顺序号')
df_id_train_file=pd.read_csv('/Users/zxf-pc/Desktop/天池文件/df_id_train.csv')
df_id_test_file=pd.read_csv('/Users/zxf-pc/Desktop/天池文件/df_id_test.csv')
train_file_title=df_train_file.columns
### 去掉全部为Nan的列
df_train_file.dropna(how='all',axis=1,inplace=True)

### 排序，将重复项放在一起
transform_train_file=df_train_file.sort_values(by='个人编码')
### 获得独立的个人id
single_id = list(set(transform_train_file['个人编码']))
###########直接在总表上去掉时间项和str项
train_start_time=time.time()
del_time_file=del_time(transform_train_file,train_file_title)
zuizong_file=search_type(del_time_file)
########人为去除错误列
del zuizong_file['出院诊断病种名称']
#zuizong_file.values().astype('float64')
#############
print('get values variable')
##################转换index
zuizong_file.index=zuizong_file['个人编码']
del zuizong_file['个人编码']
#####个人编码还是出问题，人为改变  7月20日
fenqu_list=split(zuizong_file,single_id)
print('finish split')
### 进行分区均值代替
zui_zong_train_list=[]
#######以下均值替代有问题，准备人工删除问题列56，58，59，'出院诊断病种名称'
for i in range(len(fenqu_list)):
    if pd.DataFrame(fenqu_list[i]).shape[1]==59:
        temp_index=pd.DataFrame(fenqu_list[i]).index
        temp_list=pd.DataFrame(fenqu_list[i]).mean(axis=0)
        temp_list1=pd.DataFrame(temp_list,columns=temp_index).T
        zui_zong_train_list.append(temp_list1)
    else:
        temp_index=list(set(pd.concat(fenqu_list[i]).index))
        temp_list=pd.concat(fenqu_list[i]).mean(axis=0)
        temp_dataframe=pd.DataFrame(temp_list,columns=temp_index).T
        zui_zong_train_list.append(temp_dataframe)
    print('finish%dconcating'%i)
for i in range(1,len(fenqu_list)):
    if pd.concat([zui_zong_train_list[i-1],zui_zong_train_list[i]]).shape[1]>59:
        print(i)
final_train_file=pd.concat(zui_zong_train_list,axis=0,ignore_index=False)
final_train_file=final_train_file.sort_index()
####################
print('finsh concat')
#train_end_time=time.time()
#print(train_end_time-train_start_time)
#############处理标签
loss_id=df_id_train_file.columns
add_id=list(loss_id)
df_id_train_file.columns=['个人编码','判断']
#person_index=[]
add_id=[int(x) for x in add_id]
df_id_train_file.ix[len(df_id_train_file)+1]=add_id
df_id_train_file=df_id_train_file.sort_values(by='个人编码')
########原始数据上加上标签
final_train_file['判断']=df_id_train_file['判断']
label=list(df_id_train_file['判断'])
final_train_file['判断']=label
##########
print('finish adding label')
### 数据准备完成后，先用IMpute进行简单的插值
imp=Imputer(missing_values='NaN',strategy='mean',axis=0)
imp=imp.fit(final_train_file)
label_train_file=imp.transform(final_train_file)
finally_train_file=pd.DataFrame(label_train_file,columns=final_train_file.columns,index=final_train_file.index)
########
#train_end_time=time.time()
#print(train_end_time-train_start_time)
X_file=finally_train_file.drop('判断',axis=1)
###使用svm为器学习的话太慢，超过10000的数据对就不要使用svm了
### 使用逻辑回归进行学习，判断。（没有进行隔点调参）
### 不降维情况
X=StandardScaler().fit_transform(X_file)
###数据标准话
x_train,x_test,y_train,y_test=train_test_split(X,label,train_size=0.8)
### LR分类器,这里注意到反例，即0分类较多，需要进行分类不平衡问题
### 使用缩放策略增加了标签1的数量，但这样损失了accuracy_score,不知道如何取舍
'''
print('begin LR')
LR_clf=LogisticRegression(penalty='l2',solver='sag', tol=0.0001,max_iter=500,n_jobs=-1)
LR_clf=LR_clf.fit(x_train,y_train)
LR_predict=LR_clf.predict(x_test)
train_predict_prb=LR_clf.predict_proba(x_test)
###考虑分类不平衡问题，使用再放缩(p[:,1]/p[:,0])*(m+/m-)
decision_prb=(train_predict_prb[:,1]/train_predict_prb[:,0])*(y_train.count(0)/y_train.count(1))
###
train_decision_label=[]
for i in range(len(train_predict_prb)):
    if decision_prb[i]>1:
        train_decision_label.append(1)
    if decision_prb[i]<1:
        train_decision_label.append(0)
print('LR answer','\n')
recall=recall_score(y_test,train_decision_label)
print('recall is %.4f'%recall)
accuracy=accuracy_score(y_test,train_decision_label)
print('accuracy is %.4f'%accuracy)
precision=precision_score(y_test,train_decision_label)
print('precision is %.4f'%precision)
F1=2*precision*recall/(precision+recall)
print('F1 is %.4f'%F1)
print('No change answer','\n')
print(LR_clf.score(x_test,y_test))
print('%.4f'%recall_score(y_test,LR_predict))
print('%.4f'%precision_score(y_test,LR_predict))
#LRF1=2*recall_score(y_test,LR_predict)*precision_score(y_test,LR_predict)/(recall_score(y_test,LR_predict)+precision_score(y_test,LR_predict))
#print('%.4f'%LRF1)
### 使用bagging提升LR算法
'''
'''
bagging_LR_clf=BaggingClassifier(LogisticRegression(penalty='l2',solver='sag',
    tol=0.0001,max_iter=500,n_jobs=-1),n_estimators=100,max_samples=1.0, max_features=1.0)
bagging_LR_clf=bagging_LR_clf.fit(x_train,y_train)
bagging_LR_clf_predict=bagging_LR_clf.predict_proba(x_test)
bagging_LR_clf_prob=change(bagging_LR_clf_predict,y_train)
#Print_answer(bagging_LR_clf_prob,'bagging_LR_clf')
'''
### 使用隔点调参的LR
'''
print('begin GridLR')
clf_param={'penalty':['l2'],'solver':['liblinear','newton-cg','lbfgs','sag'],'max_iter':[300,500,800,1000,1500],'tol':[1e-3,1e-4,1e-5]}
GridSearch_clf=GridSearchCV(LogisticRegression(),clf_param,cv=10)
GridSearch_clf=GridSearch_clf.fit(x_test,y_test)
GridSearch_clf_prediction=GridSearch_clf.predict(x_test)
GridSearch_clf_predict=GridSearch_clf.predict_proba(x_test)
GridSearch_clf_predict_prob=change(GridSearch_clf_predict,y_train)
Print_answer(GridSearch_clf_predict_prob,'GridSearch_clf')
print('no change answer')
GridSearch_clf_recall=recall_score(y_test,GridSearch_clf_prediction)
print('%.4f'%GridSearch_clf_recall)
GridSearch_clf_precise=precision_score(y_test,GridSearch_clf_prediction)
print('%.4f'%GridSearch_clf_precise)
GridSearch_clf_F1=2*GridSearch_clf_precise*GridSearch_clf_recall/(GridSearch_clf_precise+GridSearch_clf_recall)
print('%.4f'%GridSearch_clf_F1)
'''
### 使用随机森林，目前效果最好
print('begin randomforest')#
###初始设定量为n_estimators=15000,amx_depth=100
###开始调整（12000，200）目前0.33最高
RandomForest_clf=RandomForestClassifier(n_estimators=8000,max_features='auto',max_depth=200,n_jobs=-1)
RandomForest_clf=RandomForest_clf.fit(x_train,y_train)
RandomForest_clf_predict=RandomForest_clf.predict_proba(x_test)
RandomForest_clf_decision_prb=change(RandomForest_clf_predict,y_train)
RandomForest_clf_prediction=RandomForest_clf.predict(x_test)
Print_answer(RandomForest_clf_decision_prb,'RandomForest')
print(RandomForest_clf.score(x_test,y_test))
print('no change answer','\n')
print('%.4f'%recall_score(y_test,RandomForest_clf_prediction))
print('%.4f'%precision_score(y_test,RandomForest_clf_prediction))
Randomforest_F1=2*recall_score(y_test,RandomForest_clf_prediction)*precision_score(y_test,RandomForest_clf_prediction)/(precision_score(y_test,RandomForest_clf_prediction)+recall_score(y_test,RandomForest_clf_prediction))
print(Randomforest_F1)
###使用GBDT 第一轮使用的是默认值
'''
print('begin GBDT')
GBDT_clf=GradientBoostingClassifier(loss='exponential',n_estimators=15000,max_depth=200)
GBDT_clf=GBDT_clf.fit(x_train,y_train)
GBDT_clf_predict_proba=GBDT_clf.predict_proba(x_test)
GBDT_clf_predict=GBDT_clf.predict(x_test)
GBDT_clf_predict_proba=change(GBDT_clf_predict_proba,y_train)
Print_answer(GBDT_clf_predict_proba,'GBDT')
print('no change answer','\n')
print('%.4f'%recall_score(y_test,GBDT_clf_predict))
print('%.4f'%precision_score(y_test,GBDT_clf_predict))
GBDT_F1=2*recall_score(y_test,GBDT_clf_predict)*precision_score(y_test,GBDT_clf_predict)/(precision_score(y_test,GBDT_clf_predict)+recall_score(y_test,GBDT_clf_predict))
print(GBDT_F1)
'''
### 评价结果输出
#Print_answer(decision_prb,'LR')
#Print_answer(bagging_LR_clf_prob,'bagging_LR_clf')
#Print_answer(GridSearch_clf_predict,'GridSearch_clf')
train_end_time=time.time()
print(train_end_time-train_start_time)