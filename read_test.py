import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
################
def del_time(file,input_list):
    temp_list=[]
    for i in range(len(input_list)):
        if '时间' in  input_list[i]:
            temp_list.append(input_list[i])
    for l in range(len(temp_list)):
        file=file.drop(temp_list[l],axis=1)###删除时间项目
    return file
###############
def search_type(file):
    type_list=[]
    for i in range(len(file.columns)):
        type_list.append(type(file.iloc[0,i]))
    for l in range(len(type_list)):
        if type_list[l]==str:
            file=file.drop(type_list[l],axis=1)
        else:
            file=file
    return file
#################
def split(file,input_list):
    total_list=[]
    for i in range(len(input_list)):
        temp_list=[]
        temp_file=file.loc[input_list[i]]
        print(i)
        temp_list.append(temp_file)
        total_list.append(temp_list)
    return total_list
df_test_file=pd.read_csv('/Users/zxf-pc/Desktop/天池/df_test.csv',index_col='顺序号')
df_id_test_file=pd.read_csv('/Users/zxf-pc/Desktop/天池/df_id_test.csv')
########目标个人编码columns处理,以下仍有小问题
test_id=df_id_test_file.columns
df_id_test_file.columns=['个人编码']
df_id_test_file.ix[len(df_id_test_file)]=test_id
temp_id_datafram=pd.DataFrame(df_id_test_file).T
temp_id_datafram.insert(0,'a',int(temp_id_datafram.iloc[:,3999]))
del temp_id_datafram[3999]
person_id=temp_id_datafram.iloc[0,:]
person_id.index=range(len(temp_id_datafram.columns))
##########
test_file_title=df_test_file.columns
df_test_file.dropna(how='all',axis=1,inplace=True)
transform_test_file=df_test_file.sort_values(by='个人编码')
single_id = list(set(transform_test_file['个人编码']))
train_start_time=time.time()
del_time_file=del_time(transform_test_file,test_file_title)
zuizong_file=search_type(del_time_file)
del zuizong_file['出院诊断病种名称']
##################
zuizong_file.index=zuizong_file['个人编码']
del zuizong_file['个人编码']
#################
fenqu_list=split(zuizong_file,single_id)
print('finish split')
#################
zui_zong_test_list=[]
for i in range(len(fenqu_list)):
    if pd.DataFrame(fenqu_list[i]).shape[1]==59:
        temp_index=pd.DataFrame(fenqu_list[i]).index
        temp_list=pd.DataFrame(fenqu_list[i]).mean(axis=0)
        temp_list1=pd.DataFrame(temp_list,columns=temp_index).T
        zui_zong_test_list.append(temp_list1)
    else:
        temp_index=list(set(pd.concat(fenqu_list[i]).index))
        temp_list=pd.concat(fenqu_list[i]).mean(axis=0)
        temp_dataframe=pd.DataFrame(temp_list,columns=temp_index).T
        zui_zong_test_list.append(temp_dataframe)
    print('finish%dconcating'%i)
for i in range(1,len(fenqu_list)):
    if pd.concat([zui_zong_test_list[i-1],zui_zong_test_list[i]]).shape[1]>59:
        print(i)
final_train_file=pd.concat(zui_zong_test_list,axis=0,ignore_index=False)
final_train_file=final_train_file.sort_index()
####################
print('finsh concat')
final_train_file.fillna(0,inplace=True)
X=StandardScaler().fit_transform(final_train_file)
####载入训练之后的分类器进行分类

######是结果与目标便签匹配,file为分类器按大到小分类结果
file=pd.read_excel('/Users/zxf-pc/Desktop/天池/jieguo.xlsx')
file_index=file.index
index_list=list(file_index)
answer=[]
for i in range(len(person_id)):
    for l in range(len(index_list)):
        if index_list[l]==person_id[i]:
            answer.append(file.iloc[l,0])
    print('finish%d'%i)
            #index_list.pop(l,inplace=True)
            #print(len(index_list))
### 输出答案
answer_file=pd.DataFrame(answer,index=person_id)
answer_file.to_csv('/Users/zxf-pc/Desktop/天池/answer.csv',sep=',',header=False)###header=False表示不要列头

