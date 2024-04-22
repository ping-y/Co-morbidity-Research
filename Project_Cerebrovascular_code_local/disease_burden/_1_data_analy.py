import time
import numpy as np
import os
import sys
import cx_Oracle

import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle


def get_sql(fields,db_name):
    fields_str=''
    for i in fields:
        fields_str+=i
        if i!=fields[-1]:
            fields_str+=','
    sql='select %s from %s where flags=4 or flags=6'%(fields_str,db_name)
    return sql


def dealwith_ALLdisease(df):
    # 清洗编码数据ALL_DISEASE\
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: set([i for i in x if
                                                               len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[ 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[ 2] <= '9']))
    return df


def get_df_pa(df_group):
    all_disease=set()
    for i in  df_group[config.ALL_DISEASE]:
        all_disease=all_disease|i

    set_cerebro=set(config.celebro)
    if len(all_disease&set_cerebro)>0:
        return df_group[config.SFZH].values[0]   # 返回的是患有cerebro的患者


def check_sfzh_length(df):
    df['sfzh_length']=df[config.SFZH].apply(lambda x:len(x) if x!=None else 0)
    df=df[df['sfzh_length']==32].reset_index(drop=True)
    print(df)
    del df['sfzh_length']
    return df



def check_gender(df_group):
    if df_group[config.XB].drop_duplicates().shape[0]==1:
        return df_group[config.SFZH].values[0]   # 返回的是性别始终一致的患者id


def get_raw_data_file(connect_info,connect_mode,fields,db_name,save_path,is_pickle,code6_nd):
    """
    0- 从数据库中读取数据，清洗疾病编码， 确认研究窗口内纳入的患者是脑血管患者。数据存储pickle
    :param connect_info:
    :param connect_mode:
    :param fields: a list
    :param db_name: 'scott.YP_TTL_IHD_2YEARS'
    :param save_path:
    :param is_pickle: boolean
    :return:
    """
    # 读取脑血管患者的数据： 包括字段：
    # fields=[SFZH,RN,MZ,XB,YYDJ_J,YYDJ_D,YLFKFS,NL,CS_DATE,ZY,HY,RYTJ,RY_DATE,CY_DATE,RYQK,
    #         LYFS, ZFY, ZFJE, ZB1, ZB2, ZB3, ZB4, ZB5, ZB6, ZB7, ZB8, ZB9, QTF,
    #         DEPT_ADRRESSCODE2, XZZ_XZQH2, FLAGS,ALL_DISEASE]

    sql=get_sql(fields,db_name)
    # 读入数据库数据
    print("读入数据库数据-------")
    t=time.time()
    df=connect_db(connect_info,connect_mode,sql)

    print("读入数据库数据耗时",(time.time()-t)/60)

    # df=df.head(20000)
    # 筛一遍身份证号，只要32位长的
    df=check_sfzh_length(df)
    # 清洗编码数据ALL_DISEASE\
    print("清洗编码数据ALL_DISEASE--------------")
    df=dealwith_ALLdisease(df)
    # 时间筛一遍 2015-2020  # 按照出院时间： 2015-2020
    print('1.df.shape',df.shape)
    df=df[(df[config.CY_DATE]>=pd.to_datetime('2015-01-01'))&(df[config.CY_DATE]<=pd.to_datetime('2020-12-31'))]

    # 过一遍年龄：删除首次住院年龄小于18岁的患者的所有住院记录
    del_id=df[df[config.NL]<18][config.SFZH].drop_duplicates().values
    # del_id = pd.DataFrame(del_id, columns=[config.SFZH])   # 要删除的sfzh
    save_id=pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values)-set(del_id)), columns=[config.SFZH]) # 要保留的id
    print("18岁以下的脑血管疾病患者人数：",del_id.shape[0])
    old_dflen=df.shape[0]
    df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    print("18岁以下的脑血管疾病住院记录数：", old_dflen-df.shape[0])

    # 过一遍现住址 ： 不在四川省内的人的所有记录都删掉
    df['is_sichuanren']=df[config.XZZ_XZQH2].apply(lambda x: 1 if x in code6_nd else 0)
    del_id=df[df['is_sichuanren']==0][config.SFZH].drop_duplicates().values
    save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)), columns=[config.SFZH])  # 要保留的id
    print("现住址不在四川的脑血管疾病患者人数：", del_id.shape[0])
    old_dflen = df.shape[0]
    df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    print("现住址不在四川的脑血管疾病住院记录数：", old_dflen - df.shape[0])
    del df['is_sichuanren']

    # 过一遍医院地址编码：不在四川省内的人的所有记录都删除
    df['is_sichuanren'] = df[config.DEPT_ADDRESSCODE2].apply(lambda x: 1 if x in code6_nd else 0)
    del_id = df[df['is_sichuanren'] == 0][config.SFZH].drop_duplicates().values
    save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)), columns=[config.SFZH])  # 要保留的id
    print("医院住址不在四川的脑血管疾病患者人数：", del_id.shape[0])
    old_dflen = df.shape[0]
    df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    print("医院住址不在四川的脑血管疾病住院记录数：", old_dflen - df.shape[0])
    del df['is_sichuanren']


    # 过一遍性别 剔除性别不一致的患者
    is_cosist_gender = df.groupby(config.SFZH).apply(check_gender)  # 返回性别始终一致的患者
    del_id=is_cosist_gender.values
    del_id = np.unique(del_id[del_id != None])
    del_id = pd.DataFrame(del_id, columns=[config.SFZH])
    df = pd.merge(df, del_id, on=[config.SFZH]).reset_index(drop=True)


    print('2.df.shape', df.shape)
    # 再过一遍疾病, 删除研究窗口内不是cerebro的患者id
    t=time.time()
    print('groupby------------，再过一遍疾病, 删除研究窗口内不是cerebro的患者id')
    is_cerebro= df.groupby(config.SFZH).apply(get_df_pa)   # 返回的是患有cerebro的患者
    print("goruby耗时：",(time.time()-t)/60)
    del_id=is_cerebro.values
    del_id = np.unique(del_id[del_id != None])
    del_id = pd.DataFrame(del_id, columns=[config.SFZH])
    print('纳入的患者数.shape:',del_id.shape[0])
    df = pd.merge(df, del_id, on=[config.SFZH]).reset_index(drop=True)
    print('3.df[config.SFZH].drop_duplicates().shape[0]', df[config.SFZH].drop_duplicates().shape[0])
    print('3.df.shape', df.shape)

    if is_pickle==True:
        pickle.dump(df,open(save_path,'wb'), protocol=4)
    return df

def del_firestCERE_before(df):
    # 删除cerebro患者首次cerebro诊断年之前的所有住院记录
    # groupby 找首次cerebro; 首次打标签1；以后打标签2；之前打标签0  <—— 'is_source'
    # 删除标签
    set_cere=set(config.celebro)
    df = df.groupby(config.SFZH).apply(find_first_cerebro,set_cere)
    old_dfshape=df.shape[0]
    print('1. df.shape',df.shape)
    df = df[df['is_source'] > 0]
    print("new df shape:",df.shape[0]-old_dfshape)
    # del df['is_source']
    print('2. df.shape',df.shape)
    return df


def find_first_cerebro(df_group,set_cere):   # 跑了4个小时。。。。。。
    df_group["is_source"] = df_group[config.ALL_DISEASE].apply(lambda x: 0 if len(set_cere & x) == 0 else 1)
    first_source_rn = df_group[df_group["is_source"] == 1][config.RN].min()
    df_group['is_source'] = df_group[config.RN].apply(lambda x: 1 if x == first_source_rn else (2 if x>first_source_rn else 0))
    return df_group


def del_firestCERE_before2(df):
    # 删除cerebro患者首次cerebro诊断年之前的所有住院记录
    # groupby 找首次cerebro; 首次打标签1；以后打标签2；之前打标签0
    # 删除标签
    set_cere=set(config.celebro)
    df = df.groupby(config.SFZH).apply(find_first_cerebro,set_cere)
    old_dfshape=df.shape[0]
    print('1. df.shape',df.shape)
    df = df[df['is_source'] > 0]
    print("new df shape:",df.shape[0]-old_dfshape)
    # del df['is_source']
    print('2. df.shape',df.shape)
    print('list(df.columns)',list(df.columns))
    return df


def find_first_cerebro2(df_group,set_cere):
    df_group["is_source"] = df_group[config.ALL_DISEASE].apply(lambda x: 0 if len(set_cere & x) == 0 else 1)
    first_source_rn = df_group[df_group["is_source"] == 1][config.RN].min()
    df_group['is_source'] = df_group[config.RN].apply(lambda x: 1 if x == first_source_rn else (2 if x>first_source_rn else 0))
    return df_group


def get_df_pa_file(df,is_pickle,save_path,save_path_col):
    id_dfgroup = df.groupby(config.SFZH).apply(get_pa_dfgroup)
    # print("id_dfgroup",id_dfgroup)
    print("id_dfgroup",id_dfgroup.shape)
    id=np.array(id_dfgroup.index)
    print('id',id)
    dfgroup=id_dfgroup.values
    # print('dfgroup',dfgroup)
    np_pa = np.concatenate((id.reshape(-1, 1), dfgroup.reshape(-1, 1)), axis=1)
    print('np_pa.shape',np_pa.shape)
    df_pa=pd.DataFrame(np_pa, columns=['SFZH', 'dfgroup'])

    dfgroup_col=[i for i in df.columns if i != config.SFZH]

    if is_pickle==True:
        pickle.dump(df_pa, open(save_path, 'wb'), protocol=4)
        pickle.dump(dfgroup_col, open(save_path_col, 'wb'), protocol=4)
    return df_pa,dfgroup_col


def get_pa_dfgroup(df_group):
    df_group=df_group[[i for i in df_group.columns if i!=config.SFZH]]
    return df_group.values


if __name__ == "__main__":
    """
    """
    process_set={0:'get_raw_data_file', 1:'del_firestCERE_before', 2:'get_df_pa_file'}
    process=[1]

    pdir='E:/UESTC_yang/'

    # 0-读数据
    if 0 in process:
        connect_info='ba_data/123@127.0.0.1:1521/orcl'
        connect_mode=cx_Oracle.SYSDBA
        save_path='E:/UESTC_yang/Project_Cerebrovascular_data/cerebro_data.pkl'
        fields=['SFZH','MZ','XB','YYDJ_J','YYDJ_D','YLFKFS','NL','CS_DATE','ZY','HY','RYTJ', 'RY_DATE','CY_DATE','RYQK','LYFS',
        'ZFY', 'ZFJE', 'ZB1', 'ZB2', 'ZB3', 'ZB4', 'ZB5', 'ZB6', 'ZB7', 'ZB8', 'ZB9', 'QTF',
                'DEPT_ADRRESSCODE2', 'XZZ_XZQH2', 'FLAGS','ALL_DISEASE']
        db_name='BA_DATA.CBVD_15_20'
        is_pickle=True

        # 获取四川省区县编码
        region_code_path = '../data/dic_183region_code.xlsx'
        region_code = pd.read_excel(region_code_path)
        region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
        code6_nd=region_code['code6'].values
        print('code6_lst',code6_nd.shape[0],code6_nd)

        t=time.time()
        df=get_raw_data_file(connect_info, connect_mode, fields, db_name, save_path, is_pickle,code6_nd)
        print('0耗时：',(time.time()-t)/60)
        # if is_pickle==True:
        #     del df

    if 1 in process:
        # 1- del_firestCERE_before
        # 删除首发脑血管疾病之前的住院记录
        split_flag = True
        subproc=[120, 121]
        if 120 in subproc:
            read_path = pdir+'Project_Cerebrovascular_data/cerebro_data.pkl'
            df = pickle.load(open(read_path, 'rb'))

            save_path=pdir+'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
            t=time.time()
            if split_flag==True:
                df_flag=df[[config.SFZH,config.RN,config.ALL_DISEASE]]  # df瘦身。。。
            print("正在进行：2——",process_set[2])
            df_flag=del_firestCERE_before(df_flag)
            # pickle.dump(df_flag, open(save_path, 'wb'), protocol=4)
            # print('2——耗时：', (time.time() - t) / 60)

        if split_flag==True and 121 in subproc:
            # 分步骤处理2这一步: 这里是第二步
            # read_path = 'F:/Project_Cerebrovascular_data/cerebro_data.pkl'
            # df = pickle.load(open(read_path, 'rb'))

            # save_path = 'F:/Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
            # df_flag = pickle.load(open(save_path, 'rb'))

            df_flag = df_flag[[i for i in df_flag.columns if i!=config.ALL_DISEASE]]
            print('df_flag.shape', df_flag.shape)

            print('1. df.shape[0]',df.shape)
            print(list(df.columns))
            df=pd.merge(df, df_flag, on=[config.SFZH,config.RN])
            print('2. df.shape[0]', df.shape)
            print(list(df.columns))
            pickle.dump(df, open(save_path, 'wb'), protocol=4)  # 'is_source'  ——  首次cerebro; 首次打标签1；以后打标签2


    # 2-得到df_pa;即按人存储数据 直接存df_group.values,去除身份证那一列
    # 注意维护列名！！！
    # 注意时间开销和空间开销
    if 2 in process:
        t=time.time()
        print("正在进行：2——",process_set[1])
        is_load = True
        read_path='F:/Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        is_pickle=True
        save_path = 'F:/Project_Cerebrovascular_data/cerebro_id_dfgroup.pkl'
        save_path_col='F:/Project_Cerebrovascular_data/cerebro_id_dfgroup_col.pkl'

        if is_load==True:
            df=pickle.load(open(read_path,'rb'))
        df_pa,dfgroup_col=get_df_pa_file(df,is_pickle,save_path,save_path_col)
        # df_pa, dfgroup_col = get_df_pa_file(df, is_pickle, save_path)
        print('2——耗时：',(time.time()-t)/60)







