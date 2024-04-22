import time
import numpy as np
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle
from tqdm import tqdm




def del_patients_unconcerned(df, ls_index_dis):
    # 保留有索引疾病的人，其余剔除
    set_index_dis = set(ls_index_dis)
    df['has_index_dis'] = df['ALL_DISEASE'].apply(lambda x: len(x & set_index_dis))
    save_sfzhs = df[df['has_index_dis'] > 0][['SFZH']].drop_duplicates().reset_index(drop=True)
    df = pd.merge(save_sfzhs, df, on=['SFZH']).reset_index(drop=True)
    return df


def set_period_field(df):
    # 增加字段period_field，定义record发生时间：第一次索引疾病（0）、前（-1）、后（1）
    def set_period_field_apply(df_group):
        index_rn=df_group[df_group['has_index_dis']>0]['RN'].values.min()
        df_group['period_field']=df_group['RN']-index_rn
        df_group['period_field'] =df_group['period_field'].apply(lambda x: 1 if x>0 else (-1 if x<0 else x))
        return df_group

    tqdm.pandas(desc="set_period_field!")
    df = df.groupby('SFZH').progress_apply(set_period_field_apply)

    return df


def merge_by_patient(df):

    def collect_recs(df_group):
        case_rec=df_group[df_group['period_field']==0]

        set_diseases=set()  # 全阶段疾病
        for i in df_group['ALL_DISEASE']:
            set_diseases|=i

        set_diseases_before = set()  # 确诊前的疾病
        for i in df_group[df_group['period_field']==-1]['ALL_DISEASE']:
            set_diseases_before |= i

        set_diseases_after = set()  # 确诊及以后的疾病
        for i in df_group[df_group['period_field'] >=0]['ALL_DISEASE']:
            set_diseases_after |= i

        ls_admission_num=[df_group[df_group['period_field']==i].shape[0] for i in [-1,1]]

        return case_rec['SFZH'].values[0], case_rec['NL'].values[0], case_rec['XB'].values[0], case_rec['XZZ_XZQH2'].values[0],\
            case_rec['CY_DATE'].values[0], set_diseases, set_diseases_before, set_diseases_after,ls_admission_num

    tqdm.pandas(desc="merge_by_patient!")
    df_patient = df.groupby('SFZH').progress_apply(collect_recs)
    df_patient = pd.DataFrame([list(i) for i in df_patient], ['SFZH', 'NL', 'XB', 'XZZ_XZQH2', 'CY_DATE', 'ALL_DISEASE', 'ALL_DISEASE_BEFORE', 'ALL_DISEASE_AFTER', 'ls_admission_num'])
    return df_patient


def get_index_dis(index_dis='呼吸系统慢性疾病'):
    # 定义索引疾病
    ls_index_dis=[]

    if index_dis == '呼吸系统慢性疾病':
        df_chapter=pd.read_excel('BA_SC_05_11.xlsx', sheet_name='疾病分类编码')
        ls_index_dis=list(df_chapter[df_chapter['章-编码']=='J00-J99']['类目-编码'])

        df = pd.read_excel("dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
        set_chronic = set(list(df[df['chronic'] > 0]['dis']))

        ls_index_dis=list(set(ls_index_dis)&set_chronic)

    if index_dis == '哮喘':
        ls_index_dis=['J45','J46']

    if index_dis == 'COPD':
        ls_index_dis=['J41','J42','J43','J44']

    return ls_index_dis


if __name__ == "__main__":

    df=pickle.load(open("df.pkl", 'rb'))

    # 去除没有慢性病的住院记录  -- 不用
    # df['chronic_num'] = df['ALL_DISEASE'].apply(lambda x: len(x))
    # df = df[df['chronic_num'] > 0].reset_index(drop=True)
    # del df['chronic_num']

    # 1. 定义索引疾病 - 到函数中去定义、选取索引疾病
    index_dis = ['呼吸系统慢性疾病','哮喘','COPD'][0]
    ls_index_dis = get_index_dis(index_dis)

    # 2. 保留有索引疾病的人，其余剔除。（增加字段“has_index_dis”）
    df = del_patients_unconcerned(df, ls_index_dis)

    # 3. 增加字段'period_field'：相对于首次索引疾病发生的时间，当前住院发生在【首次索引疾病（0）、前（-1）、后（1）】
    df=set_period_field(df)

    # 4. 合并多次住院：记录sfzh、年龄、性别、现住址、出院时间；合并索引疾病前、后、全阶段慢性疾病；统计前、后、全阶段住院次数；
    df_patient=merge_by_patient(df)

    # 存储 df_patient: 一人一行，包括字段：['SFZH', 'NL', 'XB', 'XZZ_XZQH2', 'CY_DATE', 'ALL_DISEASE', 'ALL_DISEASE_BEFORE', 'ALL_DISEASE_AFTER', 'ls_admission_num']
    pickle.dump(df_patient, open("df_patient_for_IndexDis_%s.pkl"%(index_dis), 'wb'), protocol=4)



