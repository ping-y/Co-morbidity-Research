import time
import numpy as np
import os
import sys
import cx_Oracle
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix



def collect_disease(df):
    # 合并多次住院: 疾病、年龄、性别、现住址

    def collect_recs(df_group):
        df_group=df_group.sort_values(by=['RN'])
        set_diseases=set()
        for i in df_group['ALL_DISEASE']:
            set_diseases|=i
        return df_group['SFZH'].values[0],df_group['NL'].values[0],df_group['XB'].values[0],df_group['XZZ_XZQH2'].values[0],df_group['CY_DATE'].values[0],set_diseases

    tqdm.pandas(desc="合并多次住院!")
    df_patient = df.groupby('SFZH').progress_apply(collect_recs)
    df_patient = pd.DataFrame([list(i) for i in df_patient], ['SFZH', 'NL', 'XB', 'XZZ_XZQH2','CY_DATE', 'ALL_DISEASE'])
    return df_patient


if __name__ == "__main__":

    df=pickle.load(open("df.pkl", 'rb'))

    # 去除没有慢性病的住院记录
    df['chronic_num'] = df['ALL_DISEASE'].apply(lambda x: len(x))
    df = df[df['chronic_num'] > 0].reset_index(drop=True)
    del df['chronic_num']

    # 合并多次住院: sfzh、年龄、性别、现住址、疾病
    print("合并多次住院 开始时间：", time.ctime(time.time()))
    df_patient = collect_disease(df)
    print("合并多次住院 完成时间：", time.ctime(time.time()))

    # 去掉慢病数少于2的患者
    df_patient['chronic_num'] = df_patient['ALL_DISEASE'].apply(lambda x: len(x))
    df_patient = df_patient[df_patient['chronic_num'] > 1].reset_index(drop=True)
    del df_patient['chronic_num']

    # # 存储 df_patient: 一人一行，包括字段：['SFZH', 'NL', 'XB', 'XZZ_XZQH2','CY_DATE', 'ALL_DISEASE']
    # pickle.dump(df_patient, open("df_patient_for_Multimorbidity.pkl", 'wb'), protocol=4)


    # 按列拆分存储
    pickle.dump(df_patient[['SFZH', 'NL', 'XB', 'XZZ_XZQH2','CY_DATE']], open("df_patient_basicInfo_for_Multimorbidity.pkl", 'wb'), protocol=4)
    pickle.dump(df_patient[['ALL_DISEASE']], open("df_patient_AllDisease_for_Multimorbidity.pkl", 'wb'), protocol=4)

    set_all_dis=set()
    for set_i in df_patient['ALL_DISEASE']:
        set_all_dis|=set_i
    dic_dis_colNum=dict(zip(sorted(list(set_all_dis)), list(range(len(set_all_dis)))))

    rows=[]
    cols=[]
    for row_i, set_i in enumerate(df_patient['ALL_DISEASE']):
        col = [dic_dis_colNum[_] for _ in set_i]  # 列索引
        row = [row_i]*len(col)  # 行索引
        cols.extend(col)
        rows.extend(row)
    pickle.dump((rows, cols), open("Patient_Dis_for_Multimorbidity.pkl", 'wb'), protocol=4)
    # Patient_Dis = csr_matrix((data, (rows, cols)), shape=(df_patient.shape[0], len(set_all_dis)))
