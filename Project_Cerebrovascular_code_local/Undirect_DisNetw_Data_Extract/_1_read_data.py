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


def dealwith_ALLdisease(df):
    # 清洗编码数据ALL_DISEASE：去除不规范3位编码，去重，去除急性病；
    def get_chronic_disease():
        df = pd.read_excel("dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
        set_chronic = set(list(df[df['chronic'] > 0]['dis']))
        return set_chronic

    # 清洗编码数据ALL_DISEASE
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: set([i for i in x if len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[ 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[ 2] <= '9']))

    set_chronic=get_chronic_disease()
    df['ALL_DISEASE']=df['ALL_DISEASE'].apply(lambda x:x&set_chronic)
    return df


def check_gender(df):
    # 剔除性别不一致的患者
    def check_gender_apply(df_group):
        if df_group['XB'].drop_duplicates().shape[0]==1:
            return df_group['SFZH'].values[0]   # 返回的是性别始终一致的患者id

    # is_cosist_gender = df.groupby('SFZH').apply(check_gender_apply)  # 返回性别始终一致的患者

    # 用 tqdm注册 pandas.progress_apply 和 pandas.Series.map_apply 两个函数
    tqdm.pandas(desc="check_gender_apply!")
    is_cosist_gender = df.groupby('SFZH').progress_apply(check_gender_apply)  # 返回性别始终一致的患者

    del_id = is_cosist_gender.values
    del_id = np.unique(del_id[del_id != None])
    del_id = pd.DataFrame(del_id, columns=['SFZH'])
    df = pd.merge(df, del_id, on=['SFZH']).reset_index(drop=True)
    return df


if __name__ == "__main__":
    # 从数据库中读取数据
    print("从数据库中读取数据 开始时间：", time.ctime(time.time()))
    connect_info='ba_data/123456@127.0.0.1:1521/orcl'
    connect_mode=cx_Oracle.SYSDBA
    # sql= 'select SFZH, ALL_DISEASE, NL, XB, XZZ_XZQH2, CY_DATE, RN from YP_2024'

    sql="""
    select
        SFZH
        , ALL_DISEASE
        , NL
        , XB
        , XZZ_XZQH2
        , CY_DATE
        , RN
    from BA_CD_MULTI
    where FLAGS=6
    and SUBSTR(TO_CHAR(XZZ_XZQH2),1,4) in ('5120', '5132', '5133', '5134', '5101', '5103', '5104', '5105', '5106', '5107', '5108', '5109', '5110', '5111', '5113', '5114', '5115', '5116', '5117', '5118', '5119')
    and SUBSTR(TO_CHAR(DEPT_ADRRESSCODE2),1,4) in ('5120', '5132', '5133', '5134', '5101', '5103', '5104', '5105', '5106', '5107', '5108', '5109', '5110', '5111', '5113', '5114', '5115', '5116', '5117', '5118', '5119')
    and CY_DATE>=TO_DATE('2015-01-01','YYYY-MM-DD') and CY_DATE<=TO_DATE('2022-12-31','YYYY-MM-DD')
    and len(SFZH)=32
    ;
    """

    df = connect_db(connect_info, connect_mode, sql)
    print("从数据库中读取数据 完成时间：", time.ctime(time.time()))

    # 剔除性别不一致的患者
    print("剔除性别不一致的患者 开始时间：", time.ctime(time.time()))
    df=check_gender(df)
    print("剔除性别不一致的患者 完成时间：", time.ctime(time.time()))

    # 针对每条住院记录，清洗编码数据ALL_DISEASE：去除不规范3位编码，疾病去重，去除急性病；
    print("清洗编码数据ALL_DISEASE 开始时间：", time.ctime(time.time()))
    df = dealwith_ALLdisease(df)
    print("清洗编码数据ALL_DISEASE 完成时间：", time.ctime(time.time()))

    # # 去除没有慢性病的住院记录  -- 这里先不去除
    # df['chronic_num'] = df['ALL_DISEASE'].apply(lambda x: len(x))
    # df = df[df['chronic_num'] > 0].reset_index(drop=True)
    # del df['chronic_num']

    # 存储
    pickle.dump(df, open("df.pkl", 'wb'), protocol=4)




