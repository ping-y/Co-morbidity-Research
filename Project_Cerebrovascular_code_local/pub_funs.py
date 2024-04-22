import pandas as pd
import numpy as np
import time
import cx_Oracle

import config


def connect_db(connect_info,connect_mode,sql):
    # db = cx_Oracle.connect('sys/cdslyk912@192.168.101.34/orcl', mode=cx_Oracle.SYSDBA)
    db = cx_Oracle.connect(connect_info, mode=connect_mode)
    print(db)
    # 操作游标
    cr1 = db.cursor()
    # _0_sql = 'select SFZH,RN,JBDM,JBDM1,JBDM2,JBDM3,JBDM4,JBDM5,JBDM6,JBDM7,JBDM8,JBDM9,JBDM10,JBDM11,JBDM12,JBDM13,JBDM14,JBDM15,ALL_FLAGS from scott.YP_TTL_IHD_2YEARS'
    sql1=sql
    cr1.execute(sql1)
    cyzd_data = cr1.fetchall()
    names = [i[0] for i in cr1.description]
    df = pd.DataFrame(cyzd_data, columns=names)
    cr1.close()
    db.close()
    return df



def stime():
    return time.time()

def etime(task,t):
    print(task, (time.time()-t)/60)


def check_df(df_name,flag,df):
    if flag==0:
        print(df_name,'df',df)
    if flag==1:
        print(df_name,'df.shape',df.shape)
    if flag==2:
        print(df_name,'df[config.SFZH].drop_duplicates().shape[0]',df[config.SFZH].drop_duplicates().shape[0])
    if flag==3:
        print(df_name,'list(df.columns)',list(df.columns))


