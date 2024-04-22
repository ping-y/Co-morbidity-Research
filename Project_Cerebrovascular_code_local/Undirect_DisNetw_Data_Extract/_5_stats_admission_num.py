import pickle
import pandas as pd
import numpy as np
import cx_Oracle
from pub_funs import *


if __name__ == "__main__":
    # todo 先建表 ba_sc_hos_num
    """
    # 用于统计：不同区域（DEPT_ADRRESSCODE2）不同级别（YYDJ_J，YYDJ_D）的医院，每天（RY_DATE）入院的患有不同疾病（JBDM）的不同年龄（NL）的男/女性患者人数/人次
        create table ba_sc_hos_num as
        select
            SFZH
            , DEPT_ADRRESSCODE2  --医疗机构行政区划编码
            , YYDJ_J
            , YYDJ_D
            , YLJGID --医疗机构ID
            , RY_DATE
            , JBDM
            , NL
            , XB
        from BA_SC
        where FLAGS=6;
    """

# 统计：不同区域（DEPT_ADRRESSCODE2）不同级别（YYDJ_J，YYDJ_D）的医院，每天（RY_DATE）入院的患有不同疾病（JBDM）的不同年龄（NL）的男/女性患者人数/人次
df_orig= connect_db(connect_info='ba_data/123456@127.0.0.1:1521/orcl',
                connect_mode=cx_Oracle.SYSDBA,
                sql='select * from ba_sc_hos_num;')

sfzh=pickle.load(open("df.pkl", 'rb'))[['SFZH']]
# 选人
df_orig=pd.merge(sfzh,df_orig,on=['SFZH']).reset_index(drop=True)
# 主诊断保留三位；不满足ICD10格式要求的置为'---'
df_orig['JBDM'] = df_orig['JBDM'].apply(lambda i: i[:3] if len(i) > 2 and
                                                       i[0] >= 'A' and i[0] <= 'Z' and
                                                       i[1] >= '0' and i[1] <= '9' and
                                                       i[2] >= '0' and i[2] <= '9' else '---')

# # 统计人次
rst_hos=df_orig.groupby(['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB'], as_index=False).apply(lambda x:x['SFZH'].shape[0])
rst_hos.columns=['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB','hos_num']
rst_hos=rst_hos.sort_values(by=['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB']).reset_index(drop=True)
rst_hos.to_csv('dept-yydjJ-yydjD-yljgID-ryDate-jbdm-nl-xb-人次.csv', index=False)

# # 统计人数
# rst_pati=df_orig.groupby(['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB'], as_index=False).apply(lambda x:x['SFZH'].drop_duplicates().shape[0])
# rst_pati.columns=['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB','pati_num']
# rst_pati=rst_pati.sort_values(by=['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID','RY_DATE','JBDM','NL','XB']).reset_index(drop=True)
# rst_pati.to_csv('dept-yydjJ-yydjD-yljgID-ryDate-jbdm-nl-xb-人数.csv', index=False)




hospital_info=rst_hos[['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D','YLJGID']].drop_duplicates()
print(hospital_info.shape[0])
print(hospital_info['YLJGID'].drop_duplicates().shape[0])
# todo 正常情况下，每个YLJGID只属于唯一的['DEPT_ADRRESSCODE2','YYDJ_J','YYDJ_D'],如果不满足，需要去掉或校准异常数据
assert hospital_info['YLJGID'].drop_duplicates().shape[0]==hospital_info.shape[0]
# 删掉或校准重复医疗机构的信息
abnormal_yljg=list(np.unique(hospital_info['YLJGID'][hospital_info['YLJGID'].duplicated()].values))

hospital_info.to_csv('dept-yydjJ-yydjD-yljgID.csv', index=False)  # 若无异常数据，则添加一个维表，减少存储开销
rst_hos[['YLJGID','RY_DATE','JBDM','NL','XB']].to_csv('yljgID-ryDate-jbdm-nl-xb-人次.csv', index=False)