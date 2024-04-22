import pickle
import pandas as pd
import numpy as np
import cx_Oracle
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from joblib import Parallel, delayed
from disease_burden import _1_data_analy

import config    # 自己定义的
import time
import random
from tqdm import tqdm


def n2one_match_control_extract(df_case,df_control):
    """
    用于多对一匹配提取对照队列
    # 对每个case，把所有匹配的id放入一个列表；
    # 然后，再做随机挑选
    # 检查是否有重复
    # 若有，对重复的case 重新随机匹配case，直到所有case都有自己的匹配

    # 判断是否是index字段：'is_first_source'; 'is_first_target'
    # 对case和control: groupby 患者， 每个患者对应有属性：
    # is_first_source==1的时间和住院次数； index_time; index_rn
    # target/ 最后一次住院对应RN     target_time; target_rn
    # 对case:计算index—>target的天数   （is_first_target==1的时间和住院次数；年龄性别？
    # 对control: 计算index—>最有一次住院的天数  time_gap
    # 得到两个df：df_casep & df_controlp
    # 筛选后，merge
    # df_casep; df_controlp
    # sfzh, index_time, index_rn, target_time, target_rn, time_gap
    """

    df_casep=df_case[df_case['is_source']==1][[config.SFZH,config.RY_DATE,config.NL,config.XB]]
    # df_casep
    df_controlp = df_control[[config.SFZH, config.RY_DATE,config.NL,config.XB]]


    # 整理字段
    age_group = [[18, 34], [35, 44], [45, 54], [55, 64], [65, 74], [75, 84], [85, df_celebro[config.NL].max()]]
    df_casep['age_group'] = df_controlp[config.NL].apply(
        lambda x: '18-34岁' if x >= age_group[0][0] and x <= age_group[0][1] else
        ('35-44岁' if x >= age_group[1][0] and x <= age_group[1][1] else
         ('45-54岁' if x >= age_group[2][0] and x <= age_group[2][1] else
          ('55-64岁' if x >= age_group[3][0] and x <= age_group[3][1] else
           ('65-74岁' if x >= age_group[4][0] and x <= age_group[4][1] else
            ('75-84岁' if x >= age_group[5][0] and x <= age_group[5][1] else
             ('≥85岁' if x >= age_group[6][0] and x <= age_group[6][1] else (7))))))))
    df_controlp['age_group'] = df_controlp[config.NL].apply(
        lambda x: '18-34岁' if x >= age_group[0][0] and x <= age_group[0][1] else
        ('35-44岁' if x >= age_group[1][0] and x <= age_group[1][1] else
         ('45-54岁' if x >= age_group[2][0] and x <= age_group[2][1] else
          ('55-64岁' if x >= age_group[3][0] and x <= age_group[3][1] else
           ('65-74岁' if x >= age_group[4][0] and x <= age_group[4][1] else
            ('75-84岁' if x >= age_group[5][0] and x <= age_group[5][1] else
             ('≥85岁' if x >= age_group[6][0] and x <= age_group[6][1] else (7))))))))
    del df_controlp[config.NL]
    del df_casep[config.NL]

    df_casep['ryyear'] = df_casep[config.RY_DATE].dt.year
    df_casep['rymonth'] = df_casep[config.RY_DATE].dt.month
    df_controlp['ryyear'] = df_controlp[config.RY_DATE].dt.year
    df_controlp['rymonth'] = df_controlp[config.RY_DATE].dt.month
    del df_casep[config.RY_DATE]
    del df_controlp[config.RY_DATE]
    df_casep=df_casep[[config.SFZH,config.XB,'age_group','ryyear','rymonth']]
    df_controlp=df_controlp[[config.SFZH,config.XB,'age_group','ryyear','rymonth']]
    df_controlp.columns=[config.SFZH+'_ctrl',config.XB,'age_group','ryyear','rymonth']

    print(('111, df_casep[config.id].drop_duplicates().shape)', df_casep[config.SFZH].drop_duplicates().shape))

    # merge
    # 匹配规则：同年同月住院，性别、年龄组相同
    # 同一医院——暂未匹配
    df_match=pd.merge(df_casep,df_controlp,on=[config.XB,'age_group','ryyear','rymonth'])
    print(df_match.shape)
    print(('222, df_match[config.id].drop_duplicates().shape)',df_match[config.SFZH].drop_duplicates().shape))

    tmp=df_match.groupby(config.SFZH).apply(lambda x: x[config.SFZH+'_ctrl'].values)
    tmp=pd.DataFrame(zip(tmp.index,tmp.values),columns=['SFZH','ctrl_ids'])

    tmp['ctrl_num']=tmp['ctrl_ids'].apply(lambda x:x.shape[0])
    print('每个case匹配到的control患者数量： max: %.3f ，min: %.3f , mean: %.3f '%(tmp['ctrl_num'].max(),tmp['ctrl_num'].min(),tmp['ctrl_num'].mean()))

    tmp=match_x_random(tmp)
    print('none num',tmp[tmp['ctrl_choice']=='none'].shape)
    tmp=tmp[[config.SFZH,'ctrl_choice']]

    tmp=tmp[tmp['ctrl_choice']!='none']   #最终匹配对
    print('333, tmp.shape', tmp.shape)
    print('tmp',tmp)
    return tmp


def match_x_random(df):
    # 得到1v1匹配结果
    # 先匹配candid少的对象
    # df['ctrl_num'] = df['ctrl_ids'].apply(lambda x: x.shape[0])
    df=df.sort_values(by=['ctrl_num'],ascending=True)

    lst=[]
    for i in tqdm(range(df.shape[0])):
        x=df.loc[i,'ctrl_ids']
        candid=x[random.randint(0,x.shape[0]-1)]
        if candid not in lst:
            lst.append(candid)
        else:
            flag=0
            x=np.setdiff1d(x,np.array(candid))
            x_shape=x.shape[0]
            while x_shape!=0:
                candid = x[random.randint(0, x.shape[0] - 1)]
                if candid not in lst:
                    lst.append(candid)
                    flag=1
                    break
                else:
                    x = np.setdiff1d(x, np.array(candid))
                    x_shape-=1

            if flag==0:
                lst.append('none')

    df_lst=pd.DataFrame(lst,columns=['ctrl_choice'])
    df=pd.concat([df,df_lst],axis=1)
    return df


def constr_final_ctrlp(df_caseID_ctrlID, df_ctrlp_cadid,id_col):
    # 根据身份证，给case和control拼接其他字段信息。
    # df_ctrlp_cadid: [config.SFZH, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB,"ALL_DISEASE"]
    if id_col==config.SFZH:
        flag='case'
    else:
        flag='control'
    ctrl=df_caseID_ctrlID[[id_col]]
    ctrl.columns=[config.SFZH]
    df_ctrlp_cadid = pd.merge(ctrl, df_ctrlp_cadid, on=[config.SFZH])

    print(flag, ' shape',ctrl.shape)
    print(flag,'shape',df_ctrlp_cadid.shape)

    return df_ctrlp_cadid


def get_not_chronic_disease():  #input:  (1,0) or (0,1)
    df=pd.read_excel("../data/dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
    # 大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    df_ans=df[df.chronic==0]
    dic={}
    for i in df_ans["dis"].values:
        dic[i]=1

    return dic  # 返回的是急性病字典



def handle_case(df_final_casep):
    df_casep = df_final_casep[df_final_casep['is_source'] == 1][[config.SFZH, config.XZZ_XZQH2, config.NL, config.XB]]
    df_casep.columns=[config.SFZH, "idx_"+config.XZZ_XZQH2, "idx_"+config.NL, "idx_"+config.XB]
    df_final_casep=pd.merge(df_final_casep,df_casep,on=[config.SFZH])

    region_code_path = '../data/dic_183region_code.xlsx'
    region_code = pd.read_excel(region_code_path)
    region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
    code_iscity=dict(zip(region_code['code6'],region_code['城乡划分']))
    df_final_casep['idx_is_city'] = df_final_casep['idx_' + config.XZZ_XZQH2].apply(lambda x:code_iscity[x])

    return df_final_casep



def check_ctrl_cadid(df_ba_total,df_ctrl_cadid):
    # df_ba_total只有身份证号，和ALL_disease
    # df_ctrl_cadid有这些字段： [[config.SFZH, config.RN, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]]

    # 先对df_ctrl_cadid的字段重命名，删除RN,避免出错
    df_ctrl_cadid=df_ctrl_cadid[[config.SFZH, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]]
    print('list(df_ctrl_cadid.columns) ! ! !  ',list(df_ctrl_cadid.columns))
    df_ctrl_cadid.columns= [config.SFZH, 'RN', config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]
    del df_ctrl_cadid['RN']
    print('list(df_ctrl_cadid.columns) ! ! !  ', list(df_ctrl_cadid.columns))
    print(df_ctrl_cadid.shape)

    def processParallel_1(df_group, name):
        # 处理数据,如果不加name，return的data没有group信息
        # print('df_group', df_group)
        min_value = df_group[config.RY_DATE].min()
        data = df_group[df_group[config.RY_DATE] == min_value].iloc[[0],:]
        return data

    def applyParallel_1(dfGrouped, func):
        # for name, group in dfGrouped:
            # print(name, group)
            # print()
        retLst = Parallel(n_jobs=63)(delayed(func)(group, name) for name, group in dfGrouped)
        return pd.concat(retLst)

    def processParallel_4(df_group, name):
        # 处理数据,如果不加name，return的data没有group信息
        disease_set=set()
        for i in df_group[config.ALL_DISEASE]:
            disease_set=disease_set|i
        return name,disease_set

    def applyParallel_4(dfGrouped, func):
        # for name, group in dfGrouped:
            # print(name, group)
            # print()
        retLst = Parallel(n_jobs=63)(delayed(func)(group, name) for name, group in dfGrouped)
        retLst = pd.DataFrame(retLst, columns=[config.SFZH, config.ALL_DISEASE])
        return retLst

    # 1. 先把df_ctrl_cadid里面每条身份证号的最先一条住院记录取出来
    print('1. 先把df_ctrl_cadid里面每条身份证号的最先一条住院记录取出来')
    t=time.time()
    df_ctrl_cadid = applyParallel_1(df_ctrl_cadid.groupby(config.SFZH), processParallel_1)    # ctrlp_candid中，每个患者最早的一条 ；并行计算。
    print("1. 耗时：",(time.time()-t)/60)

    # 2. merge
    print('2. merge')
    df_ctrl_cadid_multirow=pd.merge(df_ba_total,df_ctrl_cadid,on=[config.SFZH])  # 增加疾病信息；同时，一个患者会有多行记录
    del df_ba_total

    # 3. 处理all_disease
    print('3. 处理all_disease')
    df_ctrl_cadid_multirow=_1_data_analy.dealwith_ALLdisease(df_ctrl_cadid_multirow)

    # 4. grouby sfzh 以合并all_disease
    print('4. grouby sfzh 以合并all_disease')
    t=time.time()
    df_sfzh_totaldisease=applyParallel_4(df_ctrl_cadid_multirow.groupby(config.SFZH), processParallel_4)   # 返回的格式：df[[sfzh, ALL_DISEASE]]
    del df_ctrl_cadid_multirow
    print("4. 耗时：", (time.time() - t) / 60)

    # 5 去除ALL_DISEASE里面的快病
    print('5 去除ALL_DISEASE里面的快病')
    dic_notchronic=get_not_chronic_disease()
    set_notchronic=set(list(dic_notchronic.keys()))
    print('set_notchronic',set_notchronic)
    print('df_sfzh_totaldisease',df_sfzh_totaldisease)
    print(type(df_sfzh_totaldisease))
    print('df_sfzh_totaldisease.cols', list(df_sfzh_totaldisease.columns))
    print('df_sfzh_totaldisease[config.ALL_DISEASE]', df_sfzh_totaldisease[config.ALL_DISEASE])
    df_sfzh_totaldisease[config.ALL_DISEASE]=df_sfzh_totaldisease[config.ALL_DISEASE].apply(lambda x: x-set_notchronic)

    # 6. merge  df_ctrl_cadid和df_sfzh_totaldisease
    # df_ctrl_cadid有这些字段： [[config.SFZH, config.RN, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]]
    print("merge前df_ctrl_cadid的长度",df_ctrl_cadid.shape)
    df_ctrl_cadid=pd.merge(df_ctrl_cadid,df_sfzh_totaldisease ,on=[config.SFZH])
    print("merge后df_ctrl_cadid的长度",df_ctrl_cadid.shape)

    # 7. 选取ALL_DISEASE不为空的行——即病人，重新将该数据存到一个文件中，用于匹配
    df_ctrl_cadid['chronic_dis_num']=df_ctrl_cadid[config.ALL_DISEASE].apply(lambda x: len(x))
    df_ctrl_cadid=df_ctrl_cadid[df_ctrl_cadid['chronic_dis_num']>0]
    return df_ctrl_cadid



if __name__ == "__main__":
    # 不用这个match。时空开销太大

    process_set = {81: ''}
    process = [81]

    # 2015-2020年四川省脑血管住院患者疾病情况
    if 81 not in process:
        load_path = config.pdir+'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        df_celebro = pickle.load(open(load_path, 'rb'))



    if 81 in process:
        # 获取control候选数据集，并处理一下候选对照集文件：
        # 纳入疾病列，同时，删去没有慢病诊断的人，最后，保留一个身份证一行的形式，该行包含所有的住院慢病诊断，即相关基本信息。
        load_path = config.pdir + 'Project_Cerebrovascular_data/21_df_ctrl_cadid.pkl'
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))

        fields = [config.SFZH, config.ALL_DISEASE]

        db_name = 'BA_DATA.BA_SC'

        # db_name='BA_DATA.CRC_15_20'   # test!!!

        connect_info = 'ba_data/123@127.0.0.1:1521/orcl'
        connect_mode = cx_Oracle.SYSDBA

        sql = _1_data_analy.get_sql(fields, db_name)
        print("读入数据库数据-------")
        t = time.time()
        df_ba_total =_1_data_analy.connect_db(connect_info, connect_mode, sql)
        print("读入数据库数据耗时", (time.time() - t) / 60)

        print('0.df.shape', df_ba_total.shape)

        # df_ctrl_cadid=df_ctrl_cadid.head(10000)   # test!!!

        t = time.time()
        print('check ctrl cadid')
        df_ctrlp_cadid=check_ctrl_cadid(df_ba_total,df_ctrl_cadid)
        print('time consumption:',(time.time()-t)/60)

        save_path = config.pdir+"Project_Cerebrovascular_data/df_ctrlp_cadid.pkl"
        pickle.dump(df_ctrlp_cadid, open(save_path, 'wb'), protocol=4)


    if 82 in process:
        # 匹配
        # 主要目的：得到caseID-control_candidID的df
        load_path = config.pdir + 'Project_Cerebrovascular_data/df_ctrlp_cadid.pkl'
        df_ctrlp_cadid = pickle.load(open(load_path, 'rb'))
        # ['SFZH', 'RN', 'RY_DATE', ..., 'ALL_DISEASE]

        t=time.time()
        print(' match -----------')
        df_caseID_ctrlID = n2one_match_control_extract(df_celebro, df_ctrlp_cadid)

        df_final_controlp = constr_final_ctrlp(df_caseID_ctrlID, df_ctrlp_cadid,'ctrl_choice')  # 匹配后的control集  （一人一行）
        df_final_casep = constr_final_ctrlp(df_caseID_ctrlID, df_celebro,config.SFZH)  # 匹配后的case集  (一人多行)


        # case集： 添加["idx_"+config.XZZ_XZQH2, "idx_"+config.NL, "idx_"+config.XB, idx_is_city]几列
        df_final_casep=handle_case(df_final_casep)

        save_path = config.pdir + "Project_Cerebrovascular_data/82_df_final_case_netw.pkl"
        pickle.dump(df_final_casep, open(save_path, 'wb'), protocol=4)
        save_path = config.pdir + "Project_Cerebrovascular_data/82_df_final_controlp_netw.pkl"
        pickle.dump(df_final_controlp, open(save_path, 'wb'), protocol=4)

    # if 83 in process:







