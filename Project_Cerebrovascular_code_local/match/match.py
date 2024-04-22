import pickle
import pandas as pd
import numpy as np
import cx_Oracle
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from joblib import Parallel, delayed

import config    # 自己定义的
import time
import random
from tqdm import tqdm

def get_sql(fields,db_name):
    fields_str=''
    for i in fields:
        fields_str+=i
        if i!=fields[-1]:
            fields_str+=','
    sql='select %s from %s where flags=4 or flags=6'%(fields_str,db_name)
    print(sql)
    return sql


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


def check_ctrl_cadid_step1 (df_ctrl_cadid):
    # df_ba_total只有身份证号，和ALL_disease
    # df_ctrl_cadid有这些字段： [[config.SFZH, config.RN, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]]

    # 先对df_ctrl_cadid的字段重命名，删除RN,避免出错
    df_ctrl_cadid=df_ctrl_cadid[[config.SFZH, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]]
    print('list(df_ctrl_cadid.columns) ! ! !  ',list(df_ctrl_cadid.columns))
    df_ctrl_cadid.columns= [config.SFZH, 'RN', config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB]
    del df_ctrl_cadid['RN']

    # 1. 先把df_ctrl_cadid里面每条身份证号的最先一条住院记录取出来
    print('step1. 先把df_ctrl_cadid里面每条身份证号的最先一条住院记录取出来')
    # df_ctrl_cadid = applyParallel_1(df_ctrl_cadid.groupby(config.SFZH), processParallel_1)    # ctrlp_candid中，每个患者最早的一条 ；并行计算。
    t=time.time()
    df_ctrl_cadid['sort_id'] = df_ctrl_cadid[config.RY_DATE].groupby(df_ctrl_cadid[config.SFZH]).rank()
    df_ctrl_cadid=df_ctrl_cadid[df_ctrl_cadid['sort_id']==1]
    print('1. 先把df_ctrl_cadid里面每条身份证号的最先一条住院记录取出来',(time.time()-t)/60)
    return df_ctrl_cadid


def step2(df_ba_total,df_ctrl_cadid):
    # 2. merge
    print('2. merge')
    t = time.time()
    df_ctrl_cadid_multirow = pd.merge(df_ba_total, df_ctrl_cadid, on=[config.SFZH])  # 增加疾病信息；同时，一个患者会有多行记录
    del df_ba_total
    print('2.merge', (time.time() - t) / 60)
    return df_ctrl_cadid_multirow



def read_ba_sc():
    fields = [config.SFZH, config.ALL_DISEASE]

    db_name = 'BA_DATA.BA_SC'
    connect_info = 'ba_data/123@127.0.0.1:1521/orcl'
    connect_mode = cx_Oracle.SYSDBA

    sql = get_sql(fields, db_name)
    print("读入数据库数据-------")
    t = time.time()
    df_ba_total = connect_db(connect_info, connect_mode, sql)
    print("读入数据库数据耗时", (time.time() - t) / 60)

    print('0.df_ba_total.shape', df_ba_total.shape)
    return df_ba_total


def read_ba_sc_yljgID():
    fields = [config.SFZH ,'CY_DATE','RY_DATE', 'NL', 'YLJGID']

    db_name = 'BA_DATA.BA_SC'
    connect_info = 'ba_data/123@127.0.0.1:1521/orcl'
    connect_mode = cx_Oracle.SYSDBA

    sql = get_sql(fields, db_name)
    print("读入数据库数据-------")
    t = time.time()
    df_ba_total = connect_db(connect_info, connect_mode, sql)
    print("读入数据库数据耗时", (time.time() - t) / 60)

    print('0.df_ba_total.shape', df_ba_total.shape)
    return df_ba_total


def dealwith_ALLdisease(df):
    # 清洗编码数据ALL_DISEASE\
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: ([i for i in x.split(',')]))
    df['ALL_DISEASE'] = df['ALL_DISEASE'].apply(lambda x: set([i for i in x if
                                                               len(i) > 2 and i[0] >= 'A' and i[0] <= 'Z' and i[ 1] >= '0' and i[1] <= '9' and i[2] >= '0' and i[ 2] <= '9']))
    return df


def step3(df_ctrl_cadid_multirow):
    # 3. 处理all_disease
    print('3. 处理all_disease')
    t=time.time()
    df_ctrl_cadid_multirow = dealwith_ALLdisease(df_ctrl_cadid_multirow)
    print('3. 处理all_disease', (time.time() - t) / 60)
    return df_ctrl_cadid_multirow


def get_not_chronic_disease():  #input:  (1,0) or (0,1)
    df=pd.read_excel("dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
    # 大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    df_ans=df[df.chronic==0]
    dic={}
    for i in df_ans["dis"].values:
        dic[i]=1

    return dic  # 返回的是急性病字典


def step4(df_sfzh_totaldisease):
    # 4 去除ALL_DISEASE里面的快病
    print('4 去除ALL_DISEASE里面的快病')
    t=time.time()
    dic_notchronic = get_not_chronic_disease()
    set_notchronic = set(list(dic_notchronic.keys()))
    # print('set_notchronic', set_notchronic)
    # print('df_sfzh_totaldisease', df_sfzh_totaldisease)
    # print(type(df_sfzh_totaldisease))
    # print('df_sfzh_totaldisease.cols', list(df_sfzh_totaldisease.columns))
    # print('df_sfzh_totaldisease[config.ALL_DISEASE]', df_sfzh_totaldisease[config.ALL_DISEASE])
    df_sfzh_totaldisease[config.ALL_DISEASE] = df_sfzh_totaldisease[config.ALL_DISEASE].apply(
        lambda x: x - set_notchronic)
    print('4 去除ALL_DISEASE里面的快病', (time.time() - t) / 60)
    return df_sfzh_totaldisease


def step5(df_ctrl_cadid):
    # 7. 选取ALL_DISEASE不为空的行——即病人，重新将该数据存到一个文件中，用于匹配
    print('step 5. 选取ALL_DISEASE不为空的行')
    t=time.time()
    df_ctrl_cadid['chronic_dis_num'] = df_ctrl_cadid[config.ALL_DISEASE].apply(lambda x: len(x))
    df_ctrl_cadid = df_ctrl_cadid[df_ctrl_cadid['chronic_dis_num'] > 0]
    del df_ctrl_cadid['chronic_dis_num']
    print('step 5. 选取ALL_DISEASE不为空的行', (time.time() - t) / 60)
    return df_ctrl_cadid


def step_():
    # 4. grouby sfzh 以合并all_disease
    def processParallel_4(df_group, name):
        # 处理数据,如果不加name，return的data没有group信息
        disease_set=set()
        for i in df_group[config.ALL_DISEASE]:
            disease_set=disease_set|i
        return name,disease_set

    def applyParallel_4(dfGrouped, func):
        retLst = Parallel(n_jobs=63)(delayed(func)(group, name) for name, group in dfGrouped)
        retLst = pd.DataFrame(retLst, columns=[config.SFZH, config.ALL_DISEASE])
        return retLst

    print('4. grouby sfzh 以合并all_disease')
    t = time.time()
    df_sfzh_totaldisease = applyParallel_4(df_ctrl_cadid_multirow.groupby(config.SFZH),
                                           processParallel_4)  # 返回的格式：df[[sfzh, ALL_DISEASE]]
    del df_ctrl_cadid_multirow
    print("4. 耗时：", (time.time() - t) / 60)
    return


def step6(df_ctrl_cadid,df_ctrl_cadid_multirow):
    del df_ctrl_cadid['sort_id']

    # 要保留的id
    sfzh_nd=df_ctrl_cadid_multirow[config.SFZH].drop_duplicates().values
    sfzh_df=pd.DataFrame(sfzh_nd, columns=[config.SFZH])

    print('1.df_ctrl_cadid.shape', df_ctrl_cadid.shape)
    df_ctrl_cadid=pd.merge(df_ctrl_cadid,sfzh_df,on=[config.SFZH])
    print('2.df_ctrl_cadid.shape', df_ctrl_cadid.shape)
    return df_ctrl_cadid


def match_x_random(df):
    # 得到1v1匹配结果
    # 先匹配candid少的对象
    # df['ctrl_num'] = df['ctrl_ids'].apply(lambda x: x.shape[0])
    df=df.sort_values(by=['ctrl_num'],ascending=True).reset_index(drop=True)
    # print('df',df)
    # print(list(df.columns))
    lst=[]
    # for i in tqdm(range(df.shape[0])):
    for i in range(df.shape[0]):
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

    # df_casep=df_case[df_case['is_source']==1][[config.SFZH,config.RY_DATE,config.NL,config.XB]]
    df_casep = df_case[df_case['is_source'] == 1][[config.SFZH, config.RY_DATE, config.NL, config.XB,'YLJGID']]  # YLJGID
    # df_casep
    # df_controlp = df_control[[config.SFZH, config.RY_DATE,config.NL,config.XB]]
    df_controlp = df_control[[config.SFZH, config.RY_DATE, config.NL, config.XB,'YLJGID']]  # YLJGID

    # 整理字段
    age_group = [[18, 34], [35, 44], [45, 54], [55, 64], [65, 74], [75, 84], [85, 1000]]
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

    del df_controlp[config.NL]    # age
    del df_casep[config.NL]      # age
    # del df_controlp['age_group']
    # del df_casep['age_group']

    df_casep['ryyear'] = df_casep[config.RY_DATE].dt.year
    df_casep['rymonth'] = df_casep[config.RY_DATE].dt.month
    # df_casep['ryday'] = df_casep[config.RY_DATE].dt.day  # day

    df_controlp['ryyear'] = df_controlp[config.RY_DATE].dt.year
    df_controlp['rymonth'] = df_controlp[config.RY_DATE].dt.month
    # df_controlp['ryday'] = df_controlp[config.RY_DATE].dt.day  # day

    del df_casep[config.RY_DATE]
    del df_controlp[config.RY_DATE]
    # df_casep=df_casep[[config.SFZH,config.XB,'age_group','ryyear','rymonth']]
    # df_controlp=df_controlp[[config.SFZH,config.XB,'age_group','ryyear','rymonth']]
    # df_controlp.columns=[config.SFZH+'_ctrl',config.XB,'age_group','ryyear','rymonth']

    df_casep = df_casep[[config.SFZH, config.XB, 'age_group', 'ryyear', 'rymonth', 'YLJGID']]            # YLJGID
    df_controlp = df_controlp[[config.SFZH, config.XB, 'age_group', 'ryyear', 'rymonth','YLJGID']]       # YLJGID
    df_controlp.columns = [config.SFZH + '_ctrl', config.XB, 'age_group', 'ryyear', 'rymonth','YLJGID']  # YLJGID

    # df_casep = df_casep[[config.SFZH, config.XB, 'age_group', 'ryyear', 'rymonth','ryday']]              # day
    # df_controlp = df_controlp[[config.SFZH, config.XB, 'age_group', 'ryyear', 'rymonth','ryday']]        # day
    # df_controlp.columns = [config.SFZH + '_ctrl', config.XB, 'age_group', 'ryyear', 'rymonth','ryday']   # day

    # df_casep = df_casep[[config.SFZH, config.XB, config.NL, 'ryyear', 'rymonth']]              # age
    # df_controlp = df_controlp[[config.SFZH, config.XB, config.NL, 'ryyear', 'rymonth']]        # age
    # df_controlp.columns = [config.SFZH + '_ctrl', config.XB, config.NL, 'ryyear', 'rymonth']   # age


    print(('111, df_casep[config.id].drop_duplicates().shape)', df_casep[config.SFZH].drop_duplicates().shape))

    # merge
    # 匹配规则：同年同月住院，性别、年龄组相同
    # 同一医院——暂未匹配
    df_casep=df_casep.reset_index(drop=True)
    len_dfcasep=df_casep.shape[0]

    lst_matchdfs=[]

    for i in range(500):   # 切分case df，分治；不然merge后得到的df_match所需存储空间会溢出内存
        print('-------i=%d----------'%(i))
        df_casep_portion=df_casep.iloc[int(i/500*len_dfcasep): int((i+1)/500*len_dfcasep),:]

        # df_match=pd.merge(df_casep_portion,df_controlp,on=[config.XB,'age_group','ryyear','rymonth'])
        df_match = pd.merge(df_casep_portion, df_controlp, on=[config.XB, 'age_group', 'ryyear', 'rymonth', 'YLJGID'])  # YLJGID
        # df_match = pd.merge(df_casep_portion, df_controlp, on=[config.XB, 'age_group', 'ryyear', 'rymonth'])  # day
        # df_match = pd.merge(df_casep_portion, df_controlp, on=[config.XB, config.NL, 'ryyear', 'rymonth'])  # age

        print(df_match.shape)

        del df_casep_portion

        print(('222, df_match[config.id].drop_duplicates().shape)',df_match[config.SFZH].drop_duplicates().shape))

        tmp=df_match.groupby(config.SFZH).apply(lambda x: x[config.SFZH+'_ctrl'].values)

        del df_match

        tmp=pd.DataFrame(zip(tmp.index,tmp.values),columns=['SFZH','ctrl_ids'])

        tmp['ctrl_num']=tmp['ctrl_ids'].apply(lambda x:x.shape[0])
        print('每个case匹配到的control患者数量： max: %.3f ，min: %.3f , mean: %.3f '%(tmp['ctrl_num'].max(),tmp['ctrl_num'].min(),tmp['ctrl_num'].mean()))

        lst_matchdfs.append(tmp)
        del tmp
    matchdfs=pd.concat(lst_matchdfs,axis=0)
    return matchdfs


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


def handle_case(df_final_casep):
    df_casep = df_final_casep[df_final_casep['is_source'] == 1][[config.SFZH, config.XZZ_XZQH2, config.NL, config.XB]]
    df_casep.columns=[config.SFZH, "idx_"+config.XZZ_XZQH2, "idx_"+config.NL, "idx_"+config.XB]
    df_final_casep=pd.merge(df_final_casep,df_casep,on=[config.SFZH])

    region_code_path = 'dic_183region_code.xlsx'
    region_code = pd.read_excel(region_code_path)
    region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
    code_iscity=dict(zip(region_code['code6'],region_code['城乡划分']))
    df_final_casep['idx_is_city'] = df_final_casep['idx_' + config.XZZ_XZQH2].apply(lambda x:code_iscity[x])

    return df_final_casep

def one_match_one(tmp):
    tmp=match_x_random(tmp)
    print('none num',tmp[tmp['ctrl_choice']=='none'].shape)
    tmp=tmp[[config.SFZH,'ctrl_choice']]

    tmp=tmp[tmp['ctrl_choice']!='none']   #最终匹配对
    print('333, tmp.shape', tmp.shape)
    # print('tmp',tmp)
    return tmp


def add_comorbidity_info(df_ctrl_cadid_multirow, df_celebro, df_final_controlp, df_final_casep):
    # 含全部共病信息的两个文档df_ctrl_cadid_multirow、df_celebro
    # 需要添加共病信息的两个df：df_final_controlp、df_final_casep  （由步骤8得到）

    df_ctrl_cadid_multirow=pd.merge(df_ctrl_cadid_multirow,df_final_controlp,on=[config.SFZH])
    df_final_casep=pd.merge(df_celebro,df_final_casep,on=[config.SFZH])

    print('df_ctrl_cadid_multirow.shape',df_ctrl_cadid_multirow.shape)
    print('df_final_casep.shape', df_final_casep.shape)

    return df_ctrl_cadid_multirow,df_final_casep


if __name__ == "__main__":
    # step 1-4： 处理数据，得到ctrl候选集
    # step 5- ： match

    process=[1]

    if 1 in process:
        load_path = config.pdir + 'Project_Cerebrovascular_data/21_df_ctrl_cadid.pkl'
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))

        df_ctrl_cadid = check_ctrl_cadid_step1(df_ctrl_cadid)

        save_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        pickle.dump(df_ctrl_cadid, open(save_path, 'wb'), protocol=4)


    if 2 in process:
        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))
        print('df_ctrl_cadid[config.SFZH].drop_duplicates().shape',df_ctrl_cadid[config.SFZH].drop_duplicates().shape)
        print('df_ctrl_cadid.shape',df_ctrl_cadid.shape)
        print('df_ctrl_cadid.columns',df_ctrl_cadid.columns)

        df_ba_total=read_ba_sc()

        df_ctrl_cadid_multirow=step2(df_ba_total, df_ctrl_cadid)
        del df_ba_total

    # if 3 in process:
        df_ctrl_cadid_multirow=step3(df_ctrl_cadid_multirow)   # 处理all_disease
        df_ctrl_cadid_multirow=step4(df_ctrl_cadid_multirow)  # 去除ALL_DISEASE里面的快病
        df_ctrl_cadid_multirow=step5(df_ctrl_cadid_multirow)   # 选取ALL_DISEASE不为空的行
        df_ctrl_cadid_multirow=df_ctrl_cadid_multirow[[config.SFZH,config.ALL_DISEASE]]
        save_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid_multirow.pkl"
        pickle.dump(df_ctrl_cadid_multirow, open(save_path, 'wb'), protocol=4)

    if 4 in process:
        # 将8_df_ctrl_cadid.pkl中，一个慢病都没有的行(人)去掉; 同时，去掉sort_id这一列; 去掉
        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))
        print('df_ctrl_cadid.shape',df_ctrl_cadid.shape)
        print('list(df_ctrl_cadid.columns)',list(df_ctrl_cadid.columns))

        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid_multirow.pkl"
        df_ctrl_cadid_multirow = pickle.load(open(load_path, 'rb'))
        print('df_ctrl_cadid_multirow.shape', df_ctrl_cadid_multirow.shape)
        print('list(df_ctrl_cadid_multirow.columns)', list(df_ctrl_cadid_multirow.columns))

        df_ctrl_cadid=step6(df_ctrl_cadid,df_ctrl_cadid_multirow)   # 将8_df_ctrl_cadid.pkl中，一个慢病都没有的行去掉，同时，去掉sort_id这一列

        save_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        pickle.dump(df_ctrl_cadid, open(save_path, 'wb'), protocol=4)

    if 5 in process:
        load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        df_celebro = pickle.load(open(load_path, 'rb'))
        print('1. df_celebro.shape', df_celebro.shape)
        print('1. list(df_celebro.columns)', list(df_celebro.columns))
        print('1. df_celebro[config.SFZH].drop_duplicates().shape', df_celebro[config.SFZH].drop_duplicates().shape)

        df_source = df_celebro.drop_duplicates(subset=['RY_DATE', 'NL', 'SFZH', 'CY_DATE'])
        print('2. df_source.shape', df_source.shape)
        print('2. list(df_source.columns)', list(df_source.columns))
        print('2. df_source[config.SFZH].drop_duplicates().shape', df_source[config.SFZH].drop_duplicates().shape)

        df_ba_total = read_ba_sc_yljgID()

        df_source = pd.merge(df_source, df_ba_total, on=['RY_DATE', 'NL', 'SFZH', 'CY_DATE'])
        print('3. df_source[config.SFZH].drop_duplicates().shape', df_source[config.SFZH].drop_duplicates().shape)
        print('3. df_celebro.shape', df_source.shape)
        print('3. list(df_celebro.columns)', list(df_source.columns))

        pickle.dump(df_source, open(config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov_v2.pkl', 'wb'), protocol=4)

    if 6 in process:
        # 添加医疗机构ID字段
        if 5 not in process:
            df_ba_total = read_ba_sc_yljgID()

        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))
        print('1. df_ctrl_cadid.shape', df_ctrl_cadid.shape)
        print('1. list(df_ctrl_cadid.columns)', list(df_ctrl_cadid.columns))

        df_ctrl_cadid=pd.merge(df_ctrl_cadid,df_ba_total,on=['RY_DATE', 'NL','SFZH', 'CY_DATE'])
        print('2. df_ctrl_cadid.shape', df_ctrl_cadid.shape)
        print('2. list(df_ctrl_cadid.columns)', list(df_ctrl_cadid.columns))

        df_ctrl_cadid=df_ctrl_cadid.drop_duplicates()
        print('3. df_ctrl_cadid.shape', df_ctrl_cadid.shape)
        print('3. list(df_ctrl_cadid.columns)', list(df_ctrl_cadid.columns))
        pickle.dump(df_ctrl_cadid, open(config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid_v2.pkl", 'wb'), protocol=4)

    # if 5.1 in process:
    #     # 添加医疗机构ID字段
    #     load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    #     df_celebro = pickle.load(open(load_path, 'rb'))
    #
    #     df_celebro['sort_id'] = df_celebro[config.RY_DATE].groupby(df_celebro[config.SFZH]).rank()
    #
    #     df_ba_total = read_ba_sc_yljgID()
    #     df_ba_total['sort_id'] = df_ba_total[config.RY_DATE].groupby(df_ba_total[config.SFZH]).rank()
    if 5.2 in process:
        # 添加医疗机构ID字段
        load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        df_celebro = pickle.load(open(load_path, 'rb'))
        print('1. df_celebro.shape', df_celebro.shape)
        print('1. list(df_celebro.columns)', list(df_celebro.columns))
        print('1. df_celebro[config.SFZH].drop_duplicates().shape', df_celebro[config.SFZH].drop_duplicates().shape)

        df_source = df_celebro[df_celebro['is_source'] == 1].drop_duplicates(subset=['SFZH', 'CY_DATE'])
        print('2. df_source.shape', df_source.shape)

        df_ba_total = read_ba_sc_yljgID()

        df_source = pd.merge(df_source, df_ba_total, on=['RY_DATE', 'NL', 'SFZH', 'CY_DATE'])
        print('3. df_source.shape', df_source.shape)
        print('3. list(df_source.columns)', list(df_source.columns))

        df_source = df_source.drop_duplicates(subset=['RY_DATE', 'NL', 'SFZH', 'CY_DATE'])
        print('4. df_source.shape', df_source.shape)
        print('4. list(df_source.columns)', list(df_source.columns))
        pickle.dump(df_source, open(config.pdir + "Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov_v2.pkl", 'wb'),protocol=4)


    if 7 in process:
        # 匹配
        # 主要目的：得到caseID-control_candidID的df
        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))
        print('1. df_ctrl_cadid.shape', df_ctrl_cadid.shape)
        print('1. df_ctrl_cadid[config.SFZH].drop_duplicates().shape', df_ctrl_cadid[config.SFZH].drop_duplicates().shape)
        print('1. list(df_ctrl_cadid.columns)', list(df_ctrl_cadid.columns))

        load_path = config.pdir + "Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov_v2.pkl"
        df_casep = pickle.load(open(load_path, 'rb'))
        print('1. df_casep.shape', df_casep.shape)
        print('1. df_casep[config.SFZH].drop_duplicates().shape', df_casep[config.SFZH].drop_duplicates().shape)
        print('1. list(df_casep.columns)', list(df_casep.columns))

        t = time.time()
        print(' match -----------')

        df_caseID_ctrlID = n2one_match_control_extract(df_casep, df_ctrl_cadid)
        pickle.dump(df_caseID_ctrlID,open(config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s1.pkl", 'wb'), protocol=4)


    if 7.1 in process:
        flag = 'group'

        if flag=='no group':
            df_caseID_ctrlID = pickle.load(open(config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s1.pkl", 'rb'))
            print('df_caseID_ctrlID')
            print(df_caseID_ctrlID)
            df_caseID_ctrlID = one_match_one(df_caseID_ctrlID)
            pickle.dump(df_caseID_ctrlID,open(config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s2.pkl", 'wb'), protocol=4)  # 1对1的df,即最终匹配

        elif flag=='group':
            df_caseID_ctrlID = pickle.load(open(config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s1.pkl", 'rb'))   # 1对多的df
            print('1. df_caseID_ctrlID',df_caseID_ctrlID.shape)
            # print(df_caseID_ctrlID)

            load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov_v2.pkl'
            df_celebrop = pickle.load(open(load_path, 'rb'))
            df_celebrop=df_celebrop[[config.SFZH,'YLJGID']]
            print('2. df_celebrop', df_celebrop.shape)

            df_caseID_ctrlID=pd.merge(df_caseID_ctrlID,df_celebrop,on=[config.SFZH])
            print('2. df_caseID_ctrlID', df_caseID_ctrlID.shape)

            yyjg_ids=df_caseID_ctrlID[config.YLJGID].drop_duplicates()
            print('yyjg_ids.shape',yyjg_ids.shape)

            lst_dfs=[]
            for index,yyjg_id in tqdm(enumerate(yyjg_ids)):
                print('----------------%d-----------------------'%(index))
                df_tmp=df_caseID_ctrlID[df_caseID_ctrlID[config.YLJGID]==yyjg_id].reset_index(drop=True)
                df_tmp = one_match_one(df_tmp)
                lst_dfs.append(df_tmp)
            pickle.dump(lst_dfs, open(config.pdir + "Project_Cerebrovascular_data/lst_dfs.pkl", 'wb'), protocol=4)

    if 7.2 in process:
        load_path = config.pdir + "Project_Cerebrovascular_data/lst_dfs.pkl"
        lst_dfs = pickle.load(open(load_path, 'rb'))
        df_caseID_ctrlID=pd.concat(lst_dfs,axis=0)
        print(df_caseID_ctrlID)
        pickle.dump(df_caseID_ctrlID,open(config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s2.pkl", 'wb'), protocol=4)

    # 8和9一起运行
    if 8 in process:
        # 连接其他字段信息
        load_path=config.pdir + "Project_Cerebrovascular_data/8_df_caseID_ctrlID_s2.pkl"
        df_caseID_ctrlID = pickle.load(open(load_path, 'rb'))

        # control集：
        # [config.SFZH, config.CY_DATE, config.DEPT_ADDRESSCODE2, config.RY_DATE, config.NL, config.XB, "ALL_DISEASE"]
        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid.pkl"
        df_ctrl_cadid = pickle.load(open(load_path, 'rb'))
        df_final_controlp = constr_final_ctrlp(df_caseID_ctrlID, df_ctrl_cadid, 'ctrl_choice')  # 匹配后的control集  （一人一行）
        print('df_final_controlp[config.SFZH].drop_duplicates().shape', df_final_controlp[config.SFZH].drop_duplicates().shape)
        print('df_final_controlp.shape', df_final_controlp.shape)
        print('df_final_controlp.columns', df_final_controlp.columns)

        # case集：
        load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov_v2.pkl'
        df_celebrop = pickle.load(open(load_path, 'rb'))
        df_final_casep = constr_final_ctrlp(df_caseID_ctrlID, df_celebrop, config.SFZH)  # 匹配后的case集  (一人一行)
        print('df_final_casep[config.SFZH].drop_duplicates().shape', df_final_casep[config.SFZH].drop_duplicates().shape)
        print('df_final_casep.shape', df_final_casep.shape)
        print('df_final_casep.columns', df_final_casep.columns)

        del df_ctrl_cadid
        del df_celebrop

    if 9 in process:
        # 添加case 和 ctrl组的共病信息。

        # 载入含全部共病信息的两个文档df_ctrl_cadid_multirow、df_celebro
        load_path = config.pdir + "Project_Cerebrovascular_data/8_df_ctrl_cadid_multirow.pkl"
        df_ctrl_cadid_multirow = pickle.load(open(load_path, 'rb'))
        df_ctrl_cadid_multirow=df_ctrl_cadid_multirow[[config.SFZH,config.ALL_DISEASE]]

        load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        df_celebro = pickle.load(open(load_path, 'rb'))
        df_celebro=df_celebro[[config.SFZH,config.ALL_DISEASE]]

        # 需要添加共病信息的两个df：df_final_controlp、df_final_casep  （由步骤8得到）
        df_final_controlp, df_final_casep =add_comorbidity_info(df_ctrl_cadid_multirow, df_celebro, df_final_controlp, df_final_casep)

        save_path = config.pdir + "Project_Cerebrovascular_data/89_df_final_case_netw.pkl"
        pickle.dump(df_final_casep, open(save_path, 'wb'), protocol=4)   # 一人多行，同一个人的每条记录基本信息完全相同，但疾病信息不同
        save_path = config.pdir + "Project_Cerebrovascular_data/89_df_final_controlp_netw.pkl"
        pickle.dump(df_final_controlp, open(save_path, 'wb'), protocol=4)  # 一人多行，同一个人的每条记录基本信息完全相同，但疾病信息不同




