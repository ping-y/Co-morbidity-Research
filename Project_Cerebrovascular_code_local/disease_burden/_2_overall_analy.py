import time

# import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cx_Oracle
# import seaborn as sns
import config
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle
import _1_data_analy



def get_total_population(fields, db_name,connect_info, connect_mode,code6_nd,df_celebro_sfzh):
    """
    # 读取大表总人数，获取一个值；  # flags==4,6; length(sfzh)==32; 时间区间;
    :return:
    """
    sql = _1_data_analy.get_sql(fields, db_name)
    # 读入数据库数据
    print("读入数据库数据-------")
    t = time.time()
    df = connect_db(connect_info, connect_mode, sql)
    print("读入数据库数据耗时", (time.time() - t) / 60)

    print('0.df.shape', df.shape)

    # # test
    # df=df.head(20000)

    # 筛一遍flags,只要4，6
    df=df[(df[config.FLAGS]=='4')|(df[config.FLAGS]=='6')]
    print('10.df.shape', df.shape)

    # 筛一遍身份证号，只要32位长的
    df = _1_data_analy.check_sfzh_length(df)
    df=df.reset_index(drop=True)
    print('1.df.shape', df.shape)
    print('df[config.SFZH].drop_duplicates().shape',df[config.SFZH].drop_duplicates().shape)

    # 筛一遍时间  # 按照入院时间： 2015-2020
    df = df[(df[config.CY_DATE] >= pd.to_datetime('2015-01-01')) & (df[config.CY_DATE] <= pd.to_datetime('2020-12-31'))]
    print('11.df.shape', df.shape)

    # 过一遍年龄：删除首次住院年龄小于18岁的患者的所有住院记录
    del_id = df[df[config.NL] < 18][config.SFZH].drop_duplicates().values
    # del_id = pd.DataFrame(del_id, columns=[config.SFZH])   # 要删除的sfzh
    save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)), columns=[config.SFZH])  # 要保留的id
    print("18岁以下的全因住院患者人数：", del_id.shape[0])
    old_dflen = df.shape[0]
    df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    print("18岁以下的全因住院记录数：", old_dflen - df.shape[0])

    # 过一遍现住址 ： 不在四川省内的人的所有记录都删掉
    df['is_sichuanren'] = df[config.XZZ_XZQH2].apply(lambda x: 1 if x in code6_nd else 0)
    del_id = df[df['is_sichuanren'] == 0][config.SFZH].drop_duplicates().values
    save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)),  columns=[config.SFZH])  # 要保留的id
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
    del_id = is_cosist_gender.values
    del_id = np.unique(del_id[del_id != None])
    del_id = pd.DataFrame(del_id, columns=[config.SFZH])
    df = pd.merge(df, del_id, on=[config.SFZH]).reset_index(drop=True)

    # 统计人数
    total_popu, dict_year_popu = get_population(df)

    t=time.time()
    print('开始生成control候选集：')
    df=df[[config.SFZH,config.RN,config.CY_DATE,config.DEPT_ADDRESSCODE2,config.RY_DATE,config.NL,config.XB]]

    cadid_sfzh=pd.DataFrame(list(set(df[config.SFZH])-set(df_celebro_sfzh)),columns=[config.SFZH])
    df = pd.merge(df, cadid_sfzh, on=[config.SFZH])   # 候选control
    print('生成control候选集用时：', (time.time() - t) / 60)

    return total_popu,dict_year_popu,df


def check_gender(df_group):
    if df_group[config.XB].drop_duplicates().shape[0]==1:
        return df_group[config.SFZH].values[0]   # 返回的是性别始终一致的患者id


def generate_table1(df_cerebro, total_popu,dict_year_popu,save_file):
    """
    生成表1- 脑血管疾病患者住院占比
        包括分年统计
    :return:
    """
    total_celebro_popu, dict_celebro_year_popu=get_population(df_cerebro)

    dict_celebro_year_popu['2015-2020']=total_celebro_popu
    dict_year_popu['2015-2020']=total_popu

    a=pd.DataFrame(zip(dict_celebro_year_popu.keys(),dict_celebro_year_popu.values()),columns=['时间','脑血管患者住院人数'])
    b=pd.DataFrame(zip(dict_year_popu.keys(),dict_year_popu.values()), columns=['时间', '总住院人数'])
    c=pd.merge(a,b,on=['时间'])
    c['占比']=c['脑血管患者住院人数']/c['总住院人数']
    c['占比'] = c['占比'].apply(lambda x: x).round(3)
    c.to_csv(save_file)
    return c


def get_population(df):
    # 统计人数
    total_popu = df[config.SFZH].drop_duplicates().shape[0]  # 总人数
    dict_year_popu = dict()  # 分年统计总人数
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (df[config.CY_DATE] <= pd.to_datetime(year_f))]
        dict_year_popu[i] = df_tmp[config.SFZH].drop_duplicates().shape[0]
    return total_popu, dict_year_popu


def generate_table2(df_celebro, save_file):
    """
    生成表2- 脑血管疾病子类住院人数
    :param df_celebro:
    :param save_file:
    :return:
    """
    dict_celebro_category=config.celebro_category

    total_num_celebro=df_celebro[config.SFZH].drop_duplicates().shape[0]

    dict_category_num=dict()
    for i in dict_celebro_category:
        category=set(dict_celebro_category[i])
        df_celebro['tmp_flag']=df_celebro[config.ALL_DISEASE].apply(lambda x: 1 if len(category&x)>0 else 0)
        num=df_celebro[df_celebro['tmp_flag']==1][config.SFZH].drop_duplicates().shape[0]
        dict_category_num[config.category_name[i]]=num

    del df_celebro['tmp_flag']

    a=pd.DataFrame(zip(dict_category_num.keys(),dict_category_num.values()), columns=['疾病名称', '住院人数'])
    a['在脑血管患者中占比']=a['住院人数']/total_num_celebro
    a['在脑血管患者中占比'] = a['在脑血管患者中占比'].apply(lambda x: x).round(3)
    a.to_csv(save_file)
    return a


def generate_table3(df_celebro,save_file):
    """
    生成表3- 2015-2020年四川省脑血管患者住院人次   ::   住院人次： 每年的总人次，平均每年住院次数，中位年住院次数（IQR）
    :param df_celebro:
    :param save_file:
    :return:
    """
    # # 筛一遍疾病，纳入的每条住院记录都需要包含脑血管疾病
    # df_celebro['is_celebro']=df_celebro[config.ALL_DISEASE].apply(lambda x: 1 if len(set(config.celebro)&x)>0 else 0)
    # df_celebro=df_celebro[df_celebro['is_celebro']==1].reset_index(drop=True)
    # del df_celebro['is_celebro']

    # 总住院人次（分年）;总人数
    dict_year_record= dict()  # 分年统计总人次数
    dict_year_popu = dict()  # 分年统计总人次数
    dict_year_record['2015-2020']=df_celebro.shape[0]
    dict_year_popu['2015-2020'] = df_celebro[config.SFZH].drop_duplicates().shape[0]

    dict_df_byyear=dict()   # 存放每年的住院记录；用于计算每个人的脑血管病因住院的住院次数

    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]
        dict_year_record[i] = df_tmp.shape[0]
        dict_year_popu[i] = df_tmp[config.SFZH].drop_duplicates().shape[0]
        dict_df_byyear[i]=df_tmp[[config.SFZH]]

    # 平均年住院次数
    a = pd.DataFrame(zip(dict_year_record.keys(), dict_year_record.values()), columns=['时间', '总住院人次'])
    b = pd.DataFrame(zip(dict_year_popu.keys(), dict_year_popu.values()), columns=['时间', '总住院人数'])
    c = pd.merge(a, b, on=['时间'])
    c['平均住院次数'] = c['总住院人次'] / c['总住院人数']

    # groupy sfzh一下，得到分年后，每个人的住院次数
    t=time.time()
    print('--------开始groupby---------')
    dict_IQR=dict()
    reco_per_person=df_celebro.groupby(config.SFZH).apply(lambda x: x.shape[0]).values
    dict_IQR['2015-2020']=[np.percentile(reco_per_person, i, interpolation='higher') for i in [25,50,75]]
    print("groupby一遍耗时：",(time.time()-t)/60)

    for i in dict_df_byyear:
        df_year=dict_df_byyear[i]
        reco_per_person_y = df_year.groupby(config.SFZH).apply(lambda x: x.shape[0]).values
        print('reco_per_person_y',reco_per_person_y)
        if reco_per_person_y.shape[0]!=0:
            dict_IQR[i] = [np.percentile(reco_per_person_y, j, interpolation='higher') for j in [25, 50, 75]]
        else:  # 说明这一年没有数据@_@
            dict_IQR[i]=[-1,-1,-1]
    d = pd.DataFrame(zip(dict_IQR.keys(), dict_IQR.values()), columns=['时间', 'IQR'])

    c = pd.merge(c, d, on=['时间'])
    c.to_csv(save_file)
    return c



def data_for_plot_25(df_celebro, time_gap):
    """"""
    if time_gap!='week':
        if time_gap=='month':
            time_col='CY_DATE_month'
            df_celebro['CY_DATE_month'] = df_celebro[config.CY_DATE].apply(lambda x: datetime.strptime(str(x.year) + str('-') + str(x.month), '%Y-%m'))
        elif time_gap=='day':
            time_col=config.CY_DATE

        cy_date_count=df_celebro[time_col].groupby(df_celebro[time_col]).count()
        cy_date_count_fp= pd.DataFrame(cy_date_count.values,list(cy_date_count.index), columns=["count"])


        cy_date__gender_count = df_celebro[[config.SFZH,time_col,config.XB]].groupby([time_col,config.XB]).count()
        cy_date__gender_fp = pd.DataFrame(cy_date__gender_count.values.reshape(-1,2), list(dict(cy_date__gender_count.index).keys()), columns=["male", "female"])


        df_celebro['age_group']=df_celebro['NL'].apply(lambda x: 1 if 18<=x<35 else(2 if 35<=x<65 else 3))
        cy_date__agegroup_count = df_celebro[[config.SFZH, time_col, 'age_group']].groupby([time_col, 'age_group']).count()

        # 更新一下groupby的结果：因为不是每天都有三个年龄段的患者出院，所以，按天和年龄段groupby时，会存在天+年龄段组合情况的缺失。
        date=np.unique(np.array([list(i) for i in cy_date__agegroup_count.index])[:, 0]) # 2015-1-1 - 2019.12.31
        index_all=[]
        for ind in range(df_celebro['age_group'].max()):
            tmp=list(zip(date,[ind+1 for i in range(date.shape[0])]))
            index_all.extend(tmp)

        data=dict(zip(cy_date__agegroup_count.index,cy_date__agegroup_count.values.reshape(-1)))
        dict_upgrade=dict()   # 这个是更新后的数据{(Timestamp('2015-01-01 00:00:00'), 1): 2, ......}
        for i in index_all:
            if i in data:
                dict_upgrade[i]=data[i]
            else:
                dict_upgrade[i]=0

        index_time=np.array([list(i) for i in dict_upgrade.keys()])[:,0]
        index_agegroup = np.array([list(i) for i in dict_upgrade.keys()])[:, 1]
        value=list(dict_upgrade.values())
        print(index_time)
        df_tmp=pd.DataFrame(zip(index_time,index_agegroup,value),columns=[time_col,'age_group','count'])
        tmp = df_tmp[['count',time_col, 'age_group']].groupby([time_col, 'age_group']).sum()
        cy_date__agegroup_fp = pd.DataFrame(tmp.values.reshape(-1, 3), list(dict(tmp.index).keys()), columns=["18-34", "35-64","≥65"])

        return cy_date_count_fp,cy_date__gender_fp,cy_date__agegroup_fp

    elif time_gap=='week':
        time_col=config.CY_DATE
        # df_celebro['age_group'] = df_celebro['NL'].apply(lambda x: 1 if 18 <= x < 35 else (2 if 35 <= x < 65 else 3))
        # df_celebro['age_group'] = df_celebro['NL'].apply(lambda x: 1 if 18 <= x < 65 else (2 if 65 <= x < 75 else (3 if 75 <= x < 85 else 4)))
        df_celebro['age_group'] = df_celebro['NL'].apply(
            lambda x: 1 if 18 <= x < 45 else (2 if 45 <= x < 55 else
                                               (3 if 55 <= x < 65 else
                                                 (4 if 65 <= x < 75 else
                                                  (5)))))

        cy_date__agegroup_count = df_celebro[[config.SFZH, time_col, 'age_group']].groupby(  [time_col, 'age_group']).count()

        # 更新一下groupby的结果：因为不是每天都有三个年龄段的患者出院，所以，按天和年龄段groupby时，会存在天+年龄段组合情况的缺失。
        date = np.unique(np.array([list(i) for i in cy_date__agegroup_count.index])[:, 0])  # 2015-1-1 - 2019.12.31
        index_all = []
        for ind in range(df_celebro['age_group'].max()):
            tmp = list(zip(date, [ind + 1 for i in range(date.shape[0])]))
            index_all.extend(tmp)

        data = dict(zip(cy_date__agegroup_count.index, cy_date__agegroup_count.values.reshape(-1)))
        dict_upgrade = dict()  # 这个是更新后的数据{(Timestamp('2015-01-01 00:00:00'), 1): 2, ......}
        for i in index_all:
            if i in data:
                dict_upgrade[i] = data[i]
            else:
                dict_upgrade[i] = 0

        index_time = np.array([list(i) for i in dict_upgrade.keys()])[:, 0]
        index_agegroup = np.array([list(i) for i in dict_upgrade.keys()])[:, 1]
        value = list(dict_upgrade.values())
        print(index_time)
        df_tmp = pd.DataFrame(zip(index_time, index_agegroup, value), columns=[time_col, 'age_group', 'count'])
        tmp = df_tmp[['count', time_col, 'age_group']].groupby([time_col, 'age_group']).sum()
        # cy_date__agegroup_fp = pd.DataFrame(tmp.values.reshape(-1, 3), list(dict(tmp.index).keys()),columns=["18-34", "35-64", "≥65"])
        # cy_date__agegroup_fp = pd.DataFrame(tmp.values.reshape(-1, 4), list(dict(tmp.index).keys()),columns=["18-64", "65-74", "75-84","≥85"])
        cy_date__agegroup_fp = pd.DataFrame(tmp.values.reshape(-1, 5), list(dict(tmp.index).keys()),columns=["18-44", "45-54","55-64", "65-74", "≥75"])
        cy_date__agegroup_fp=cy_date__agegroup_fp.sort_index()  # 按时间排序
        print('1. cy_date__agegroup_fp',cy_date__agegroup_fp)
        flag=[int(i/7) for i in range(cy_date__agegroup_fp.shape[0])]
        print('max(flag)',max(flag))
        cy_date__agegroup_fp['flag']=flag
        # tmp2 = cy_date__agegroup_fp[['flag', "18-34", "35-64", "≥65"]].groupby(['flag']).sum()
        # tmp2 = cy_date__agegroup_fp[['flag',"18-64", "65-74", "75-84","≥85"]].groupby(['flag']).sum()
        tmp2 = cy_date__agegroup_fp[['flag', "18-44", "45-54","55-64", "65-74",  "≥75"]].groupby(['flag']).sum()
        print(tmp2)
        print('tmp2.sort_values',list(tmp2.sort_values("≥75").index))
        # cy_date__agegroup_fp = pd.DataFrame(tmp2.values, list(tmp2.index),  columns=["18-34", "35-64", "≥65"])
        cy_date__agegroup_fp = pd.DataFrame(tmp2.values, list(tmp2.index), columns=["18-44", "45-54","55-64", "65-74",  "≥75"])

        print(cy_date__agegroup_fp)
        return cy_date__agegroup_fp


def plot_25(data):
    """
    data数据格式见： http://seaborn.pydata.org/examples/wide_data_lineplot.html
    """
    sns.set_theme(style="white")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tick_params(labelsize=30)
    plt.xlabel("时间", fontsize=40)
    plt.ylabel("住院人次", fontsize=40)

    sns.lineplot(data=data, palette="tab10", linewidth=5,linestyle='-')
    # plt.legend(bbox_to_anchor=(0, 1.05), loc=3, borderaxespad=0, fontsize=30, ncol=1)

    orig_x = [i for i in range(max(list(data.index)))]
    new_x = []
    for j in ['2015', '2016', '2017', '2018', '2019', '2020']:
        for i in range(52):
            if i==int(52/2):
                new_x.append('%s' % (j))
            else:
                new_x.append('')
    if len(new_x)<len(orig_x):
        for p in range(len(orig_x)-len(new_x)):
            new_x.append('')
    new_x=new_x[:max(list(data.index))]
    print(new_x)
    print(len(new_x))
    plt.xticks(orig_x, new_x)

    plt.legend(fontsize =30)
    plt.show()



if __name__ == "__main__":
    """
    """
    process_set = {21: 'get_total_population', 22: 'generate_table1', 23: 'generate_table2', 24: 'generate_table3',
                   25: 'groupbyCYDate_admissionCount', 26: 'groupby_CYDate_and_XZZ__admissionCount'}
    process = [25]

    already_load = False
    load_path = config.pdir+'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(load_path, 'rb'))

    # 21- 总人数获取
    # flags==4,6; length(sfzh)==32; 时间区间;
    # 得到一个人数
    if 21 in process:   # 5个半小时——扫描全川数据集，生成ctrl的候选集
        is_rerun=False
        if is_rerun==True:
            fields=[config.SFZH,config.FLAGS,config.CY_DATE,config.NL,config.XB,config.DEPT_ADDRESSCODE2,config.RY_DATE,config.XZZ_XZQH2]

            db_name='BA_DATA.BA_SC'
            # db_name = 'BA_DATA.CBVD_15_20'

            # db_name = 'scott.YP_CEREBRO'

            connect_info='ba_data/123456@127.0.0.1:1521/orcl'
            connect_mode = cx_Oracle.SYSDBA

            # 获取四川省区县编码
            region_code_path = '../data/dic_183region_code.xlsx'
            region_code = pd.read_excel(region_code_path)
            region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
            code6_nd = region_code['code6'].values
            print('code6_lst', code6_nd.shape[0], code6_nd)

            total_popu,dict_year_popu,df_ctrl_cadid=get_total_population(fields,db_name,connect_info, connect_mode,code6_nd, df_celebro[config.SFZH].drop_duplicates())  # 读数据库，获取总人数

            pickle.dump([total_popu,dict_year_popu], open('../data/temp/total_popu.pkl', 'wb'), protocol=4)
            pickle.dump(df_ctrl_cadid, open(config.pdir+'Project_Cerebrovascular_data/21_df_ctrl_cadid.pkl', 'wb'), protocol=4)
        else:
            total_popu, dict_year_popu=pickle.load(open('../data/temp/total_popu.pkl','rb'))

    # 22- 表1- 脑血管疾病患者住院占比
    if 22 in process:
        print('22')
        save_file=config.pdir+"Project_Cerebrovascular_data/results_tables/表1- 脑血管疾病患者住院占比.csv"
        table1=generate_table1(df_celebro, total_popu, dict_year_popu,save_file)

    # 23- 表2- 脑血管疾病子类住院人数
    if 23 in process:
        print('23')
        save_file = config.pdir+"Project_Cerebrovascular_data/results_tables/表2- 脑血管疾病子类住院人数.csv"
        table2 = generate_table2(df_celebro, save_file)


    # 24- 表3- 2015-2020年四川省脑血管患者住院人次   ::   住院人次： 每年的总人次，平均每年住院次数，中位年住院次数（IQR）
    if 24 in process:
        print('24')
        # 注意：是否保留非脑血管疾病住院的住院记录！！！

        save_file =config.pdir+"Project_Cerebrovascular_data/results_tables/表3- 2015-2020年四川省脑血管患者住院人次.csv"
        table3= generate_table3(df_celebro, save_file)

    if 25 in process:   # 按时间顺序统计住院人次： 每天一计； ——>折线图
        print('25')
        is_plot = False
        time_gap = ['week','day','month'][0]
        if time_gap!='week':
            cy_date_count_fp,cy_date__gender_fp,cy_date__agegroup_fp = data_for_plot_25(df_celebro, time_gap)
            save_path_25 = config.pdir+"Project_Cerebrovascular_data/median_data/25_groupbyCYDate_admissionCount.pkl"
            pickle.dump([cy_date_count_fp, cy_date__gender_fp, cy_date__agegroup_fp], open(save_path_25, 'wb'),  protocol=4)  # 性别没有明显差别，按照年龄分层即可
        else:
            cy_date__agegroup_fp=data_for_plot_25(df_celebro, time_gap)
            save_path_25 = config.pdir+"Project_Cerebrovascular_data/median_data/25_groupbyCYDate_admissionCount_5groups.pkl"
            pickle.dump(['-', '-', cy_date__agegroup_fp], open(save_path_25, 'wb'),  protocol=4)  # 性别没有明显差别，按照年龄分层即可

        if is_plot==True:
            plot_25(cy_date__agegroup_fp)


    if 26 in process:

        cy_date_xzz_count = df_celebro[[config.SFZH,config.CY_DATE,config.XZZ_XZQH2]]\
            .groupby([config.CY_DATE,config.XZZ_XZQH2]).count()   # cy_date_xzz_count is a df

        tmp = np.array([list(i) for i in cy_date_xzz_count.index])
        tmp = np.concatenate((tmp, cy_date_xzz_count[config.SFZH].values.reshape(-1, 1)), axis=1)

        c = pd.DataFrame(tmp, columns=[config.CY_DATE, config.XZZ_XZQH2, 'admission count'])

        save_path_26 = config.pdir+"Project_Cerebrovascular_data/median_data/26_groupby_CYDate_and_XZZ__admissionCount.pkl"
        pickle.dump(c, open(save_path_26, 'wb'), protocol=4)


