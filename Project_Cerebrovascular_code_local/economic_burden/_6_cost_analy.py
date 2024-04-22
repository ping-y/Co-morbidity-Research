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
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.sparse import *


def get_population(df):
    # 统计人数
    dict_year_popu = dict()  # 分年统计总人数
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (df[config.CY_DATE] <= pd.to_datetime(year_f))]
        dict_year_popu[i] = df_tmp[config.SFZH].drop_duplicates().shape[0]
    return  dict_year_popu


def generate_table61(df_celebro, region_code,save_file):
    """
    表：2015-2020四川省脑血管住院患者的年住院费用和年住院时长
    """
    print('df_celebro.shape',df_celebro.shape)
    print('df_celebro[df_celebro[config.ZFY]>=0].shape',df_celebro[df_celebro[config.ZFY]>=0].shape)

    # 把住院费用为负的人去掉
    del_id = df_celebro[df_celebro[config.ZFY] < 0][config.SFZH].drop_duplicates().values
    save_id = pd.DataFrame(list(set(df_celebro[config.SFZH].drop_duplicates().values) - set(del_id)), columns=[config.SFZH])  # 要保留的id
    print("住院费用为负的人数：", del_id.shape[0])
    df_celebro = pd.merge(df_celebro, save_id, on=[config.SFZH]).reset_index(drop=True)

    if df_celebro.shape[0]!=df_celebro[df_celebro[config.ZFY]>=0].shape[0]:
        raise Exception("有住院费用缺失的患者！")

    df_celebro['cy_year']=df_celebro[config.CY_DATE].apply(lambda x:x.year)
    df_celebro['los']=df_celebro[config.CY_DATE]-df_celebro[config.RY_DATE]
    df_celebro['los'] =df_celebro['los'] .astype('str').apply(lambda x:x[:-5]).astype('int32')
    # print('los',df_celebro['los'])

    id_cost_year = df_celebro[[config.SFZH, 'cy_year', config.ZFY, 'los']].groupby([config.SFZH,'cy_year'],as_index=False).mean()  # 一个id有多行，每行对应一年
    print('id_cost_year',id_cost_year)

    # 再次groupby  ,  得到每个患者的年均住院费用
    id_cost_year2=id_cost_year[[config.SFZH,  config.ZFY,  'los']].groupby([config.SFZH],as_index=False).mean() # 一个id有一行，对应于该患者的年均...
    print('id_cost_year2', id_cost_year2)

    print("住院费用为0的人数", id_cost_year2[id_cost_year2['ZFY'] == 0].shape)
    print("住院费用最低", id_cost_year2['ZFY'].min())
    print("住院los最长", id_cost_year2['los'].max())
    print("住院los>180d", id_cost_year2[id_cost_year2['los'] > 180].shape)

    # 剔除los>180d和住院费用为0的患者
    id_cost_year2 = id_cost_year2[(id_cost_year2['ZFY'] > 0) & (id_cost_year2['los'] <= 180)& (id_cost_year2['los'] >=0)]

    # 计算年均住院费用的统计量
    mean_c,std_c=id_cost_year2['ZFY'].mean(),id_cost_year2['ZFY'].std()
    iqr_c=np.array([np.percentile(id_cost_year2['ZFY'].values, i, interpolation='higher') for i in [50, 25, 75]])
    mean_los,std_los = id_cost_year2['los'].mean(), id_cost_year2['los'].std()
    iqr_los=[ np.percentile(id_cost_year2['los'].values, i, interpolation='higher') for i in [50, 25, 75]]

    dic_61=dict()
    dic_61['全部患者']=[mean_c/10000,std_c/10000,iqr_c/10000,mean_los,std_los,iqr_los]

    padding=[0 for i in range(6)]


    # 先计算年龄df，然后merge
    age_group = [[18, 44], [45, 54], [55, 64], [65, 74], [75, df_celebro[config.NL].max()]]
    dic_age_group=dict(zip(range(len(age_group)),age_group))
    df_age=df_celebro[df_celebro['is_source']==1].drop_duplicates(subset=['SFZH', 'CY_DATE'])[[config.SFZH,config.NL,config.XB,config.XZZ_XZQH2]]
    df_age['age_group']=df_age[config.NL].apply(lambda x: 0 if x>=age_group[0][0] and x<=age_group[0][1] else
                                                                                        ( 1 if x>=age_group[1][0] and x<=age_group[1][1] else
                                                                                        ( 2 if x>=age_group[2][0] and x<=age_group[2][1] else
                                                                                        ( 3 if x>=age_group[3][0] and x<=age_group[3][1] else
                                                                                        ( 4 if x>=age_group[4][0] and x<=age_group[4][1] else
                                                                                        ( 5 ) ) )) ))

    city_re = set(region_code[region_code['城乡划分'] == '城市地区']['code6'].values)
    country_re = set(region_code[region_code['城乡划分'] == '农村地区']['code6'].values)
    print('city_re',city_re)
    df_age['is_city']=df_age[config.XZZ_XZQH2].apply(lambda x:1 if len(set([x])&city_re)>0 else (0 if len(set([x])&country_re)>0 else 2))
    df_age=df_age[[config.SFZH,config.XB, 'age_group','is_city' ]]
    print('1',id_cost_year2.shape)
    print('3',df_age.shape)
    id_cost_year2 = pd.merge(id_cost_year2, df_age, on=[config.SFZH]).reset_index(drop=True)
    print('2',id_cost_year2.shape)
    print('2', id_cost_year2.columns)

    # 年龄组
    dic_61['年龄(岁)'] = padding
    xb_zfy = id_cost_year2[[config.ZFY, 'los', 'age_group']].groupby(['age_group'], as_index=False)
    mean_df = xb_zfy.mean()
    std_df = xb_zfy.std()
    mid_cost_dict = {}
    mid_los_dict = {}
    for name, group in xb_zfy:  # 查看分组的具体信息
        mid_cost_dict[name] = np.array([np.percentile(group['ZFY'].values/10000, i, interpolation='higher') for i in [50, 25, 75]])
        mid_los_dict[name] = np.array(  [np.percentile(group['los'].values, i, interpolation='higher') for i in [50, 25, 75]])
    print(mid_cost_dict)
    print(mid_los_dict)
    for i in range(len(age_group)):
        dic_61[str(age_group[i])] = [mean_df[mean_df['age_group'] == i][config.ZFY].values[0]/10000,
                                     std_df[std_df['age_group'] == i][config.ZFY].values[0]/10000,
                                     mid_cost_dict[i],
                                     mean_df[mean_df['age_group'] == i]['los'].values[0],
                                     std_df[std_df['age_group'] == i]['los'].values[0],
                                     mid_los_dict[i]]


    # 性别
    dic_61['性别'] = padding
    xb_zfy = id_cost_year2[[config.ZFY,  'los',config.XB]].groupby([config.XB],as_index=False)
    mean_df=xb_zfy.mean()
    std_df=xb_zfy.std()
    mid_cost_dict={}
    mid_los_dict={}
    for name, group in xb_zfy:  # 查看分组的具体信息
        mid_cost_dict[name]=np.array([np.percentile(group['ZFY'].values/10000, i, interpolation='higher') for i in [50, 25, 75]])
        mid_los_dict[name] = np.array( [np.percentile(group['los'].values, i, interpolation='higher') for i in [50, 25, 75]])
    print(mid_cost_dict)
    print(mid_los_dict)
    dic_61['男性']=[mean_df[mean_df[config.XB]=='1'][config.ZFY].values[0]/10000,
                  std_df[std_df[config.XB]=='1'][config.ZFY].values[0]/10000,
                  mid_cost_dict['1'],
                  mean_df[mean_df[config.XB]=='1']['los'].values[0],
                  std_df[std_df[config.XB]=='1']['los'].values[0],
                  mid_los_dict['1']]
    dic_61['女性']=[mean_df[mean_df[config.XB]=='2'][config.ZFY].values[0]/10000,
                  std_df[std_df[config.XB]=='2'][config.ZFY].values[0]/10000,
                  mid_cost_dict['2'],
                  mean_df[mean_df[config.XB]=='2']['los'].values[0],
                  std_df[std_df[config.XB]=='2']['los'].values[0],
                  mid_los_dict['2']]

    # 城乡分布
    dic_61['城乡'] = padding
    xb_zfy = id_cost_year2[[config.ZFY,  'los', 'is_city']].groupby(['is_city'],as_index=False)
    mean_df=xb_zfy.mean()
    std_df=xb_zfy.std()
    mid_cost_dict={}
    mid_los_dict={}
    for name, group in xb_zfy:  # 查看分组的具体信息
        mid_cost_dict[name]=np.array([np.percentile(group['ZFY'].values/10000, i, interpolation='higher') for i in [50, 25, 75]])
        mid_los_dict[name] = np.array( [np.percentile(group['los'].values, i, interpolation='higher') for i in [50, 25, 75]])
    print(mid_cost_dict)
    print(mid_los_dict)
    dic_61['城市地区']=[mean_df[mean_df['is_city']==1][config.ZFY].values[0]/10000,
                    std_df[std_df['is_city']==1][config.ZFY].values[0]/10000,
                    mid_cost_dict[1],
                  mean_df[mean_df['is_city']==1]['los'].values[0],
                    std_df[std_df['is_city']==1]['los'].values[0],
                    mid_los_dict[1]]
    dic_61['农村地区']=[mean_df[mean_df['is_city']==0][config.ZFY].values[0]/10000,
                    std_df[std_df['is_city']==0][config.ZFY].values[0]/10000,
                    mid_cost_dict[0],
                  mean_df[mean_df['is_city']==0]['los'].values[0],
                    std_df[std_df['is_city']==0]['los'].values[0],
                    mid_los_dict[0]]

    # 分年统计
    dic_61['时间趋势'] = padding
    for i in [2015,2016,2017,2018,2019,2020]:
        print('----------',str(i))
        df_year=id_cost_year[id_cost_year['cy_year']==i]
        df_year = df_year[(df_year['ZFY'] != 0) & (df_year['los'] <= 180) & (df_year['los'] >=0)]

        # 计算年均住院费用的统计量
        if df_year.shape[0]>0:
            mean_c, std_c = df_year['ZFY'].mean(), df_year['ZFY'].std()
            iqr_c = np.array([np.percentile(df_year['ZFY'].values, q, interpolation='higher') for q in [50, 25, 75]])
            mean_los, std_los = df_year['los'].mean(), df_year['los'].std()
            iqr_los = [np.percentile(df_year['los'].values, q, interpolation='higher') for q in [50, 25, 75]]

            dic_61[str(i)+'年'] = [mean_c / 10000, std_c / 10000, iqr_c / 10000, mean_los, std_los, iqr_los]
        else:
            dic_61[str(i)+'年'] = padding

    table61 = pd.DataFrame(dic_61.values(), columns=['均值', '标准差', '中位数(IQR)', 'los 均值', 'los 标准差', 'los 中位数(IQR)'])
    table61['项目'] = list(dic_61.keys())
    print(table61)
    table61 = table61.round(2)
    table61.to_csv(save_file)
    return table61


def generate_table62(df_celebro, save_file):
    # check
    # 去除zfy不是大于0的住院记录，去除los>180天的住院记录
    df_celebro['los'] = df_celebro[config.CY_DATE] - df_celebro[config.RY_DATE]
    df_celebro['los'] = df_celebro['los'].astype('str').apply(lambda x: x[:-5]).astype('int32')
    df = df_celebro[(df_celebro['ZFY'] > 0) & (df_celebro['los'] <= 180)& (df_celebro['los'] >=0)]
    print('住院费用不是大于0的，或los>180的记录数：',df_celebro.shape[0]-df.shape[0])

    # 第一遍 去费用子类有缺失和去负值的住院记录
    df=check_sub_cost_1(df)
    print('第一遍 去有缺失和去负值的住院记录后， df.shape',df.shape)
    # 第二遍 纳入单项之和等于zfy的住院记录
    df = check_sub_cost_2(df)
    df['cha']=df[config.ZFY]-df['sum_sub_cost']
    print(df['cha'].describe())
    print()
    print(df[df['cha']==0].shape[0])     # 1111363条住院记录 /1839516
    df=df[df['cha']==0]   # 只要费用构成符合加和逻辑的住院记录

    col = ['ZB' + str(j) for j in range(1, 10)]
    col.append(config.QTF)

    dic_year_subcost=dict()
    index_=None
    for flag in ['2015','2016','2017','2018','2019','2020']:
        year_s = flag + '-01-01'
        year_f = flag + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (df[config.CY_DATE] <= pd.to_datetime(year_f))]
        # 统计每年的人数
        popu_year=df_tmp[config.SFZH].drop_duplicates().shape[0]

        df_tmp=df_tmp[col]   # 行：住院记录； 列：10个费用子类
        sum_sub_cost=df_tmp.sum()   # a series
        # sum_sub_cost/=10000

        dic_year_subcost[flag]=sum_sub_cost.values
        dic_year_subcost[flag] = dic_year_subcost[flag] / popu_year  # 123
        index_=list(sum_sub_cost.index)

        # 占比
        sum_total=sum_sub_cost.sum()
        rate=100*sum_sub_cost/sum_total
        dic_year_subcost[flag+' 比例']=rate.values

    dic_year_subcost['费用构成[总额(%)] code']=index_
    dic_code_costname={config.ZB1:'综合医疗服务类',config.ZB2:'诊断类',config.ZB3:'治疗类',config.ZB4:'康复类',config.ZB5:'中医类',
                       config.ZB6:'西药类',config.ZB7:'中药类',config.ZB8:'血液和血液制品类',config.ZB9:'耗材类',config.QTF:'其他类'}
    dic_year_subcost['费用构成[总额(%)]  千万']=[dic_code_costname[i] for i in index_]
    table_62=pd.DataFrame(dic_year_subcost)
    table_62 = table_62.round(2)
    table_62.to_csv(save_file)

    return table_62

def iqr_q3(column):
    return column.quantile(0.75)

def iqr_q1(column):
    return column.quantile(0.25)

def generate_table63(df_celebro, save_file1,save_file2):
    """# table_次均住院费用——住院级别 和 住院天数"""
    # check
    # 去除zfy不是大于0的住院记录，去除los>180天的住院记录
    df_celebro['los'] = df_celebro[config.CY_DATE] - df_celebro[config.RY_DATE]
    df_celebro['los'] = df_celebro['los'].astype('str').apply(lambda x: x[:-5]).astype('int32')
    df = df_celebro[(df_celebro['ZFY'] > 0) & (df_celebro['los'] <= 180)& (df_celebro['los'] >=0)]
    print('住院费用不是大于0的，或los>180的记录数：',df_celebro.shape[0]-df.shape[0])

    #los分层
    df['los_group']=df['los'].apply(lambda x: '1-7天' if (x>=1 and x<=7) else('8-14天' if (x >= 8 and x <= 14) else(
                                                                   '15-21天' if ( x >= 15 and x <= 21) else(  '22-28天' if (x >= 22 and x <= 28) else '>=29天' ) )))

    dic_year_meancost=dict()
    dic_year_meanlos=dict()
    los_group_name=[]
    for flag in ['2015','2016','2017','2018','2019','2020']:
        year_s = flag + '-01-01'
        year_f = flag + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (df[config.CY_DATE] <= pd.to_datetime(year_f))]
        # print(df_tmp)
        yyjb_costsum = df_tmp[[config.ZFY, config.YYDJ_J]].groupby([config.YYDJ_J], as_index=False).mean()
        yyjb_costsum_sd = df_tmp[[config.ZFY, config.YYDJ_J]].groupby([config.YYDJ_J], as_index=False).std()  # 123
        yyjb_costsum_med = df_tmp[[config.ZFY, config.YYDJ_J]].groupby([config.YYDJ_J], as_index=False).median()  # 123
        yyjb_costsum_iqr1=df_tmp[[config.ZFY, config.YYDJ_J]].groupby(config.YYDJ_J, as_index=False).agg(iqr_q1)
        yyjb_costsum_iqr3 = df_tmp[[config.ZFY, config.YYDJ_J]].groupby(config.YYDJ_J, as_index=False).agg(iqr_q3)

        yyjb_lossum = df_tmp[[config.ZFY,'los_group']].groupby(['los_group'], as_index=False).mean()
        yyjb_lossum_sd = df_tmp[[config.ZFY, 'los_group']].groupby(['los_group'], as_index=False).std()  # 123
        yyjb_lossum_med = df_tmp[[config.ZFY, 'los_group']].groupby(['los_group'], as_index=False).median()   #123
        yyjb_lossum_iqr1 = df_tmp[[config.ZFY, 'los_group']].groupby('los_group', as_index=False).agg(iqr_q1)
        yyjb_lossum_iqr3 = df_tmp[[config.ZFY, 'los_group']].groupby('los_group', as_index=False).agg(iqr_q3)


        print('yyjb_lossum',yyjb_lossum)
        if yyjb_lossum.shape[0]>0:
            los_group_name=list(yyjb_lossum['los_group'].values)

        if len(list(yyjb_costsum[config.ZFY].values))!=0:
            dic_year_meancost[flag]=np.array(list(yyjb_costsum[config.ZFY].values))
            dic_year_meancost[flag + 'std'] = np.array(list(yyjb_costsum_sd[config.ZFY].values))  # 123
            dic_year_meancost[flag + 'med'] = np.array(list(yyjb_costsum_med[config.ZFY].values))  # 123
            dic_year_meancost[flag + 'iqr1'] = np.array(list(yyjb_costsum_iqr1[config.ZFY].values))  # 123
            dic_year_meancost[flag + 'iqr3'] = np.array(list(yyjb_costsum_iqr3[config.ZFY].values))  # 123

            # sum_tmp=dic_year_meancost[flag].sum()
            # dic_year_meancost[flag+'占比']=100*dic_year_meancost[flag]/sum_tmp

        if len(list(yyjb_lossum[config.ZFY].values))!=0:
            dic_year_meanlos[flag]=np.array(list(yyjb_lossum[config.ZFY].values))
            dic_year_meanlos[flag + 'std'] = np.array(list(yyjb_lossum_sd[config.ZFY].values))  # 123
            dic_year_meanlos[flag + 'med'] = np.array(list(yyjb_lossum_med[config.ZFY].values))  # 123
            dic_year_meanlos[flag + 'iqr1'] = np.array(list(yyjb_lossum_iqr1[config.ZFY].values))  # 123
            dic_year_meanlos[flag + 'iqr3'] = np.array(list(yyjb_lossum_iqr3[config.ZFY].values))  # 123

            # sum_tmp = dic_year_meanlos[flag].sum()
            # dic_year_meanlos[flag + '占比'] = 100 * dic_year_meanlos[flag] / sum_tmp

    dic_year_meancost['费用构成[总额(%)]']=['二级医院','三级医院']
    print(dic_year_meancost)
    cost_table=pd.DataFrame(dic_year_meancost)
    cost_table=cost_table.round(2)
    cost_table.to_csv(save_file1)

    dic_year_meanlos['费用构成[总额(%)]'] = los_group_name
    print('dic_year_meanlos',dic_year_meanlos)
    los_table = pd.DataFrame(dic_year_meanlos)
    los_table = los_table.round(2)
    los_table.to_csv(save_file2)

    return cost_table,los_table



def check_sub_cost_2(df):
    col = ['ZB' + str(j) for j in range(1, 10)]
    col.append(config.QTF)

    df['sum_sub_cost']=0
    for i in col:
        df['sum_sub_cost']+=df[i]

    return df


def check_sub_cost_1(df):
    col = ['ZB' + str(j) for j in range(1, 10)]
    col.append(config.QTF)
    df_num=df.shape[0]

    for i in col:
        df = df.fillna({i: -1})  # 先填充缺失值
        num=df.shape[0]
        df=df[df[i]>=0]
        print('%s中，值为负或缺失的记录数:%d'%(i,num-df.shape[0]))

    print('总共删除记录数：',df_num-df.shape[0])
    return df


def check_sub_cost(df_celebro):
    col=['ZB'+str(j) for j in range(1,10)]
    # col=[]
    col.append(config.QTF)
    col.append(config.ZFJE)
    for i in col:
        print('-----------------',i,'-----------------------')
        df_tmp=df_celebro.fillna({i:-1})
        a=df_tmp[i].groupby(df_tmp[i]).count()
        print(a)
        print(a.sum())
        print('df_celebro.shape',df_tmp.shape)
        print()


if __name__ == "__main__":
    """
    """
    process_set = {61: ''}
    process = [62]

    # 2015-2020年四川省脑血管住院患者疾病情况
    load_path =config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(load_path, 'rb'))
    # print('df_celebro[config.XZZ_XZQH2]',df_celebro[config.XZZ_XZQH2])


    if 61 in process:
        region_code_path = '../data/dic_183region_code.xlsx'
        region_code = pd.read_excel(region_code_path)
        region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str

        save_file =config.pdir + "Project_Cerebrovascular_data/results_tables/61- 2015-2020年住院费用和年住院时长.csv"
        table61=generate_table61(df_celebro, region_code, save_file)

    if 62 in process:
        check=False
        generate_t=True

        if check==True:
            check_sub_cost(df_celebro)

        if generate_t==True:
            save_file = config.pdir +"Project_Cerebrovascular_data/results_tables/62- 住院费用构成.csv"
            table_62=generate_table62(df_celebro, save_file)


    if 63 in process:
        save_file1 = config.pdir +"Project_Cerebrovascular_data/results_tables/63- 次均住院费用.csv"
        save_file2 = config.pdir +"Project_Cerebrovascular_data/results_tables/63- 次均住院时长.csv"
        cost_table,los_table=generate_table63(df_celebro, save_file1,save_file2)

