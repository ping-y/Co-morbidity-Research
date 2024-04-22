import time

import numpy as np
import os
import sys
import cx_Oracle
# import wordcloud
# import imageio
import config
# from bokeh.palettes import OrRd,Oranges,Blues,Spectral,RdBu,YlGnBu
# from bokeh.models import FactorRange
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.sparse import *
from disease_burden._3_population_diff import choose_diseases_based_on_matrix
# import fp_growth
from math import pi

# from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
# from bokeh.plotting import figure, show
# from bokeh.sampledata.unemployment1948 import data



def get_disease_prevalence(df_celebro, is_load,has_cerebro,prev_threshold):
    if is_load==True:
        if has_cerebro==True:
            load_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_dic_disease_prevalence_rate_%s.pkl"%(str(prev_threshold))  # 包含脑血管疾病
            dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))
            load_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))
            dic_disease_prevalence = pickle.load(open(load_file, 'rb'))
            load_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_num_male_female_%s.pkl"%(str(prev_threshold))
            total_popu= pickle.load(open(load_file, 'rb'))
            load_file =config.pdir + 'Project_Cerebrovascular_data/median_data/71_patient_disease_csr_matrix_withCBVD_%s.pkl'%(str(prev_threshold))
            csc_matrix_final=pickle.load(open(load_file, 'rb'))
            load_file =config.pdir + 'Project_Cerebrovascular_data/median_data/71_dic_cols_withCBVD_%s.pkl'%(str(prev_threshold))
            dic_cols_new = pickle.load(open(load_file, 'rb'))
            load_file = config.pdir + "Project_Cerebrovascular_data/median_data/71_dic_rows_%s.pkl" % (str(prev_threshold))
            dic_rows = pickle.load(open(load_file, 'rb'))

        else:
            load_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl"%(str(prev_threshold)) # 包含脑血管疾病
            dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))
            load_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))
            dic_disease_prevalence = pickle.load(open(load_file, 'rb'))
            load_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_num_male_female_%s.pkl"%(str(prev_threshold))
            total_popu = pickle.load(open(load_file, 'rb'))
            load_file = config.pdir +'Project_Cerebrovascular_data/median_data/33_patient_disease_csr_matrix_withoutCBVD_%s.pkl'%(str(prev_threshold))
            csc_matrix_final=pickle.load(open(load_file, 'rb'))
            load_file =config.pdir + 'Project_Cerebrovascular_data/median_data/33_dic_cols_withoutCBVD_%s.pkl'%(str(prev_threshold))
            dic_cols_new=pickle.load(open(load_file, 'rb'))
            load_file = config.pdir + "Project_Cerebrovascular_data/median_data/33_dic_rows_%s.pkl" % (str(prev_threshold))
            dic_rows = pickle.load(open(load_file, 'rb'))

    else:
        if has_cerebro==True:
            flag=0
        else:
            flag=1
        if 'is_source' in df_celebro.columns:
            tmp = df_celebro[df_celebro['is_source'] == 1].drop_duplicates(subset=['SFZH', 'CY_DATE'])[[config.XB]]
        else:  #是contro
            tmp=df_celebro.drop_duplicates(subset=[config.SFZH])[[config.XB]]
            print('tmp',tmp)

        num_male = tmp[tmp[config.XB] == '1'].shape[0]
        num_female = tmp[tmp[config.XB] == '2'].shape[0]
        total_popu=[num_male,num_female]
        dic_disease_prevalence_rate,dic_disease_prevalence,csc_matrix_final,dic_cols_new,dic_rows = \
            choose_diseases_based_on_matrix(df_celebro[[config.SFZH, config.ALL_DISEASE]], num_male, num_female, prev_threshold, flag=flag)

    return dic_disease_prevalence_rate,dic_disease_prevalence ,csc_matrix_final,dic_cols_new, total_popu,dic_rows


def get_table71_precalence_rate(dic_disease_prevalence_rate,dic_disease_prevalence,dic_icd3,dic_icd_sex):
    print(dic_disease_prevalence)
    disease_prevalence=sorted(dic_disease_prevalence.items(), key=lambda d: d[1],reverse=True)
    disease_prevalence_rate = sorted(dic_disease_prevalence_rate.items(), key=lambda d: d[1], reverse=True)
    dic_table=dict()

    # 以disease_prevalence_rate为序
    lst_dname = []
    for i in disease_prevalence_rate:
        if i[0] in dic_icd3:
            lst_dname.append(dic_icd3[i[0]])
    dic_table['慢性病'] = lst_dname

    dic_table['慢性病 编码']=[i[0] for i in disease_prevalence_rate]
    dic_table['患病率（男女特异性疾病的总体不同）']= [100*i[1] for i in disease_prevalence_rate]
    popu_count=[]
    is_sex=[]
    for i in dic_table['慢性病 编码']:
        popu_count.append(dic_disease_prevalence[i]/10000)
        is_sex.append(dic_icd_sex[i])
    dic_table['出院人次(单位 万)'] = popu_count
    dic_table['性别特异性'] = is_sex
    # lst_accu=[]
    # tmp=np.array(dic_table['构成比'])
    # for i in range(len(tmp)):
    #     lst_accu.append(tmp[:i+1].sum())
    # dic_table['累计构成比(%)']=lst_accu

    table_71=pd.DataFrame(dic_table)
    table_71=table_71.round(2)
    return table_71


def get_wordcloud(dic_disease_prevalence,dic_icd3,save_file):
    mask = imageio.imread('../data/sichuan_image.jpg')
    wc = wordcloud.WordCloud(font_path='simhei.ttf',  # 指定字体类型
                             background_color="white",  # 指定背景颜色
                             max_words=200,  # 词云显示的最大词数
                             max_font_size=255,  # 指定最大字号
                             scale=32,
                             # color_func = lambda *args, **kwargs: "black",
                             mask=mask)  # 指定模板

    icd_codes=dic_disease_prevalence.keys()
    prevalence=dic_disease_prevalence.values()
    d_name=[]
    for i in icd_codes:
        d_name.append(dic_icd3[i].strip())

    dic_disease_name_prevalence=dict(zip(d_name,prevalence))
    print(dic_disease_name_prevalence)
    wc = wc.generate_from_frequencies(dic_disease_name_prevalence)  ##生成词云

    wc.to_file(save_file)

    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def get_sex_chronic_disease(chronic,sex):
    #对疾病进行筛选，得到符合条件的疾病 慢性和性别特异性疾病 sex=1 代表要考虑性别特异性疾病
    """两个字典 chronic=1 返回慢病字典 sex=1 返回性别特异性字典  1代表男性 2代表女性 0代表不是性别特异性"""
    df=pd.read_excel("../data/dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
        #大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    if(chronic==1  and sex==0):#性别的1是男性 2代码女性特异性
        df_ans=df[ df.chronic>0]
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1
    else:
        df_ans=df[ df.SexDisease==1]#男性特异性
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1

        df_ans=df[ df.SexDisease==2]#女性特异性
        for i in df_ans["dis"].values:
            dic[i]=2

    return dic# 能够查询代表满足要求



def construct_sparse_matrix(df_sfzh_diseases,save_path,save_path_dic_cols):
    """
    输入参数df[['SFZH','diseases']]
    该函数输出的结果：字典：dic_cols,dic_rows；
    存储pkl文件 稀疏矩阵：patient_disease_csr_matrix
    功能：根据从数据库读入的数据生成患者-慢病稀疏矩阵，（一人多条记录合并为一条记录）
    """
    identity=df_sfzh_diseases['SFZH'].drop_duplicates()
    identity=sorted(identity)
    #身份证号-行标 字典
    dic_rows=dict(i for i in zip(identity,range(len(identity))))

    disease = set()
    pastt=time.time()
    print("开始生成所有疾病的集合-------------------")
    for diesease_set in df_sfzh_diseases[config.ALL_DISEASE]:
        disease=disease.union(diesease_set)  #所有疾病的集合，去重后
    print("未除去急性病时，疾病种类数：",len(disease))
    dic = get_sex_chronic_disease(1, 0)  # 得到慢病字典
    disease.intersection_update(dic)  #移除disease中不属于dic的元素 ，得到的是慢病集合
    print("除去急性病后，疾病种类数：", len(disease))
    print("生成所有慢病疾病的集合 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))

    disease=sorted(disease)
    #所有纳入患者中存在的慢病-列标 字典
    dic_cols=dict(j for j in zip(disease,range(len(disease))))

    row=[]
    col=[]
    print("开始生成稀疏矩阵的row,col列表")
    pastt2=time.time()
    for sfzh, diseases in df_sfzh_diseases.groupby('SFZH'):
        sfzh_row_value=dic_rows[sfzh]
        for d_set in diseases[config.ALL_DISEASE]:
            for d in d_set:
                if dic_cols.get(d, -1) != -1:
                    row.append(sfzh_row_value)
                    col.append(dic_cols[d])
    print("生成稀疏矩阵的row,col 耗时：%.3f 分钟"%((time.time()-pastt2)/60))

    #对(row,col)去重
    df_row_col=pd.DataFrame({'row':row,'col':col}).drop_duplicates()
    # print(df_row_col)
    row=df_row_col.iloc[:,0].values
    col = df_row_col.iloc[:,1].values   #ndarray类型

    print("创建稀疏矩阵中-------------------------")
    pastt3=time.time()
    #创建稀疏矩阵，用scipy.sparse中的coo_matrix
    data=np.ones(len(df_row_col),dtype=np.int8)
    patient_disease_coo_matrix=coo_matrix((data,(row,col)),shape=(len(dic_rows),len(dic_cols)),dtype=np.int32)

    patient_disease_csr_matrix=patient_disease_coo_matrix.tocsr()

    print("共慢病稀疏矩阵的维度为：",patient_disease_csr_matrix.shape)
    print("数据保存中-----------------------------")
    if save_path!=None:
        pickle.dump(patient_disease_csr_matrix, open(save_path, 'wb'), protocol=4)
        pickle.dump(dic_cols, open(save_path_dic_cols, 'wb'), protocol=4)

        print("创建稀疏矩阵，保存稀疏矩阵 耗时：%.3f 分钟"%((time.time()-pastt3)/60))
    return  patient_disease_csr_matrix,dic_cols    # 这个稀疏矩阵： 列是慢性病，但未剔除患病率小于1%的慢性病



def get_data_for_fp(csc_matrix_final, dic_disea_cols):
    """ """
    dic_cols_disea = dict([i for i in zip( range(len(dic_disea_cols)),dic_disea_cols.keys())])  # 列-疾病映射
    print('dic_cols_disea',dic_cols_disea)
    print('dic_disea_cols',dic_disea_cols)

    # 获取稀疏矩阵的行列值，，然后按照行聚合，，得到一个二维列表， ，行为病人，列为慢性病编码
    coo_mtx_final=csc_matrix_final.tocoo()
    df_colrow=pd.DataFrame(zip(coo_mtx_final.row,coo_mtx_final.col),columns=['row','col'])
    df_colrow['疾病名称']=df_colrow['col'].apply(lambda x: dic_cols_disea[x])

    lst_tmp=[]
    for row_value, df_group in df_colrow.groupby(['row']):
        lst_tmp.append(list(set(df_group['疾病名称'])))
    print('患有1%以上疾病的病人数：',len(lst_tmp))
    print('lst_tmp[:20]',lst_tmp[:20])

    # 生成fp growth的输入格式
    fp_input = np.full((len(lst_tmp), len(dic_cols_disea)),None)
    for index,row in enumerate(lst_tmp):
        # print('row',row)
        fp_input[index,:len(row)]=row
    print('list(fp_input[:5,:])',list(fp_input[:2,:]))
    fp_input=pd.DataFrame(fp_input)
    # fp_input.to_csv(save_path,header=False,index=False)
    return fp_input


def plot_hotmap_comor_spectrum(data,layer_col,region_code):   #列为分层，行为疾病（只要top 20）
    data = data.set_index(layer_col)

    data.columns.name = '共病'

    years = list(data.index)
    months = list(data.columns)

    df = pd.DataFrame(data.stack(), columns=['rate']).reset_index()
    print(df)

    colors=[i for i in RdBu[9]]
    mapper = LinearColorMapper(palette=colors, low=df.rate.min(), high=df.rate.max())

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    if  layer_col=='区县':
        factors = list(zip(region_code['市州'], region_code['县市区']))

        dic_city_country = dict(zip(region_code['县市区'], region_code['市州']))
        print('factors', factors)

        p = figure(title="",
                   x_range=list(reversed(months)), y_range=FactorRange(*factors),
                   x_axis_location="below", width=2800, height=10000,
                   tools=TOOLS, toolbar_location='below',
                   tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])
        # df=pd.DataFrame([[('成都', '锦江区'),'特发性(原发性)高血压',55],[('成都', '青羊区'),'特发性(原发性)高血压',100]],columns=['地区','共病','rate'])
        df['区县'] = df['区县'].apply(lambda x: (dic_city_country[x], x))
        print(df)
        print('factors', factors)

        p = figure(title="",
                   x_range=list(reversed(months)), y_range=FactorRange(*factors),
                   x_axis_location="below", width=10000, height=2800,
                   tools=TOOLS, toolbar_location='below',
                   tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])

    elif layer_col == '市州-城乡':
        tmp_df=region_code[['市州','城乡划分']].drop_duplicates()
        tmp_df['市州长度']=tmp_df['市州'].apply(lambda x: len(x))
        tmp_df['市州']=tmp_df['市州'].apply(lambda x: x[:2] if len(x)>3 else x)

        factors = list(zip(tmp_df['市州'], tmp_df['城乡划分']))
        p = figure(title="",
                   x_range=list(reversed(months)), y_range=FactorRange(*factors),
                   x_axis_location="below", width=8000, height=14000,
                   tools=TOOLS, toolbar_location='below',
                   tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])
        print('1',data)

    else:
        p = figure(title="",
                   x_range=years, y_range=list(reversed(months)),
                   x_axis_location="below", width=5000, height=40000,
                   tools=TOOLS, toolbar_location='below',
                   tooltips=[('date', '@Month @Year'), ('rate', '@rate%')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None

    p.yaxis.group_text_font_size = "150px"

    p.axis.major_label_text_font_size = "150px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = ['vertical', 'horizontal'][0]

    p.rect( x='共病',y=layer_col, width=1, height=1,
           source=df,
           fill_color={'field': 'rate', 'transform': mapper},
           line_color='white')

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="150px",width=100,padding=100,
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"),
                         label_standoff=6, border_line_color=None)
    p.add_layout(color_bar, 'right')

    show(p)


def data_for_hopmap_comorSpec(df_celebro, dic_rows, csc_matrix_final, dic_cols, region_code):
    # 列为分层，行为疾病（只要top 20），值为对应的患病率
    # 生成df:  字段： 【年龄、年龄组、性别（男女）、现住址（名称）——[区县、地级市、城乡]、top20的疾病二值变量、一列全1列】
    dic_row_sfzh=dict(zip(dic_rows.values(),dic_rows.keys()))
    df_sfzh_row=pd.DataFrame(zip(dic_rows.keys(),dic_rows.values()),columns=[config.SFZH,'row numb'])

    df=df_celebro[df_celebro['is_source']==1].drop_duplicates(subset=['SFZH', 'CY_DATE'])[[config.SFZH,config.XB,config.NL,config.XZZ_XZQH2]]
    print('1. df_sfzh_row.shape',df_sfzh_row.shape)
    print('1. df_sfzh_row.columns', df_sfzh_row.columns)
    df_sfzh_row = df_sfzh_row.merge(df, how='left', on=[config.SFZH])
    print('2. df_sfzh_row.shape',df_sfzh_row.shape)
    print('2. df_sfzh_row.columns', df_sfzh_row.columns)

    # 去除csc中不需要的疾病
    # col_names_new = dict(sorted(dic_disease_prevalence_rate.items(), key=lambda d: d[1], reverse=True)[:top_k]).keys()      # 需要的疾病
    # print('col_names_new', col_names_new)
    # csc_matrix_final = csc_matrix_final[:, [dic_cols[i] for i in col_names_new]]

    dic_d=dict()
    for i in dic_cols:
        dic_d[i]=csc_matrix_final[:,dic_cols[i]].toarray().reshape(-1)

    df_disea=pd.DataFrame(dic_d)
    print('df_disea',df_disea)
    print('list(df_disea.columns)',list(df_disea.columns))
    print('df_disea.shape',df_disea.shape)

    # df_sfzh_row.sort_values()        by row numb
    df_sfzh_row.sort_values(by=['row numb'],ascending=True,inplace=True)
    print('list(df_sfzh_row.columns)',list(df_sfzh_row.columns))
    print('df_sfzh_row.shape',df_sfzh_row.shape)
    # 拼接
    df_sfzh_row=pd.concat([df_sfzh_row,df_disea],axis=1)
    print('df_sfzh_row.shape',df_sfzh_row.shape)
    # print(df_sfzh_row['row numb'])

    # 生成年龄组分层
    # 先计算年龄df，然后merge
    age_group = [[18, 44], [45, 54], [55, 64], [65, 74], [75, df_celebro[config.NL].max()]]
    dic_age_group = dict(zip(range(len(age_group)), age_group))
    df_sfzh_row['年龄组'] = df_sfzh_row[config.NL].apply(lambda x: '18-44岁' if x >= age_group[0][0] and x <= age_group[0][1] else
    ('45-54岁' if x >= age_group[1][0] and x <= age_group[1][1] else
     ('55-64岁' if x >= age_group[2][0] and x <= age_group[2][1] else
      ('65-74岁' if x >= age_group[3][0] and x <= age_group[3][1] else
       ('≥75岁' if x >= age_group[4][0] and x <= age_group[4][1] else (7))))))

    print('df_sfzh_row[config.XB]', df_sfzh_row[config.XB])
    df_sfzh_row['性别'] = df_sfzh_row[config.XB].apply(lambda x: '男性' if x=='1' else '女性')

    dic_code_region=dict(zip(region_code['code6'],region_code['县市区']))
    df_sfzh_row['区县']=df_sfzh_row[config.XZZ_XZQH2].apply(lambda x: dic_code_region[x])
    dic_code_region = dict(zip(region_code['code6'], region_code['市州']))
    df_sfzh_row['市州'] = df_sfzh_row[config.XZZ_XZQH2].apply(lambda x: dic_code_region[x])
    dic_code_region = dict(zip(region_code['code6'], region_code['城乡划分']))
    df_sfzh_row['城乡划分'] = df_sfzh_row[config.XZZ_XZQH2].apply(lambda x: dic_code_region[x])

    # groupby
    # 年龄组聚合sum()
    dict_layer_df=dict()
    for layer_i in ['年龄组','性别','区县','市州']:
        layer=[layer_i]
        layer.extend(list(dic_cols.keys()))
        ageg_comorb = df_sfzh_row[layer].groupby([layer_i],as_index=False).sum()
        print('ageg_comorb',ageg_comorb)

        n40_num=ageg_comorb['N40']

        ageg_id=df_sfzh_row[[layer_i,config.SFZH]].groupby([layer_i],as_index=False).count()
        ageg_comorb[list(dic_cols.keys())]=ageg_comorb[list(dic_cols.keys())].div(ageg_id[config.SFZH].values,axis=0)*100

        # 处理性别特异性基本——前列腺增生N40
        if layer_i!='性别':
            male_num=df_sfzh_row[df_sfzh_row[config.XB]=='1'][[layer_i,config.SFZH]].groupby([layer_i],as_index=False).count()
            ageg_comorb['N40']=n40_num.div(male_num[config.SFZH].values, axis=0)*100

        print('ageg_comorb',ageg_comorb)
        print('ageg_comorb.N40',ageg_comorb['N40'])

        dict_layer_df[layer_i]=ageg_comorb

    # 市州-城乡
    x=['市州', '城乡划分' ]
    x.extend(list(dic_cols.keys()))
    print('xxxxxx',x)
    city_country = df_sfzh_row[x].groupby(['市州', '城乡划分'],as_index=False).sum()
    city_country_id = df_sfzh_row[['市州', '城乡划分',config.SFZH]].groupby(['市州', '城乡划分'], as_index=False).count()
    city_country[list(dic_cols.keys())] = city_country[list(dic_cols.keys())].div(city_country_id[config.SFZH].values, axis=0) * 100
    city_country['市州-城乡'] = list(zip(city_country['市州'], city_country['城乡划分']))
    dict_layer_df['市州-城乡'] = city_country
    print(dict_layer_df['市州-城乡'])

    # 人群差异
    tmp=dict_layer_df['性别'].copy()
    tmp.columns=dict_layer_df['年龄组'].columns
    tmp=pd.concat([dict_layer_df['年龄组'], tmp], axis=0)
    tmp_col=['人群差异']
    tmp_col.extend([i for i in tmp.columns if i!='年龄组'])
    tmp.columns = tmp_col
    dict_layer_df['人群差异']=tmp
    print('dict_layer_df人群差异', dict_layer_df['人群差异'])
    return dict_layer_df


def deal_data_for_plot_hotcomor(dict_layer_df, layer_col,dic_disease_prevalence_rate,plot_comor_num,dic_icd3):
    """"""
    disease_prevalence_rate = sorted(dic_disease_prevalence_rate.items(), key=lambda d: d[1], reverse=True)
    comor_col=list(dict(disease_prevalence_rate[:plot_comor_num]).keys())
    data=dict_layer_df[layer_col]
    comor_col.append(layer_col)
    data=data[comor_col]
    print(data)
    print()
    new_colname=[dic_icd3[i] for i in data.columns if i!=layer_col]
    new_colname.append(layer_col)
    data.columns=new_colname
    print(data)
    print(list(data.columns))
    return data


def data_for_plot_year_comor_spec(df_celebro,disease_lst):  # disease_lst： icd编码

    dict_year_comor=dict()

    for flag in ['2015','2016','2017','2018','2019','2020']:
        year_s = flag + '-01-01'
        year_f = flag + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]
        dict_d_count=dict(zip(disease_lst,[0 for _ in range(len(disease_lst))] ))

        for j in df_tmp[config.ALL_DISEASE]:
            for k in j:
                if k in dict_d_count:
                    dict_d_count[k]+=1

        dict_year_comor[flag]=list(dict_d_count.values())

    lst_df=[]
    for i in dict_year_comor:
        year_col=[i for _ in range(len(dict_year_comor[i]))]
        tmp=pd.DataFrame(zip(year_col, disease_lst, dict_year_comor[i]),columns=['年份','共病','人次(万)'])
        tmp.sort_values(by=['人次(万)'], ascending=False, inplace=True)
        tmp=tmp.reset_index(drop=True)
        tmp['排序']=np.array(list(tmp.index))+1

        lst_df.append(tmp)

    df_for_plot=pd.concat(lst_df,axis=0)
    df_for_plot['人次(万)']=df_for_plot['人次(万)']/10000
    df_for_plot=df_for_plot.round(1)
    print('df_for_plot',df_for_plot)
    print('df_for_plot.shape',df_for_plot.shape)
    return df_for_plot


def plot_year_comor_hotspec(df_for_plot_year_comor_hotspec,comor_col,dic_icd3):
    # df_for_plot_year_comor_hotspec["共病名称"] = df_for_plot_year_comor_hotspec["共病名称"].apply(lambda x: x.strip())
    dic_disea_num=dict(zip(comor_col,[i for i in range(len(comor_col))]))
    df_for_plot_year_comor_hotspec['共病名称'] = df_for_plot_year_comor_hotspec['共病'].apply(lambda x: dic_disea_num[x])


    flights = df_for_plot_year_comor_hotspec.pivot("共病名称", "年份", "人次(万)")

    year=['2015', '2016', '2017', '2018', '2019', '2020']

    text=np.zeros((len(comor_col),6))
    for i in range(6):
        for j in range(len(comor_col)):
            text[j,i]=df_for_plot_year_comor_hotspec[(df_for_plot_year_comor_hotspec['共病']==comor_col[j])
                                                     &(df_for_plot_year_comor_hotspec['年份']==year[i])]['排序'].values[0]
    text=text.astype(int)
    # print('tedt',text)
    # print('text.shape',text.shape)
    # print('flights.shape',flights.shape)
    # print('flights',flights)

    # Draw a heatmap with the numeric values in each cell
    f, ax = plt.subplots(figsize=(9, 36))
    sns.set_theme(style='white')
    # sns.heatmap(flights, annot=True, fmt=".1f", linewidths=.5, ax=ax,cmap=[i for i in RdBu[9]],square=False)
    sns.heatmap(flights, annot=text, fmt="d", linewidths=.5, ax=ax, cmap=[i for i in RdBu[9]], square=False)

    plt.rcParams["figure.edgecolor"] = "none"
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.xlabel("年份", fontsize=20)
    plt.ylabel("脑血管共病", fontsize=20)
    plt.tick_params(labelsize=20)

    plt.legend(fontsize=20)
    plt.xticks(rotation=0)

    old=list(dic_disea_num.values())
    print('old',old)
    new=[dic_icd3[i].strip() for i in dic_disea_num.keys()]
    print('new',new)

    plt.yticks(old,new)
    plt.yticks(rotation=360)
    plt.show()



if __name__ == "__main__":
    """
    """
    process_set = {71: ''}
    # process = [71]   #   is_load,is_save,has_cerebro = False,True,True
    process = [75]     #   is_load,is_save,has_cerebro = True,False,False

    # 2015-2020年四川省脑血管住院患者疾病情况
    load_path = config.pdir +'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(load_path, 'rb'))

    xlsxpath = "../data/BA_CD_MULTI.xlsx"
    dic_icd3 = pd.read_excel(xlsxpath, sheet_name='疾病分类编码')
    dic_icd3 = dict(zip(dic_icd3['类目-编码'], dic_icd3['类目']))

    xlsxpath = "../data/dis_sex_chronic.xlsx"
    dic_icd_sex = pd.read_excel(xlsxpath, sheet_name='dis_sex_chronic')
    dic_icd_sex = dict(zip(dic_icd_sex['dis'], dic_icd_sex['SexDisease']))

    region_code_path = '../data/dic_183region_code.xlsx'
    region_code = pd.read_excel(region_code_path)

    region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str


    # --------------------------------------------------------------
    # 71的使用方法： 单独运行，跑两次，分别对应下述1和2两种情况：
    # 1.
    #  is_load=False, is_save=True,has_cerebro=True跑一次——>为了得到含有脑血管疾病本身的慢病文件。
    #
    # 2. 然后，——>为了生成表71
    #  如果33没有保存相应的pkl文件，则is_load=False, is_save=True,has_cerebro=False跑一次
    #  如果33有保存相应的pkl文件，则直接用33生成的文件即可：is_load=True, is_save=False,has_cerebro=False跑一次即可
    #
    # --------------------------------------------------------------
    if 71 in process:
        # 2015-2020脑血管疾病的共病的患病率表格
        # is_load,is_save,has_cerebro = False,True,True        # 为了得到含有脑血管疾病本身的慢病文件
        is_load,is_save,has_cerebro = True,False,False     # 注意慢病中是否要包含脑血管疾病本身（has_cerebro字段）！！！ # 71里面要用Fasle!!!
        prev_threshold=0.01
        dic_disease_prevalence_rate, dic_disease_prevalence,csc_matrix_final,dic_cols_new, total_popu,dic_rows=\
            get_disease_prevalence(df_celebro, is_load,has_cerebro,prev_threshold)
        print(len(dic_disease_prevalence_rate),dic_disease_prevalence_rate)

        if is_save==True:   # 33——对应不含脑血管本身的慢病数据； 71——对应包含脑血管本身的慢病数据   Oh so complicated ...
            if has_cerebro==False:
                save_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl"%(str(prev_threshold))  # 不包含脑血管疾病
                pickle.dump(dic_disease_prevalence_rate, open(save_file, 'wb'), protocol=4)
                save_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))
                pickle.dump(dic_disease_prevalence, open(save_file, 'wb'), protocol=4)
                save_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_num_male_female_%s.pkl"%(str(prev_threshold))
                pickle.dump(total_popu, open(save_file, 'wb'), protocol=4)
                save_path =config.pdir + 'Project_Cerebrovascular_data/median_data/33_patient_disease_csr_matrix_withoutCBVD_%s.pkl'%(str(prev_threshold))
                pickle.dump(csc_matrix_final, open(save_path, 'wb'), protocol=4)
                save_path_dic_cols = 'Project_Cerebrovascular_data/median_data/33_dic_cols_withoutCBVD_%s.pkl'%(str(prev_threshold))
                pickle.dump(dic_cols_new, open(save_path_dic_cols, 'wb'), protocol=4)
                save_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_dic_rows_%s.pkl" % (str(prev_threshold))
                pickle.dump(dic_rows, open(save_file, 'wb'), protocol=4)

            else:
                save_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))  # 包含脑血管疾病
                pickle.dump(dic_disease_prevalence_rate, open(save_file, 'wb'), protocol=4)
                save_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))
                pickle.dump(dic_disease_prevalence, open(save_file, 'wb'), protocol=4)
                save_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_num_male_female_%s.pkl"%(str(prev_threshold))
                pickle.dump(total_popu, open(save_file, 'wb'), protocol=4)
                save_path =config.pdir + 'Project_Cerebrovascular_data/median_data/71_patient_disease_csr_matrix_withCBVD_%s.pkl'%(str(prev_threshold))
                pickle.dump(csc_matrix_final, open(save_path, 'wb'), protocol=4)
                save_path_dic_cols = config.pdir +'Project_Cerebrovascular_data/median_data/71_dic_cols_withCBVD_%s.pkl'%(str(prev_threshold))
                pickle.dump(dic_cols_new, open(save_path_dic_cols, 'wb'), protocol=4)
                save_file = config.pdir +"Project_Cerebrovascular_data/median_data/71_dic_rows_%s.pkl" % (str(prev_threshold))
                pickle.dump(dic_rows, open(save_file, 'wb'), protocol=4)

        # 生成2015-2020脑血管疾病的共病的患病率表格
        if has_cerebro==False:
            table_71=get_table71_precalence_rate(dic_disease_prevalence_rate,dic_disease_prevalence,dic_icd3,dic_icd_sex)
            table_71.to_csv(config.pdir +"Project_Cerebrovascular_data/results_tables/71- 2015-2020疾病谱前k位.csv")


    if 72 in process:
        # 构建词云图——体现常见共病
        is_load =  True
        has_cerebro = False
        prev_threshold=0.01

        save_file = config.pdir +"Project_Cerebrovascular_data/results_tables/72_wordcloud.png"
        dic_disease_prevalence_rate, dic_disease_prevalence,csc_matrix_final,dic_cols_new, total_popu,dic_rows = get_disease_prevalence(df_celebro, is_load,has_cerebro,prev_threshold)
        # get_wordcloud(dic_disease_prevalence,dic_icd3,save_file)


    if 73 in process:
        # 获取用于挖掘频繁集使用的数据
        prev_threshold=0.01

        load_path =config.pdir +'Project_Cerebrovascular_data/median_data/71_patient_disease_csr_matrix_withCBVD_%s.pkl'%(str(prev_threshold))
        csc_matrix_final = pickle.load(open(load_path, 'rb'))
        save_path_dic_cols = config.pdir +'Project_Cerebrovascular_data/median_data/71_dic_cols_withCBVD_%s.pkl' % (str(prev_threshold))
        dic_disea_cols = pickle.load(open(save_path_dic_cols, 'rb'))

        fp_input=get_data_for_fp(csc_matrix_final, dic_disea_cols)
        fp_input.to_csv(config.pdir +'Project_Cerebrovascular_data/median_data/73_fp_input%s.csv'% (str(prev_threshold)), header=False, index=False)

    if 74 in process:
        # 频繁项挖掘
        prev_threshold=0.01
        path = config.pdir +'Project_Cerebrovascular_data/median_data/73_fp_input%s.csv'% (str(prev_threshold))
        data_set = fp_growth.load_data(path)  # 获取并预处理数据

        for support_rate in [0.02,0.03,0.04]:
            min_support = int(support_rate*len(data_set)+0.5)  # 最小支持度
            min_conf = 0.7  # 最小置信度

            save_path =config.pdir +'Project_Cerebrovascular_data/median_data/%s_74_fp_output_rules%s.txt'% (str(support_rate),str(prev_threshold))

            fp = fp_growth.Fp_growth()
            t=time.time()
            rule_list,L,support_data = fp.generate_R(data_set, min_support, min_conf)
            print('频繁项挖掘用时：', (time.time()-t)/60)
            fp_growth.save_rule(rule_list, save_path)

            save_path = config.pdir +'Project_Cerebrovascular_data/median_data/%s_74_fp_output_L_%s.pkl' % (str(support_rate),str(prev_threshold))
            pickle.dump(L, open(save_path, 'wb'), protocol=4)

            support_data = pd.DataFrame(zip(support_data.keys(), support_data.values()))
            save_path =config.pdir + 'Project_Cerebrovascular_data/median_data/%s_74_fp_output_support_data_%s.csv' % (str(support_rate),str(prev_threshold))
            support_data.to_csv(save_path)

        # 生成表格数据

    if 75 in process:
        # 共病图谱——年龄性别分层
        prev_threshold=0.01

        load_path = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_rows_%s.pkl" % (str(prev_threshold))
        dic_rows = pickle.load(open(load_path, 'rb'))  # 身份证号-mtx的行号

        load_file = config.pdir +'Project_Cerebrovascular_data/median_data/33_patient_disease_csr_matrix_withoutCBVD_%s.pkl' % (str(prev_threshold))
        csc_matrix_final = pickle.load(open(load_file, 'rb'))

        load_file = config.pdir +'Project_Cerebrovascular_data/median_data/33_dic_cols_withoutCBVD_%s.pkl' % (str(prev_threshold))
        dic_cols_new = pickle.load(open(load_file, 'rb'))

        load_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl" % ( str(prev_threshold))
        dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        dict_layer_df=data_for_hopmap_comorSpec(df_celebro, dic_rows,csc_matrix_final,dic_cols_new,region_code)
        pickle.dump(dict_layer_df, open(config.pdir +'Project_Cerebrovascular_data/median_data/75_dict_layer_df_hotcomorSpec.pkl', 'wb'), protocol=4)

    if 76 in process:
        # 用75的数据画共病图谱
        dict_layer_df = pickle.load(open(config.pdir +'Project_Cerebrovascular_data/median_data/75_dict_layer_df_hotcomorSpec.pkl', 'rb'))
        # 年龄组
        # dict_layer_df['年龄组']
        prev_threshold=0.01
        plot_comor_num=30

        load_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl" % (str(prev_threshold))
        dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        layer_col= ['人群差异','区县','市州','市州-城乡'][3]
        # if layer_col!='市州-城乡':
        data=deal_data_for_plot_hotcomor(dict_layer_df, layer_col,dic_disease_prevalence_rate,plot_comor_num,dic_icd3)   # 选择要呈现哪些共病
        plot_hotmap_comor_spectrum(data,layer_col,region_code)  # 画图函数
        # else:
        #     plot_hotmap_comor_spectrum(dict_layer_df['市州-城乡'], layer_col, region_code)  # 画图函数

    if 77 in process:
        # 行-年；列-共病；颜色-人次
        prev_threshold = 0.01

        load_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl" % ( str(prev_threshold))
        dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        df_for_plot_year_comor_hotspec=data_for_plot_year_comor_spec(df_celebro, list(dic_disease_prevalence_rate.keys()))

        save_path = config.pdir +'Project_Cerebrovascular_data/median_data/77_df_for_plot_year_comor_hotspec_%s.csv' % ( str(prev_threshold))
        pickle.dump(df_for_plot_year_comor_hotspec, open(save_path, 'wb'), protocol=4)

    if 78 in process:
        #用77的数据画年-共病-人次图谱

        prev_threshold = 0.01
        plot_comor_num = 30

        load_file =  config.pdir +'Project_Cerebrovascular_data/median_data/77_df_for_plot_year_comor_hotspec_%s.csv' % ( str(prev_threshold))
        df_for_plot_year_comor_hotspec = pickle.load(open(load_file, 'rb'))
        load_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl" % (str(prev_threshold))
        dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        # 处理一下用于可视化的数据
        disease_prevalence_rate = sorted(dic_disease_prevalence_rate.items(), key=lambda d: d[1], reverse=True)
        comor_col = list(dict(disease_prevalence_rate[:plot_comor_num]).keys())

        df_for_plot_year_comor_hotspec['top k的共病']=df_for_plot_year_comor_hotspec['共病'].apply(lambda x: 1 if x in comor_col else 0)  # 只画出患病率最高的top-k个共病
        df_for_plot_year_comor_hotspec=df_for_plot_year_comor_hotspec[df_for_plot_year_comor_hotspec['top k的共病']==1]
        df_for_plot_year_comor_hotspec['共病名称']=df_for_plot_year_comor_hotspec['共病'].apply(lambda x: dic_icd3[x])

        print(df_for_plot_year_comor_hotspec)
        print(df_for_plot_year_comor_hotspec.shape)

        plot_year_comor_hotspec(df_for_plot_year_comor_hotspec,comor_col,dic_icd3)
