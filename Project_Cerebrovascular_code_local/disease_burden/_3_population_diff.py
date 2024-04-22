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



def age_layering():
    """

    :return:
    """

def age_plot():
    f1 = open(
        r"E:\YangPing\code\Predict_High_Cost_v6\Predict_High_Cost_v6\Predict_High_Cost_v2\data\med_data_mtx_dic\df_sfzh_age.pkl",
        'rb')
    df_age = pickle.load(f1)

    plt.figure(figsize=(20, 10))
    sns.histplot(data=df_age, x="NL", stat="probability", discrete=True, alpha=0.7, kde=True)
    sns.color_palette("Paired")
    # sns.set(style="white")
    # sns.set(rc={"lines.linewidth": 1})
    sns.set_style("whitegrid")
    # sns.set(style="white")

    # mpl.rcParams['font.size'] = 16
    # plt.rcParams["figure.edgecolor"] = 'white'
    plt.rcParams["figure.edgecolor"] = "none"
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("2015-2018年IHD患者的年龄分布", fontsize=24)
    plt.xlabel("年龄", fontsize=24)
    plt.ylabel("概率", fontsize=24)
    plt.tick_params(labelsize=20)
    plt.savefig('E:\YangPing\输出\年龄分布5.png')
    plt.show()


def get_not_chronic_disease(chronic,sex):  #input:  (1,0) or (0,1)
    """两个字典 chronic=1,sex=0: 返回慢病字典;
    chronic=0,sex=1:返回性别特异性字典 (1代表男性 2代表女性 0代表不是性别特异性)"""
    df=pd.read_excel("../data/dis_sex_chronic.xlsx", sheet_name='dis_sex_chronic')
    # print(df)
    # 大于0 是宽松的条件， 等于1 是严格的条件，我们暂定宽松的条件
    if(chronic==1  and sex==0):  # 性别的1是男性 2代码女性特异性
        df_ans=df[df.chronic==0]
        dic={}
        for i in df_ans["dis"].values:
            dic[i]=1

    return dic  # 返回的是急性病


def get_comorbidity_num(df_celebro,dic_disease_prevalence_rate):
    """
    33
    :return:
    """
    dic_chronic=get_not_chronic_disease(1, 0)
    lst_cerebro=config.celebro
    t=time.time()
    print('get_comorbidity_num 开始groupby')
    chronic_comor=df_celebro.groupby(config.SFZH).apply(compute_comorb_num,dic_disease_prevalence_rate)
    print('get_comorbidity_num耗时：',(time.time()-t)/60)
    print(chronic_comor)

    # 整理数据格式
    data_df=[list(i) for i in list(chronic_comor.values)]
    # print(data_df)
    chronic_comor = pd.DataFrame(data_df, columns=['comor_num', 'age', 'gender'])
    chronic_comor['gender']=chronic_comor['gender'].apply(lambda x:'男性' if x=='1' else '女性')
    print(chronic_comor['comor_num'].mean(),chronic_comor['comor_num'].max(),chronic_comor['comor_num'].median())


    return chronic_comor



def compute_comorb_num(df_group,dic_disease_prevalence_rate):
    disease=set()
    for i in df_group[config.ALL_DISEASE]:
        disease=disease|i
    # print(disease)
    chronic_comor=disease&set(list(dic_disease_prevalence_rate.keys()))
    # print(chronic_comor)
    return len(chronic_comor),df_group[config.NL].min(),df_group[config.XB].values[0]


def generate_popu_layers_years(df_celebro):
    """
    生成数据，用于构建表：2015-2020四川省脑血管住院患者基本情况
    :return:
    """
    # 性别分层
    dic_gender_popu=dict()
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]
        df_xb=df_tmp[[config.SFZH,config.XB]].drop_duplicates()

        print("性别不一致的人数：",df_xb.shape[0]-df_xb[config.SFZH].drop_duplicates().shape[0])
        if df_xb[config.SFZH].drop_duplicates().shape[0]!=df_xb.shape[0]:
            print('------------------------------患者性别出错！-----------------------------------')
        male = df_xb[df_xb[config.XB] == '1'].shape[0]
        female=df_xb[df_xb[config.XB]=='2'].shape[0]
        dic_gender_popu[i]=[male,female]
    print(dic_gender_popu)

    # 年龄分层
    dic_agegroup_popu = dict()
    dic_age_mean_std=dict()
    # 分组
    age_group = [[18, 34], [35, 44], [45, 54], [55, 64],[65,74],[75,84],[85,df_celebro[config.NL].max()]]
    age_group_name = ['18-34岁', '35-44岁', '45-54岁', '55-64岁', '65-74岁', '75-84岁', '>=85']
    age_group=dict(zip(age_group_name,age_group))

    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]
        print('find_first_admission_per_year----')
        t=time.time()
        age_series=df_tmp.groupby(config.SFZH).apply(find_first_admission_per_year)
        print('find_first_admission_per_year----',(time.time()-t)/60)
        # print('age_series.shape',age_series.shape)
        age_mean=age_series.mean()
        age_std=age_series.std()
        dic_age_mean_std[i]=[age_mean,age_std]

        age_nda=np.array(list(age_series.values))
        dic_agegroup_year=dict()
        for j in age_group:
            dic_agegroup_year[j]=age_nda[(age_nda>=age_group[j][0])&(age_nda<=age_group[j][1])].shape[0]
        dic_agegroup_popu[i]=dic_agegroup_year
    print('dic_age_mean_std',dic_age_mean_std)
    print(dic_agegroup_popu)
    return dic_gender_popu,dic_age_mean_std,dic_agegroup_popu


def find_first_admission_per_year(df_group):
    """
    返回每个患者在指定年份内的第一次住院记录的年龄
    :return:
    """
    first_age=df_group[config.NL].min()
    return first_age


def plot_31(data):
    """
    data数据格式见：
    """

    # 画图
    sns.set_theme(style="white")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.tick_params(labelsize=30)


    for index, df_idnumber in data.groupby([config.XB]):
        case_xb = df_idnumber[config.XB].values[0]
        if case_xb=='2':
            bins_num=50
            color =  "#e84d60"
            label_value = '女性'
        else:
            bins_num=50
            color="#718dbf"
            label_value='男性'
            # print(bins_num)

        sns.distplot((df_idnumber[config.NL]), label=label_value, hist=True,bins=bins_num,color=color)

    plt.xlabel("年龄", fontsize=40)
    plt.ylabel("频率", fontsize=40)

    plt.legend(fontsize =30)
    plt.show()


def plot_33(chronic_comor):

    # 画图
    sns.set_theme(style="white")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tick_params(labelsize=30)
    # ['comor_num', 'age', 'gender']
    sns.lineplot(data=chronic_comor, x="comor_num", y="age", hue="gender",markers=True, dashes=False, linewidth=5)

    plt.xlabel("慢性共病个数", fontsize=40)
    plt.ylabel("年龄", fontsize=40)

    plt.legend(fontsize=30)
    plt.show()



def statistical_age_gender(age__gender_fp,save_path_age,save_path_gender):
    """统计：# 男女比   以及   平均年龄：18 - 34岁占比、35 - 64占比、65岁以上占比"""
    age=age__gender_fp[config.NL].copy()
    gender=age__gender_fp[config.XB].copy()
    age_stat=age.groupby(age).count()
    gender_stat=gender.groupby(gender).count()
    print(age_stat)
    print(gender_stat)

    age_stat.to_csv(save_path_age)
    gender_stat.to_csv(save_path_gender)


def generate_table32(df_celebro, save_file):
    """
    表：2015-2020四川省脑血管住院患者基本情况
    :param df_celebro:
    :param save_file:
    :return:
    """
    dict_year_popu=get_population(df_celebro)
    print(dict_year_popu)
    dic_gender_popu,dic_age_mean_std,dic_agegroup_popu=generate_popu_layers_years(df_celebro)

    # 构表： 2015-2020四川省脑血管住院患者基本情况
    popu_year=list(dict_year_popu.values())
    nda_age_mean_std=np.array(list(dic_age_mean_std.values()))
    age_mean=nda_age_mean_std[:,0]
    age_std=nda_age_mean_std[:,1]
    padding=[-1 for i in range(len(popu_year))]
    dic_table32 = {'脑血管住院总人数': popu_year, '年龄 均值': age_mean, '年龄 方差': age_std, '年龄分组[例(%)]': padding}

    tmp=[]
    for i in dic_agegroup_popu:
        tmp.append(list(dic_agegroup_popu[i].values()))
    tmp=np.array(tmp)
    tmp=tmp.T
    j=0
    for i in list(dic_agegroup_popu['2015'].keys()):
        dic_table32[i]=tmp[j]
        j+=1

    # 性别人数和比例
    nda_gender = np.array(list(dic_gender_popu.values()))
    male = nda_gender[:, 0]
    female = nda_gender[:, 1]
    sum_=male+female
    male_r=male/sum_*100
    female_r=female/sum_*100

    dic_table32['性别[例(%)]']=padding
    dic_table32['男 数值']=male
    dic_table32['女 数值']=female
    dic_table32['男 比例'] = male_r
    dic_table32['女 比例'] = female_r

    # 年龄组比例
    tmp =tmp / popu_year*100
    j = 0
    for i in list(dic_agegroup_popu['2015'].keys()):
        dic_table32[i+' 比例'] = tmp[j]
        j += 1

    table32=pd.DataFrame(dic_table32.values(),columns= ['2015', '2016', '2017', '2018', '2019', '2020'])
    table32['项目']=list(dic_table32.keys())
    print(table32)
    table32=table32.round(2)
    table32.to_csv(save_file)
    return table32
    # c['占比'] = c['占比'].apply(lambda x: x).round(3)



def get_population(df):
    # 统计人数
    dict_year_popu = dict()  # 分年统计总人数
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (df[config.CY_DATE] <= pd.to_datetime(year_f))]
        dict_year_popu[i] = df_tmp[config.SFZH].drop_duplicates().shape[0]
    return  dict_year_popu


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


def construct_sparse_matrix(df_sfzh_diseases):
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
    # if save_path!=None:
        # pickle.dump(patient_disease_csr_matrix, open(save_path, 'wb'), protocol=4)
        # pickle.dump(dic_cols, open(save_path_dic_cols, 'wb'), protocol=4)
    print("创建稀疏矩阵，保存稀疏矩阵 耗时：%.3f 分钟"%((time.time()-pastt3)/60))
    return  patient_disease_csr_matrix,dic_cols,dic_rows



def choose_diseases_based_on_matrix(df_sfzh_diseases,num_male,num_female,prevalence_threshold,flag):
    """
    处理稀疏矩阵的列，去除流行率小于1%的疾病
    输入参数：
        _num_male,nem_female为纳入患者中的男女人数，由函数construct_sex_count_dict()可计算得到
        _flag： flag为1，则去除I20-I25，共病网络不考虑这六种疾病 ; flag=0,则将I20-I25纳入共病网络
        _prevalence_threshold:选取的疾病的流行率下限
        _save_dir:读取及保存文件的文件夹名字
    输出文件：csc_matrix_final.pkl 去除了流行率小于1%的列后的新矩阵；dic_cols_new.pkl 新的疾病-列映射
    返回值：_dic_disease_prevalence   返回疾病流行度字典，与稀疏矩阵的列顺序对应
    note:该函数中，将csr矩阵转换为了csc矩阵(多余...)
    """

    patient_disease_csr_matrix,dic_cols,dic_rows=construct_sparse_matrix(df_sfzh_diseases)

    print("csr->csc matrix--------------------------------")
    patient_disease_csc_matrix=patient_disease_csr_matrix.tocsc()  #转变为按列存储

    print("去除流行率小于1%的疾病前，稀疏矩阵的维度：", patient_disease_csc_matrix.shape)
    print("除去流行程度小于1%的疾病-----------------------")
    pastt=time.time()

    if flag==1:
        # ###去除cerebro
        for i in config.celebro:
            if i in dic_cols:
                del dic_cols[i]

    sex_disease_dic=get_sex_chronic_disease(0,1)  # 获得性别特异性疾病的字典 ; 1:男性 2：女性
    # 处理性别特异性疾病，计算prevalence_rate时，计入的总人数是不同的
    col_names_new=[]
    dic_disease_prevalence={}
    dic_disease_prevalence_rate={}
    for key in dic_cols.keys():
        num_patient=num_female+num_male
        if key in sex_disease_dic:
            if sex_disease_dic[key]==1:
                num_patient=num_male
            else:
                num_patient=num_female
        col_index=dic_cols[key]
        if num_patient!=0:
            prevalence_rate=patient_disease_csc_matrix[:,col_index].sum()/num_patient
            if (prevalence_rate>prevalence_threshold):
                col_names_new.append(key)
                dic_disease_prevalence[key]=patient_disease_csc_matrix[:,col_index].sum()
                dic_disease_prevalence_rate[key]=prevalence_rate

    csc_matrix_final=patient_disease_csc_matrix[:,[dic_cols[i] for i in col_names_new]]  # 生成新的稀疏矩阵，去除了流行率小于1%的疾病后
    print("去除流行率小于1%的疾病后，稀疏矩阵的维度：",csc_matrix_final.shape)
    dic_cols_new=dict([i for i in zip(col_names_new,range(len(col_names_new)))]) # 新的疾病-列映射
    print(" 除去流行程度小于0.01的疾病 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))

    preva_df = pd.DataFrame(dic_disease_prevalence_rate, index=[1])
    preva_df.sort_values(by=1, axis=1, ascending=False,inplace=True)
    preva_df=preva_df.iloc[:,0:20]
    print("流行率前20的疾病：")
    for i in preva_df:
        print(i, preva_df.loc[1][i])

    return  dic_disease_prevalence_rate,dic_disease_prevalence,csc_matrix_final,dic_cols_new,dic_rows  # 返回疾病流行度字典，与稀疏矩阵的列顺序对应



if __name__ == "__main__":
    """
    """
    process_set = {31: 'statistical_age_gender', 32: 'table  principal distribution', 33: 'get_comorbidity_num', 34: ''}
    process = [35]

    # 2015-2020年四川省脑血管住院患者疾病情况
    load_path =config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(load_path, 'rb'))

    # # popu_path='F:/Project_Cerebrovascular_data/cerebro_id_dfgroup.pkl'
    # # celebro_id_dfgroup=pickle.load(open(popu_path, 'rb'))
    # # popu_col_path='F:/Project_Cerebrovascular_data/cerebro_id_dfgroup_col.pkl'
    # # celebro_id_dfgroup_col=pickle.load(open(popu_col_path, 'rb'))

    # 做性别年龄分布图
    if 31 in process:
        is_plot =False
        df_source = df_celebro[df_celebro['is_source'] == 1].drop_duplicates(subset=['SFZH', 'CY_DATE'])
        print('df_source.shape',df_source.shape)
        age__gender_fp=df_source[[config.NL,config.XB]]
        save_path_31 = config.pdir +"Project_Cerebrovascular_data/median_data/31_df_age__gender_fp.pkl"
        pickle.dump(age__gender_fp, open(save_path_31, 'wb'), protocol=4)

        save_path_age=config.pdir +"Project_Cerebrovascular_data/results_tables/31-age_stat.csv"
        save_path_gender=config.pdir +"Project_Cerebrovascular_data/results_tables/31-gender_stat.csv"
        statistical_age_gender(age__gender_fp,save_path_age,save_path_gender)
        if is_plot == True:
                plot_31(age__gender_fp)  # 统计和画图

    # 表：2015-2020四川省脑血管住院患者基本情况
    if 32 in process:
        save_file = config.pdir +"Project_Cerebrovascular_data/results_tables/32- 2015-2020四川省脑血管住院患者基本情况.csv"
        table32=generate_table32(df_celebro, save_file)


    # # 民族分层（含 比例的对比）

    # 共病数量分层分布情况
    if 33 in process:
        is_plot=False
        is_read=False
        if is_read==False:
            #计算患病率
            prev_threshold=0.01

            df_source = df_celebro[df_celebro['is_source'] == 1].drop_duplicates(subset=['SFZH', 'CY_DATE'])
            tmp=df_source[[config.XB]]

            num_male=tmp[tmp[config.XB]=='1'].shape[0]
            num_female=tmp[tmp[config.XB]=='2'].shape[0]
            dic_disease_prevalence_rate,dic_disease_prevalence,csc_matrix_final,dic_cols_new,dic_rows=\
                choose_diseases_based_on_matrix(df_celebro[[config.SFZH,config.ALL_DISEASE]], num_male, num_female, prev_threshold,  flag=1)
            print(dic_disease_prevalence_rate)

            save_path =config.pdir + 'Project_Cerebrovascular_data/median_data/33_patient_disease_csr_matrix_withoutCBVD_%s.pkl'%(str(prev_threshold))  # 不包含脑血管疾病
            pickle.dump(csc_matrix_final, open(save_path, 'wb'), protocol=4)
            save_path_dic_cols = config.pdir +'Project_Cerebrovascular_data/median_data/33_dic_cols_withoutCBVD_%s.pkl'%(str(prev_threshold))
            pickle.dump(dic_cols_new, open(save_path_dic_cols, 'wb'), protocol=4)
            save_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_rate_%s.pkl"%(str(prev_threshold))  # 不包含脑血管疾病
            pickle.dump(dic_disease_prevalence_rate, open(save_file, 'wb'), protocol=4)
            save_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_disease_prevalence_%s.pkl"%(str(prev_threshold))
            pickle.dump(dic_disease_prevalence, open(save_file, 'wb'), protocol=4)
            save_file =config.pdir + "Project_Cerebrovascular_data/median_data/33_num_male_female_%s.pkl"%(str(prev_threshold))
            pickle.dump([num_male, num_female], open(save_file, 'wb'), protocol=4)
            save_file = config.pdir +"Project_Cerebrovascular_data/median_data/33_dic_rows_%s.pkl" % (str(prev_threshold))
            pickle.dump(dic_rows, open(save_file, 'wb'), protocol=4)

            # 用于画分层共病图的数据：
            chronic_comor=get_comorbidity_num(df_celebro,dic_disease_prevalence_rate)
            save_file=config.pdir +"Project_Cerebrovascular_data/median_data/33_chronic_comor_gender_age.pkl"
            pickle.dump(chronic_comor, open(save_file, 'wb'), protocol=4)

        if is_plot==True:
            chronic_comor = pickle.load(open(config.pdir +"Project_Cerebrovascular_data/median_data/33_chronic_comor_gender_age.pkl", 'rb'))
            plot_33(chronic_comor)

    if 35 in process:
        df_source = df_celebro[df_celebro['is_source'] == 1].drop_duplicates(subset=['SFZH', 'CY_DATE'])
        print('df_source.shape', df_source.shape)
        age__gender_fp = df_source[[config.NL, config.XB]]
        print('age__gender_fp[config.Nl].describe()',age__gender_fp[config.NL].describe())
        print('age__gender_fp[xb=1].describe()', age__gender_fp[age__gender_fp[config.XB]=='1'][config.NL].describe())
        print('age__gender_fp[xb=2].describe()', age__gender_fp[age__gender_fp[config.XB] == '2'][config.NL].describe())
    # # # 住院次数分布情况：按年龄性别分层
    # # load_path = 'F:/Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    # # df_celebro = pickle.load(open(load_path, 'rb'))
    # # det_admission_num()