import cx_Oracle
import pandas as pd
import numpy as np
import time
from scipy.sparse import *
import pickle
import os
import math
# import igraph
# from igraph import *
from tqdm import tqdm
# from disease_burden._3_population_diff import choose_diseases_based_on_matrix
import config
from  comorbidity_spectrum._7_comorbidity import get_disease_prevalence


def compute_Cij(dic_cols,csc_matrix_final):
    """
    计算Cij矩阵,是一个上三角矩阵，同时患i和j两种疾病的人数，主对称轴元素均为0
    输入参数：dic_cols:疾病-列表映射；
    输入参数：csc_matrix_final:慢病稀疏矩阵
    返回值：Cij矩阵
    """
    print("开始计算Cij--------------------------")
    pastt=time.time()
    Cij=np.zeros((len(dic_cols),len(dic_cols)))
    for i in tqdm(range(len(dic_cols))):
        for j in range(i+1,len(dic_cols)):
            cij=0
            two_cols_sum=(csc_matrix_final[:,i]+csc_matrix_final[:,j])
            for s in two_cols_sum.data:
                if s==2:  #一个人同时患两种病
                    cij+=1
            Cij[i][j]=cij

    # print(Cij)
    print(" 生成Cij矩阵 耗时：%.3f 分钟" % ((time.time() - pastt) / 60))
    return Cij


def compute_RR_significated(Cij,prevalence,N,dic_cols):
    """
    计算RR值及其置信区间 ,99%的置信区间，置信区间不包含1，则有意义
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间,同时患两种疾病的人数],....]
    """
    Cij_num=0
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            if Cij[i][j]!=0:
                Cij_num+=1
                prevalence1 = prevalence[list_cols_name[i]]
                prevalence2 = prevalence[list_cols_name[j]]
                RR_ij=(Cij[i][j]/prevalence1)*(N/prevalence2)
                if RR_ij<0:
                    print("#############RR 溢出啦###########################################",RR_ij)

                Sigma=1/Cij[i][j]+(1/prevalence1)*(1/prevalence2)-1/N-(1/N)*(1/N)   #会产生除零错误，所以应该在计算前判断Cij是否为零；（Cij为零时，RR值也为零）
                low=RR_ij*np.exp(-1*2.56*Sigma)
                high=RR_ij*np.exp(2.56*Sigma)
                if(RR_ij>1 and low>1):  #这里只考虑了两个节点联系比随机情况下更强的情况
                    edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,RR_ij,low,high,Cij[i][j]])
                    #上面一行：添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间，同时患两种病的人数],....]
    print("不为零的Cij的个数：",Cij_num)
    print("RR，有意义的边数：", len(edge_list))
    return edge_list


def compute_phi_significated(Cij,prevalence,N,dic_cols):
    """
    计算phi值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    """
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=prevalence[list_cols_name[i]]
            prevalence2 = prevalence[list_cols_name[j]]
            a = (0.1*prevalence1) * (0.1*prevalence2) * (0.1*(N - prevalence1))*(0.1*(N - prevalence2))
            # a = prevalence1 * prevalence2 * (N - prevalence1) * (N - prevalence2)
            if (a) <= 0:
                print("##############################phi 溢出啦#######################",a)
            phi_ij=((0.1*Cij[i][j])*(0.1*N)-(0.1*prevalence1)*(0.1*prevalence2))/np.sqrt(a)
            t=0  #初始化t
            n=0
            if abs(phi_ij) < 1:  # phi=1时，会发生除零错误,|phi|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N     # 注意测试一下
                t = (phi_ij * math.sqrt(n - 2)) / np.sqrt(1 - (phi_ij ** 2))
            elif phi_ij>1 or phi_ij<-1: # 不会大于1
                print("###############有phi大于1 或者小于-1 ，考虑截断,phi值为：################",phi_ij)
                # 若phi=1，只能是这种情况：A病和B病必定同时出现，且A病和B病不单独出现，这时的phi=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
            else:
                t=2.77
                n = max(prevalence1, prevalence2)
                print("###############有phi等于-1、1 ，n = max(prevalence1, prevalence2)值为：################", n)
            if ((n>1000 and phi_ij>0 and t>=2.58) or (n>500 and phi_ij>0 and t>=2.59) or (n>200 and phi_ij>0 and t>=2.60) or (n>90 and phi_ij>0 and t>=2.63) or (n>80 and phi_ij>0 and t>=2.64) or (n>70 and phi_ij>0 and t>=2.65) or (n>60 and phi_ij>0 and t>=2.66) or (n>50 and phi_ij>0 and t>=2.68) or (n>40 and phi_ij>0 and t>=2.70) or (n>38 and phi_ij>0 and t>=2.71) or (n>35 and phi_ij>0 and t>=2.72) or (n>33 and phi_ij>0 and t>=2.73) or (n>31 and phi_ij>0 and t>=2.74) or (n>30 and phi_ij>0 and t>=2.75) or (n>28 and phi_ij>0 and t>=2.76) or (n>27 and phi_ij>0 and t>=2.77) ):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,phi_ij,t,-999,Cij[i][j]])
                # 添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]

    print("phi，有意义的边数：",len(edge_list))
    # print("########n选N,还是选max(prevalence1, prevalence2)")
    return edge_list


def compute_CCxy_significated(Cij,prevalence,N,dic_cols,save_dir):
    '''
    计算CCxy值及t值 ,99%的置信水平
    输入参数：Cij:上三角矩阵，由函数compute_Cij()计算所得；prevalence:字典，由函数choose_diseases_based_on_matrix()计算可得；N:纳入的总人数；dic_cols：疾病-稀疏矩阵列的映射
    返回值：有意义的边组成的列表edge_list
    返回列表的结构：[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CCxy值，t值，无意义位，同时患两种病的人数],....]
    '''
    edge_list=[]
    list_cols_name=list(dic_cols.keys())
    for i in range(Cij.shape[0]):
        for j in range(i+1,Cij.shape[0]):
            prevalence1=prevalence[list_cols_name[i]]
            prevalence2 = prevalence[list_cols_name[j]]
            CCxy=(0.1*Cij[i][j]*math.sqrt(2))/math.sqrt(((0.1*prevalence1)**2)+((0.1*prevalence2)**2))

            t=0  #初始化t
            n=0
            if CCxy < 0:
                print("###############有CCxy溢出啦#################", CCxy)
                t = 0
            elif CCxy < 1:  # CCxy=1时，会发生除零错误,|CCxy|>1时，会发生计算错误
                n = max(prevalence1, prevalence2)
                # n=N
                if n>1:
                    t = (CCxy * math.sqrt(n - 2)) / math.sqrt(1 - (CCxy ** 2))
                else:
                    t=0
            elif CCxy == 1:
                #若CCxy=1，只能是这种情况：对任何一个人，必定同时患A病和B病，且A病和B病不单独出现，这时的CCxy=1；因为前面步骤去除了流行度小于1%的疾病，所以这种情况基本不会发生吧
                t=0
                n = max(prevalence1, prevalence2)
                print("###############有CCxy等于1", "n= max(prevalence1, prevalence2)值为：", n)
            else:
                print("###############有CCxy大于等于1？", "CCxy值为：#################", CCxy)
            if ((n>1000 and t>=2.58) or (n>500 and t>=2.59) or (n>200 and t>=2.60) or (n>90 and t>=2.63) or (n>80 and t>=2.64) or (n>70  and t>=2.65) or (n>60 and t>=2.66) or (n>50 and t>=2.68) or (n>40 and t>=2.70) or (n>38 and t>=2.71) or (n>35 and t>=2.72) or (n>33 and t>=2.73) or (n>31 and t>=2.74) or (n>30 and t>=2.75) or (n>28 and t>=2.76) or (n>27 and t>=2.77)
                    or (n>26 and t>=2.78) or (n>25 and t>=2.79) or (n>24 and t>=2.80) or (n>23 and t>=2.81) or (n>22 and t>=2.82) or (n>21  and t>=2.83) or (n>20 and t>=2.85) or (n>19 and t>=2.86) or (n>18 and t>=2.88) or (n>17 and t>=2.90) or (n>16 and t>=2.92) or (n>15 and t>=2.95) or (n>14 and t>=2.98) or (n>13 and t>=3.01) or (n>12 and t>=3.06) or (n>11 and t>=3.11) or (n>10 and t>=3.17) or (n>9 and t>=3.25) or (n>8 and t>=3.36) or (n>7 and t>=3.50) or (n>6 and t>=3.71) or (n>5 and t>=4.03) or (n>4 and t>=4.60) or (n>3 and t>=5.84) or (n>2 and t>=9.93) or (n>1 and t>=63.66)):#这里只考虑了两个节点联系比随机情况下更强的情况
                edge_list.append([list_cols_name[i],list_cols_name[j],prevalence1,prevalence2,CCxy,t,-999,Cij[i][j]])
                #添加有意义的边到边列表中，[[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数],....]
    print("CCxy，有意义的边数:",len(edge_list))
    f1 = open(save_dir+"/med_data_mtx_dic/edge_list_CC.pkl", 'wb')
    pickle.dump(edge_list, f1)
    f1.close()
    return edge_list





def construct_network_graph(dic_disease_prevalence_rate, type, percentile,edge_list):
    '''功能：构网构图
    输入参数：percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：type: type='RR': RR；type='phi':phi；type='CC':CC
    '''

    if type=='RR':
        # edge_list_RR的结构：[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间, 同时患两种疾病的人数], ....]
        print("RR the num of edge:",len(edge_list))
        node_set=set()

        quantile_value=pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_RR=[i for i in edge_list if i[4]>=quantile_value]

        for edge in edge_list_RR:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list=sorted(list(node_set))  #节点名称，排序后
        print("RR the num of node:", len(node_name_list))
        prevalence_rate=[dic_disease_prevalence_rate[i] for i in node_name_list]  #节点对应的流行率

        g=Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name']=node_name_list
        g.vs['label']=node_name_list
        g.vs['prevalence']=prevalence_rate
        g.add_edges((edge[0],edge[1]) for edge in edge_list_RR)
        RR_list=[0 for j in range(len(edge_list_RR))]
        CI_high=[0 for j in range(len(edge_list_RR))]
        CI_low=[0 for j in range(len(edge_list_RR))]
        for edge in edge_list_RR:
            edge_id=g.get_eid(edge[0],edge[1])
            RR_list[edge_id]=edge[4]
            CI_high[edge_id] = edge[6]
            CI_low[edge_id] = edge[5]
        g.es['weight'] = RR_list
        g.es['RR_CI_high']=CI_high
        g.es['RR_CI_low']=CI_low
        print(summary(g))
        # g.write(save_dir+"/gml_dir/RR_Graph_all.gml","gml")
        # plot(g)

    if type=='phi':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数], ....]
        print("phi the num of edge:",len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_phi = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_phi:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        print("phi the num of node:", len(node_name_list))
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        g.add_edges((edge[0], edge[1]) for edge in edge_list_phi)
        phi_list = [0 for j in range(len(edge_list_phi))]
        for edge in edge_list_phi:
            edge_id = g.get_eid(edge[0], edge[1])
            phi_list[edge_id] = edge[4]
        g.es['weight'] = phi_list
        print(summary(g))
        # g.write(save_dir+"/gml_dir/phi_Graph_all.gml", "gml")
        # plot(g)

    if type=='CC':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CC值，t值，无意义位，同时患两种病的人数], ....]
        print("CC the num of edge:",len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_CC = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_CC:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        print("CC the num of node:",len(node_name_list))
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        g.add_edges((edge[0], edge[1]) for edge in edge_list_CC)
        CC_list = [0 for j in range(len(edge_list_CC))]
        for edge in edge_list_CC:
            edge_id = g.get_eid(edge[0], edge[1])
            CC_list[edge_id] = edge[4]
        g.es['weight'] = CC_list
        print(summary(g))
        # g.write(save_dir+"/gml_dir/CC_Graph_all.gml", "gml")
    return g


def core_constr_ntw( df_layer, is_load, prev_threshold, type, has_cerebro):

    dic_disease_prevalence_rate, dic_disease_prevalence, csc_matrix_final, dic_cols_new, total_popu, dic_rows = get_disease_prevalence(df_layer, is_load, has_cerebro, prev_threshold)
    Cij = compute_Cij(dic_cols_new, csc_matrix_final)
    if type=='phi':
        edge_list = compute_phi_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]), dic_cols_new)
    elif type=='RR':
        edge_list = compute_RR_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]), dic_cols_new)

    return dic_disease_prevalence_rate, Cij, edge_list


def save_info(dic_disease_prevalence_rate, Cij, edge_list,prev_threshold,type,straight_info):
    save_path = config.pdir +"Project_Cerebrovascular_data/median_data/95_%s_dic_disease_prevalence_rate_%s.pkl" % ( straight_info, str(prev_threshold))
    pickle.dump(dic_disease_prevalence_rate, open(save_path, 'wb'), protocol=4)

    save_path = config.pdir + "Project_Cerebrovascular_data/median_data/95_%s_Cij_%s.pkl" % (straight_info, str(prev_threshold))
    pickle.dump(Cij, open(save_path, 'wb'), protocol=4)

    save_path = config.pdir + "Project_Cerebrovascular_data/median_data/95_%s_edge_list_%s_%s.pkl" % (straight_info, type, str(prev_threshold))
    pickle.dump(edge_list, open(save_path, 'wb'), protocol=4)


def load_set(prev_threshold,straight_info,type):
        load_file = config.pdir + "Project_Cerebrovascular_data/median_data/95_%s_edge_list_%s_%s.pkl" % (straight_info, type, str(prev_threshold))
        edge_list = pickle.load(open(load_file, 'rb'))

        load_file = config.pdir +"Project_Cerebrovascular_data/median_data/95_%s_dic_disease_prevalence_rate_%s.pkl" % ( straight_info, str(prev_threshold))
        dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))
        return edge_list,dic_disease_prevalence_rate



if __name__=="__main__":
    process_set = {91: ''}
    process = [91,93,95]
    # 91-96过程完成构网，生成gml；


    if 91 in process:
        # 构建 case - 整体网络

        # prev_threshold=0.01
        # load_file = 'F:/Project_Cerebrovascular_data/median_data/71_patient_disease_csr_matrix_withCBVD_%s.pkl' % ( str(prev_threshold))
        # csc_matrix_final = pickle.load(open(load_file, 'rb'))
        # load_file = 'F:/Project_Cerebrovascular_data/median_data/71_dic_cols_withCBVD_%s.pkl' % (str(prev_threshold))
        # dic_cols_new = pickle.load(open(load_file, 'rb'))
        # load_file = "F:/Project_Cerebrovascular_data/median_data/71_dic_disease_prevalence_%s.pkl" % (str(prev_threshold))
        # dic_disease_prevalence = pickle.load(open(load_file, 'rb'))
        # load_file = "F:/Project_Cerebrovascular_data/median_data/71_num_male_female_%s.pkl" % (str(prev_threshold))
        # total_popu = pickle.load(open(load_file, 'rb'))

        # 用匹配后的case纳入人群，来计算case网络的各种指标； （不用原始整个人群）
        load_file = config.pdir + "Project_Cerebrovascular_data/89_df_final_case_netw.pkl"
        df_final_controlp = pickle.load(open(load_file, 'rb'))

        is_load, is_save = False, False
        has_cerebro = False  # 注意慢病中是否要包含脑血管疾病本身（has_cerebro字段）！！！ # 71里面要用Fasle!!!
        prev_threshold = 0.01
        dic_disease_prevalence_rate, dic_disease_prevalence, csc_matrix_final, dic_cols_new, total_popu, dic_rows = get_disease_prevalence(df_final_controlp, is_load, has_cerebro, prev_threshold)

        Cij = compute_Cij(dic_cols_new, csc_matrix_final)

        # type = 'phi'
        type='RR'
        if type=='phi':
            edge_list = compute_phi_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]), dic_cols_new)
        elif type=='RR':
            edge_list = compute_RR_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]), dic_cols_new)

        save_path = config.pdir +"Project_Cerebrovascular_data/median_data/case_91_dic_disease_prevalence_rate_%s.pkl" % ( str(prev_threshold))
        pickle.dump(dic_disease_prevalence_rate, open(save_path, 'wb'), protocol=4)
        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/case_91_Cij_%s.pkl"% (str(prev_threshold))
        pickle.dump(Cij, open(save_path, 'wb'), protocol=4)
        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/case_91_edge_list_%s_%s.pkl" % ( type, str(prev_threshold))
        pickle.dump(edge_list,open(save_path, 'wb'), protocol=4)

    if 92 in process:
        # 接着91——  构建 case - 整体网络（生成gml画图文件）
        # 因为要根据可视化效果，调整显示的边数，所以和91分开了
        type = 'phi'
        prev_threshold=0.01
        low_edge_quantile=0

        # load_file = config.pdir + "Project_Cerebrovascular_data/median_data/case_91_edge_list_%s_%s.pkl" % (  type, str(prev_threshold))
        # edge_list = pickle.load(open(load_file, 'rb'))
        #
        # load_file =config.pdir + "Project_Cerebrovascular_data/median_data/case_91_dic_disease_prevalence_rate_%s.pkl"%(str(prev_threshold))
        # dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        g=construct_network_graph(dic_disease_prevalence_rate, type, low_edge_quantile,  edge_list)
        g.write(config.pdir + "Project_Cerebrovascular_data/results_tables/92_case_%s_graph_%s.gml"%(type, str(low_edge_quantile)), "gml")



    if 93 in process:
        # 构建control 整体网络
        load_file = config.pdir + "Project_Cerebrovascular_data/89_df_final_controlp_netw.pkl"
        df_final_controlp = pickle.load(open(load_file, 'rb'))

        is_load, is_save = False, False
        has_cerebro = False  # 注意慢病中是否要包含脑血管疾病本身（has_cerebro字段）！！！ # 71里面要用Fasle!!!
        prev_threshold = 0.01
        dic_disease_prevalence_rate, dic_disease_prevalence, csc_matrix_final, dic_cols_new, total_popu, dic_rows = get_disease_prevalence( df_final_controlp, is_load, has_cerebro, prev_threshold)

        Cij = compute_Cij(dic_cols_new, csc_matrix_final)

        type = 'RR'
        if type=='phi':
            edge_list = compute_phi_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]), dic_cols_new)
        elif type=='RR':
            edge_list = compute_RR_significated(Cij, dic_disease_prevalence, (total_popu[0] + total_popu[1]),dic_cols_new)

        save_path = config.pdir +"Project_Cerebrovascular_data/median_data/ctrl_93_dic_disease_prevalence_rate_%s.pkl" % ( str(prev_threshold))
        pickle.dump(dic_disease_prevalence_rate, open(save_path, 'wb'), protocol=4)
        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/ctrl_93_Cij_%s.pkl" % (str(prev_threshold))
        pickle.dump(Cij, open(save_path, 'wb'), protocol=4)
        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/ctrl_93_edge_list_%s_%s.pkl" % ( type, str(prev_threshold))
        pickle.dump(edge_list, open(save_path, 'wb'), protocol=4)

    if 94 in process:
        # 接着93——  构建 ctrl - 整体网络（生成gml画图文件）
        # 因为要根据可视化效果，调整显示的边数，所以和93分开了
        type = 'RR'
        prev_threshold=0.01
        low_edge_quantile=0

        # load_file = config.pdir + "Project_Cerebrovascular_data/median_data/ctrl_93_edge_list_%s_%s.pk" % (  type, str(prev_threshold))
        # edge_list = pickle.load(open(load_file, 'rb'))
        #
        # load_file = config.pdir +"Project_Cerebrovascular_data/median_data/ctrl_93_dic_disease_prevalence_rate_%s.pkl" % (str(prev_threshold))
        # dic_disease_prevalence_rate = pickle.load(open(load_file, 'rb'))

        g=construct_network_graph(dic_disease_prevalence_rate, type, low_edge_quantile,  edge_list)
        g.write(config.pdir + "Project_Cerebrovascular_data/results_tables/94_ctrl_%s_graph_%s.gml"%(type, str(low_edge_quantile)), "gml")



    # 分层
    if 95 in process:
        # 性别分层
        # 2015-2020年四川省脑血管住院患者疾病情况
        load_path = config.pdir +'Project_Cerebrovascular_data/89_df_final_case_netw.pkl'
        df_final_case = pickle.load(open(load_path, 'rb'))

        is_load, is_save = False, False
        has_cerebro = False  # 注意慢病中是否要包含脑血管疾病本身（has_cerebro字段）！！！ # 71里面要用Fasle!!!
        prev_threshold = 0.01
        type = 'RR'

        sub_nets=['gender','age group','is city']
        # sub_nets = ['gender']
        if 'gender' in sub_nets:
            # 性别分层
            for gender in ['1','2']:
                df_layer=df_final_case[df_final_case[config.XB]==gender]
                straight_info='gender=%s'%(gender)

                dic_disease_prevalence_rate, Cij, edge_list = core_constr_ntw(df_layer,  is_load, prev_threshold, type, has_cerebro)
                save_info(dic_disease_prevalence_rate, Cij, edge_list,prev_threshold,type,straight_info)

        if 'age group' in sub_nets:
            # 年龄组分层
            for i in range(5):
                age_group = [[18, 44], [45, 54], [55,64],[65,74], [75, df_final_case[config.NL].max()]]
                df_layer = df_final_case[(df_final_case[config.NL] >= age_group[i][0]) & (df_final_case[config.NL] <= age_group[i][1])]  # 年龄组分层
                straight_info = 'age_group=%s' % (str(i))

                dic_disease_prevalence_rate, Cij, edge_list = core_constr_ntw(df_layer, is_load, prev_threshold, type,has_cerebro)
                save_info(dic_disease_prevalence_rate, Cij, edge_list, prev_threshold, type, straight_info)

        if 'is city' in sub_nets:
            # 城乡分层
            region_code_path = '../data/dic_183region_code.xlsx'
            region_code = pd.read_excel(region_code_path)
            region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
            code_iscity = dict(zip(region_code['code6'], region_code['城乡划分']))
            df_final_case['is_city']=df_final_case[config.XZZ_XZQH2].apply(lambda x: code_iscity[x])

            for i in ['城市地区','农村地区']:
                df_layer=df_final_case[df_final_case['is_city']==i]
                if i == '城市地区':
                    straight_info = 'city'
                else:
                    straight_info='country'

                dic_disease_prevalence_rate, Cij, edge_list = core_constr_ntw(df_layer, is_load, prev_threshold, type, has_cerebro)
                save_info(dic_disease_prevalence_rate, Cij, edge_list, prev_threshold, type, straight_info)

    if 96 in process:
        # 接着95——  生成gml画图文件
        # 因为要根据可视化效果，调整显示的边数，所以和95分开了
        type = 'RR'
        prev_threshold=0.01
        low_edge_quantile=0

        straight_info=['gender=1','gender=2','age_group=0','age_group=1','age_group=2','city','country'][0]
        edge_list,dic_disease_prevalence_rate=load_set(prev_threshold,straight_info,type)

        g=construct_network_graph(dic_disease_prevalence_rate, type, low_edge_quantile,  edge_list)
        g.write(config.pdir + "Project_Cerebrovascular_data/results_tables/96_%s_%s_graph_%s.gml"%(straight_info, type, str(low_edge_quantile)), "gml")


    # # 对网络结构参数的统计分析
    # if 97 in process:
    #     #






