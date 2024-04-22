import cx_Oracle
import pandas as pd
import numpy as np
import time
from scipy.sparse import *
import pickle
import os
import math
import igraph
from igraph import *
from tqdm import tqdm
import config
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


def Compute_Centrality(mygraph):
    """
    输入参数：_gml_graph
    计算中心性指标: 介数中心度betweenness_centrality， 度中心度degree_centrality，强度中心度strength degree,邻居平均度
    返回值：均为字典
    """
    print('len(mygraph.nodes)',len(mygraph.nodes))
    print('len(mygraph.edges)',len(mygraph.edges))
    betweenness_centrality=nx.betweenness_centrality(mygraph, k=None, normalized=False, weight='weight', endpoints=False, seed=None)  # 介数中心度
    degree_centrality=nx.degree_centrality(mygraph)   # 点度中心性

    # 强度中心度
    # print(mygraph.edges)  # [('D50', 'D64'), ('D50', 'D70'),....
    dic_strength_centrality= dict(zip(mygraph.nodes, [0 for i in range(len(mygraph.nodes))]))  #初始化
    for i in mygraph.edges:
        weight=mygraph.edges[i]['weight']
        dic_strength_centrality[i[0]]+=weight
        dic_strength_centrality[i[1]] += weight

    # 计算邻居平均度
    dic_neighbor_deg = dict(zip(degree_centrality.keys(), [0 for i in range(len(degree_centrality))]))  # 初始化
    for i in mygraph.edges:
        dic_neighbor_deg[i[0]] += degree_centrality[i[1]]
        dic_neighbor_deg[i[1]] += degree_centrality[i[0]]
    neigh_deg=np.array(list(dic_neighbor_deg.values()))/np.array(list(degree_centrality.values()))
    dic_neighbor_deg=dict(zip(dic_neighbor_deg.keys(),neigh_deg))

    return betweenness_centrality,degree_centrality,dic_strength_centrality,dic_neighbor_deg


def compute_centrality(dict_gmls):
    lst_cent = []

    for gml_g in dict_gmls:
        betweenness_centrality,degree_centrality,dic_strength_centrality,dic_neighbor_deg = Compute_Centrality(dict_gmls[gml_g])

        for i in betweenness_centrality.keys():
            i_centr=[gml_g, i,betweenness_centrality[i],degree_centrality[i],dic_strength_centrality[i],dic_neighbor_deg[i]]
            lst_cent.append(i_centr)

    df_centrality=pd.DataFrame(lst_cent,columns=['gml_graph', 'node','betweenness_centrality','degree_centrality','strength_centrality','neighbor_mean_deg'])
    return df_centrality


def get_topk_correlation(dict_gmls):
    lst=[]
    for gml_g in dict_gmls:
        mygraph=dict_gmls[gml_g]

        lst_edges=[]
        for i in mygraph.edges:
            lst_edges.append([i[0], i[1], mygraph.edges[i]['weight']])
        df_corr=pd.DataFrame(lst_edges,columns=['node1','node2','edge_w'])
        df_corr=df_corr.sort_values(by=['edge_w'],ascending=False).reset_index(drop=True)
        df_corr=df_corr.iloc[0:10]
        df_corr['gml_graph']=gml_g
        lst.append(df_corr)
    topk_corr=pd.concat(lst,axis=0)
    return topk_corr  # df


if __name__=="__main__":
    process_set = {101: ''}
    process = [101]
    # 网络结构参数分析

    # 5组分析： 1-总体网络参数；2-中心疾病；3-相关系数最高的top-10共病对；4-聚类分析；5-度分布（节点）；6-相关稀疏分布（边分组）
    # 先完成1-4

    if 100 in process:
        # 必须运行步骤！！！
        # 读取要参与分析的gml文件（按组：病例对照；性别分层；年龄组分层；城乡分层）
        type_='phi'
        participant = ['case', 'ctr', 'gender=1', 'gender=2', 'age_group=0', 'age_group=1', 'age_group=2', 'city','country']

        gml_case=config.pdir + "Project_Cerebrovascular_data/results_tables/92_case_%s_graph_%s.gml"%(type_, str(0))
        gml_ctr=config.pdir + "Project_Cerebrovascular_data/results_tables/94_ctrl_%s_graph_%s.gml"%(type_, str(0))

        dict_gmls=dict()
        for i in participant:
            if i=='case':
                gml_path=gml_case
            elif i=='ctr':
                gml_path=gml_ctr
            else:
                gml_path=config.pdir + "Project_Cerebrovascular_data/results_tables/96_%s_%s_graph_%s.gml" % (i, type_, str(0))

            dict_gmls[i]=nx.read_gml(gml_path)

    if 101 in process:  # 1-总体网络参数
        # 用gephi进行统计即可；不再计算
        print()

    if 102 in process:   # 2-中心疾病
        # 得到介数中心性、加权度中心性、度中心性和邻居平均度的取值
        df_centrality=compute_centrality(dict_gmls)
        df_centrality.to_csv(config.pdir +"Project_Cerebrovascular_data/results_tables/102_centrality.csv")

    if 103 in process:  # 3-相关系数最高的top-10共病对
        topk_corr=get_topk_correlation(dict_gmls)
        topk_corr.to_csv(config.pdir + "Project_Cerebrovascular_data/results_tables/103_topk_corr.csv")

    # if 104 in process:  # 4-聚类分析

