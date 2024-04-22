import pickle
from igraph import *
import networkx as nx
import pandas as pd
import numpy as np
import _10_netw_analy

def construct_network_graph(type,percentile,dic_disease_prevalence_rate,edge_list,list_modularity):
    """功能：构网构图
    输入参数：_percentile 分位数，只画出相关系数在percentile以上的边
    输入参数：_type: type='RR': RR；type='phi':phi；type='CC':CC
    输入参数：_dic_disease_prevalence_rate  流行率字典
    输入参数：_edge_list 边集，应和_type类型对应
    输入参数：_write_file:保存gml图的路径文件名
    输入参数：_list_modularity:每个节点所属社区

    备注：Community_Detection_v2中函数build_Graph()调用了该函数
    """
    if type == 'RR':
        # edge_list_RR的结构：[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的RR值，置信下区间，置信上区间, 同时患两种疾病的人数], ....]

        print(len(edge_list))
        node_set=set()

        quantile_value=pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_RR=[i for i in edge_list if i[4]>=quantile_value]

        for edge in edge_list_RR:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list=sorted(list(node_set))  # 节点名称，排序后

        prevalence_rate=[dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g=Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name']=node_name_list
        g.vs['label']=node_name_list
        g.vs['prevalence']=prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
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
        # g.write(write_file,"gml")
        # plot(g)

    if type=='phi':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的phi值，t值，无意义位，同时患两种病的人数], ....]
        print(len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_phi = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_phi:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率
        print(node_name_list)
        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
        g.add_edges((edge[0], edge[1]) for edge in edge_list_phi)
        phi_list = [0 for j in range(len(edge_list_phi))]
        for edge in edge_list_phi:
            edge_id = g.get_eid(edge[0], edge[1])
            phi_list[edge_id] = edge[4]
        g.es['weight'] = phi_list
        print(summary(g))
        # g.write(write_file, "gml")
        # plot(g)

    if type=='CC':
        # [[边的节点1名称，边的节点2名称，节点1的流行度，节点2的流行度，边的CC值，t值，无意义位，同时患两种病的人数], ....]
        print(len(edge_list))
        node_set = set()

        quantile_value = pd.Series([edge[4] for edge in edge_list]).quantile(percentile)
        edge_list_CC = [i for i in edge_list if i[4] >= quantile_value]

        for edge in edge_list_CC:
            node_set.add(edge[0])
            node_set.add(edge[1])
        node_name_list = sorted(list(node_set))  # 节点名称，排序后
        prevalence_rate = [dic_disease_prevalence_rate[i] for i in node_name_list]  # 节点对应的流行率

        g = Graph()
        g.add_vertices(len(node_name_list))
        g.vs['name'] = node_name_list
        g.vs['label'] = node_name_list
        g.vs['prevalence'] = prevalence_rate
        if list_modularity:
            g.vs['modularity_class'] = list_modularity
        g.add_edges((edge[0], edge[1]) for edge in edge_list_CC)
        CC_list = [0 for j in range(len(edge_list_CC))]
        for edge in edge_list_CC:
            edge_id = g.get_eid(edge[0], edge[1])
            CC_list[edge_id] = edge[4]
        g.es['weight'] = CC_list
        print(summary(g))
        # g.write(write_file, "gml")
    return g


def build_Graph(community_list,edge_list,type,dic_disease_prevalence_rate):
    """
    功能：将社区（聚类）信息加入到gml文件中（每个节点新增一个属性—>所属社区编号）
    :param community_list:a list of lists ，[['I50','I10',...],...]，社区组成的列表
    _path_edge_list:需要读入的edge_list.pkl文件路径
    _type:"CC""RR""phi",需与path_edge_list对应！
    """

    classid=-1
    dic_modularity_class={}
    for i in community_list:
        classid+=1
        for j in i:
            dic_modularity_class[j]=classid
    s_list1 = sorted(dic_modularity_class.items(), key=lambda x: x[0])
    print(s_list1)
    list_modularity=[]
    for i in s_list1:
        list_modularity.append(i[1])
        # print(s_list1)
    print(list_modularity)

    outp_g=construct_network_graph(type,0,dic_disease_prevalence_rate,edge_list,list_modularity)
    # print(len(s_dic1))  93
    return outp_g


def fastunfolding(graph,layer):
    """
    读入图文件，进行社区发现
    输入：layer:取第一层社区发现的结果还是最后一层社区发现的结果？  0或-1
    返回值：community_list：a list of lists ，[['I50','I10',...],...]，社区组成的列表
    返回值：Q：模块度
    返回值：len(community_list)：社区个数
    """
    # graph=Graph.Read_GML(gml_path)
    # print(g.es["weight"] )
    print(len(graph.es["weight"]))
    g = graph.community_multilevel(graph.es["weight"], return_levels=True)
    # print(g)
    community_list=[]
    for c in g:
        print(c)
        Q = c.modularity
        print("模块度为：",Q)
        print()
        print()

    c=g[layer]    # 取第一层社区发现结果
    # print(c)
    Q=c.modularity
    print("模块度为：",Q)
    for node_list in c:
        community=[]
        for node in node_list:
            community.append(graph.vs["label"][node])
        community_list.append(community)
    for i in community_list:
        print(i)
    return community_list,Q,len(community_list)



def Compute_EC(numOfModule,mygraph,straight_flag):
    """
    _numOfModule : a int ,the num of Modularity Class
    _gml_path:需要计算的图文件路径
    该函数用于显示各个社区的大小，计算图中社区特征向量中心度指标，并显示出各社区中特征向量中心度指标排名前五的节点
    """
    # 读入模块化后的网络
    # mygraph = nx.read_gml(gml_path)
    print(len(mygraph.nodes))
    print(len(mygraph.edges))

    cummunity_no=[]
    node_ns=[]
    edge_ns=[]
    disea_lists=[]
    community_graphs=dict()

    for module_id in range(numOfModule):
        cummunity_no.append(module_id)   # 1. 簇ID

        # 社区单独成网
        module0=[]
        for i in mygraph.nodes:
            if mygraph.nodes[i]['modularityclass']==module_id:
                # print(mygraph.nodes[i]['modularityclass'])
                module0.append(mygraph.nodes[i]['name'])

        community0=nx.subgraph(mygraph,module0)

        node_ns.append(len(community0.nodes))   # 2. 节点数量
        edge_ns.append(len(community0.edges))   # 3. 边数量

        disea_lists.append(community0.nodes)  # 4. 疾病（该社区中全部疾病节点）
        community_graphs[straight_flag+str(module_id)]=community0  # 5. 社区子图列表

    for_df=dict()
    for_df['疾病']=disea_lists
    for_df['簇ID']=cummunity_no
    for_df['节点数量']=node_ns
    for_df['边数量']=edge_ns
    df_commu_statistic=pd.DataFrame(for_df)
    df_commu_statistic['分层']=straight_flag
    df_commu_statistic['社区个数'] = num_of_community

    df_centrality_straight=_10_netw_analy.compute_centrality(community_graphs)      # 社区中各个节点的中心性df
    return df_centrality_straight, df_commu_statistic



if __name__=="__main__":
    process_set = {111: ''}
    process = [111]

    if 110 in process:
        # 必须运行步骤！！！
        # 读取要参与分析的gml文件（按组：病例对照；性别分层；年龄组分层；城乡分层）
        type_='phi'
        prev_threshold = 0.01
        participant = ['case', 'ctr', 'gender=1', 'gender=2', 'age_group=0', 'age_group=1', 'age_group=2', 'city','country']

        gml_case=config.pdir + "Project_Cerebrovascular_data/results_tables/92_case_%s_graph_%s.gml"%(type_, str(0))
        gml_ctr=config.pdir + "Project_Cerebrovascular_data/results_tables/94_ctrl_%s_graph_%s.gml"%(type_, str(0))

        dict_gmls=dict()
        dict_prev=dict()
        dict_edge_list=dict()
        for i in participant:
            if i=='case':
                gml_path=gml_case
                prev_path= config.pdir + "Project_Cerebrovascular_data/median_data/case_91_dic_disease_prevalence_rate_%s.pkl" % (str(prev_threshold))
                edge_list_path= config.pdir + "Project_Cerebrovascular_data/median_data/case_91_edge_list_%s_%s.pkl" % ( type_, str(prev_threshold))
            elif i=='ctr':
                gml_path=gml_ctr
                prev_path=config.pdir +"Project_Cerebrovascular_data/median_data/ctrl_93_dic_disease_prevalence_rate_%s.pkl" % ( str(prev_threshold))
                edge_list_path=config.pdir + "Project_Cerebrovascular_data/median_data/ctrl_93_edge_list_%s_%s.pk" % ( type_, str(prev_threshold))
            else:
                gml_path=config.pdir + "Project_Cerebrovascular_data/results_tables/96_%s_%s_graph_%s.gml" % (i, type_, str(0))
                prev_path = config.pdir + "Project_Cerebrovascular_data/median_data/95_%s_dic_disease_prevalence_rate_%s.pkl" % (i, str(prev_threshold))
                edge_list_path = config.pdir + "Project_Cerebrovascular_data/median_data/95_%s_edge_list_%s_%s.pkl" % (i, type_, str(prev_threshold))

            dict_gmls[i]=nx.read_gml(gml_path)
            dict_prev[i]=pickle.load(open(prev_path, 'rb'))
            dict_edge_list[i] = pickle.load(open(edge_list_path, 'rb'))

    if 111 in process:
        for i in dict_gmls:
            graph=dict_gmls[i]
            dic_disease_prevalence_rate=dict_prev[i]
            edge_list=dict_edge_list[i]

            layer = -1
            type_='phi'
            outp_gml_path = config.pdir + "Project_Cerebrovascular_data/results_tables/111_CD_%s_%s_graph_%s_layer%s.gml" % (i, type_, str(0), str(layer))

            # 社区发现算法
            community_list, Q, num_of_community= fastunfolding(graph,layer)  # 取第一层/最后一层社区发现结果
            #重构图，将社区信息加入gml文件
            outp_g=build_Graph(community_list,edge_list,type_,dic_disease_prevalence_rate) #存储为gml文件，包含社区
            outp_g.write(outp_gml_path, "gml")

    if 112 in process:
        participant = ['case', 'ctr', 'gender=1', 'gender=2', 'age_group=0', 'age_group=1', 'age_group=2', 'city', 'country']
        layer = -1
        type_ = 'phi'

        lst_centri_dfs=[]
        lst_commu_statis_dfs=[]
        for i in participant:
            outp_gml_path = config.pdir + "Project_Cerebrovascular_data/results_tables/111_CD_%s_%s_graph_%s_layer%s.gml" % (i, type_, str(0), str(layer))
            outp_g=nx.read_gml(outp_gml_path)
            # 计算社区个数
            num_of_community=np.unique(np.array(list(dict(outp_g.nodes.data('modularity_class')).values()))).shape[0]
            # 统计聚类相关的网络参数——表格
            df_centrality_straight, df_commu_statistic=Compute_EC(num_of_community,outp_g,i)
            # 合并为一个表格, 并保存
            lst_centri_dfs.append(df_centrality_straight)
            lst_commu_statis_dfs.append(df_commu_statistic)
        centri_df=pd.concat(lst_centri_dfs,axis=0)
        commu_statis=pd.concat(lst_commu_statis_dfs,axis=0)
        centri_df.to_csv(config.pdir + "Project_Cerebrovascular_data/results_tables/112_communitys_centrality.csv")
        commu_statis.to_csv(config.pdir + "Project_Cerebrovascular_data/results_tables/112_commulitys_statistics.csv")


    # 社区子图可视化： 用gephi可视化111的gml文件

    # 分层网络图可视化——基于pyecharts












