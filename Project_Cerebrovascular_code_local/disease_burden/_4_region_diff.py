import time

import numpy as np
import os
import sys
import cx_Oracle
# import seaborn as sns
import config
# from bokeh.palettes import brewer,Spectral,YlGn
# import bokeh
# import matplotlib.pyplot as plt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import pandas as pd
from pub_funs import *
import pickle
import scipy.sparse as sp
# from pyecharts.charts import Map,Geo
# from pyecharts import options as opts
# from pyecharts.globals import ChartType, SymbolType
# from pyecharts.render import make_snapshot
# from snapshot_selenium import snapshot


def check_geo_info(sichuan_city_country,dic_code_region,sichuan_city_country_echarts_path,flag):
    """
    flag='check' or 'modify'
    检查code-region字典和地图信息是否完全一致
    :return:
    """
    if flag=='check':
        region_echarts=[]
        for i in sichuan_city_country:
            region_echarts.extend(sichuan_city_country[i])
        region_dict= set(dic_code_region.values())
        a=(set(region_echarts)|region_dict)-(set(region_echarts)&region_dict)

        print('sichuan_city_country',sichuan_city_country)
        print('dic_code_region',dic_code_region)
        print("不一致的地名：",a)
        print('len(region_echarts)',len(region_echarts))
        print('len(region_dict)',len(region_dict))
        return a
    elif flag=='modify':
        # 检查更改字典：
        # 宜宾县更名为叙州区； 射洪县更名为射洪市； 罗江县 > 罗江区 ； 隆昌县 > 隆昌市;  新津县 > 新津区
        # set(['射洪县','射洪市','罗江县','罗江区','隆昌县','隆昌市','宜宾县','叙州区','新津县' , '新津区'])

        # 1. 修改地图地理信息map_js/si4_chuan1_countries_yangping.js     2. 修改sichuan_city_country的提取文件
        sichuan_city_country = pickle.load(open(sichuan_city_country_echarts_path, 'rb'))

        modi_dict={'射洪县':'射洪市','罗江县':'罗江区','隆昌县':'隆昌市','宜宾县':'叙州区','新津县' : '新津区'}

        modified_sichuan_city_country=dict()
        for i in sichuan_city_country:
            list_tmp=[]
            for j in sichuan_city_country[i]:
                if j not in modi_dict:
                    list_tmp.append(j)
                else:
                    list_tmp.append(modi_dict[j])
            modified_sichuan_city_country[i]=list_tmp
        return modified_sichuan_city_country



def get_hotMAP_DATA(df_celebro, sichuan_city_country, dic_code_region, flag):
    """
    获取用于画183个区县地图热力图的数据
    :param df_celebro:
    :param sichuan_city_country:
    :param dic_code_region:
    :param flag: '2015-2020', '2015'
    :return:
    """
    # MAP_DATA示例       MAP_DATA = [["青羊区", 20057.34], ["成华区", 1547007.48], ['井研县', 111992.6], ]
    if flag in ['2015','2016','2017','2018','2019','2020']:
        year_s = flag + '-01-01'
        year_f = flag + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]

    elif flag=='2015-2020':
        # 总的分布情况
        df_tmp=df_celebro
    # print('dic_code_region',dic_code_region)
    region_count=df_tmp[config.XZZ_XZQH2].groupby(df_tmp[config.XZZ_XZQH2]).count()  # return a Series
    # print(region_count)

    region_df=pd.DataFrame(zip(region_count.index,region_count.values),columns=['region_code','count'])
    region_df['region_name']=region_df['region_code'].apply(lambda x: dic_code_region[x] if x in dic_code_region else 'none')
    print('region_df[region_df[region_name]!=none].shape[0]',region_df[region_df['region_name']!='none'].shape[0])
    region_df=region_df[region_df['region_name']!='none']

    MAP_DATA=list(zip(region_df['region_name'], region_df['count']))

    MAP_DATA=[list(i) for i in MAP_DATA]
    print(MAP_DATA)
    return MAP_DATA


def hotmap_sicuhan_counties(year,MAP_DATA,min_value,max_value):
    """
    作183区县的地图热力图
    :param MAP_DATA:
    :return:
    """
    map=(Map(init_opts=opts.InitOpts(width="700px", height="700px"))
        .add(
            series_name="",
            maptype="四川-区县",    # E:/python37/Lib/site-packages/echarts_china_cities_pypkg/resources/echarts-china-cities-js/
            data_pair=MAP_DATA,
            is_map_symbol_show=False,
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 不显示标签
        .set_global_opts(

            title_opts=opts.TitleOpts(
                title="四川省 %s年"%(year),
                title_textstyle_opts=opts.TextStyleOpts(font_size=25)
                # subtitle="人口密度数据来自Wikipedia"
            ),

            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{b}<br/>{c} (p / km2)",

            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=min_value,
                max_=max_value,
                range_text=["住院人次", ""],
                is_piecewise=True,
                precision=0,
                split_number=4,
                # is_calculable=True,
                range_color=["lightskyblue", "white", "red"],
                textstyle_opts=opts.TextStyleOpts(font_size = 15)
                # range_color=[bokeh.palettes.YlOrRd[9][8-i] for i in range(9)],
                # range_color=[bokeh.palettes.RdBu[9][i] for i in range(9)],
            ),
        )
         # .set_series_opts(label_opts=opts.LabelOpts(font_size = 10),# 标签字体调整)
    )
    map.render("map_visual_html/hotmap_sichuan_183counties%s.html" % (year))
    # 输出保存为图片
    make_snapshot(snapshot, map.render(), "maps_png/hotmap_183counties%s.png"% (year))


def generate_table43(df_celebro, save_file,region_code):
    """
    表：2015-2020四川省脑血管住院患者城乡分布、经济区分布
    :param df_celebro:
    :param save_file:
    :return:
    """
    city_re=set(region_code[region_code['城乡划分']=='城市地区']['code6'].values)
    print(city_re)
    country_re = set(region_code[region_code['城乡划分'] == '农村地区']['code6'].values)

    set_cd=set(region_code[region_code['五大经济区']=='成都经济区']['code6'].values)
    set_cdb = set(region_code[region_code['五大经济区'] == '川东北经济区']['code6'].values)
    set_cn = set(region_code[region_code['五大经济区'] == '川南经济区']['code6'].values)
    set_cxb = set(region_code[region_code['五大经济区'] == '川西北经济区']['code6'].values)
    set_px = set(region_code[region_code['五大经济区'] == '攀西经济区']['code6'].values)

    # 先按照[sfzh, cy_date]排序； 再[sfzh, cy_year]去重，keep='first';  # df=df_celebro中每年的第一条
    df=df_celebro.sort_values(by=[config.SFZH,config.CY_DATE],ascending=True).reset_index(drop=True)
    df['cy_year']=df[config.CY_DATE].apply(lambda x:x.year)
    print('cy_year',df['cy_year'].values)
    print('5, df.shape',df.shape)
    df=df.drop_duplicates(subset=[config.SFZH,'cy_year'],keep='first')
    df=df[[config.XZZ_XZQH2,config.SFZH,config.CY_DATE]]
    print('6, df.shape', df.shape)

    print('type',list(df[config.XZZ_XZQH2].drop_duplicates().values))

    df['is_city']=df[config.XZZ_XZQH2].apply(lambda x:1 if len(set([x])&city_re)>0 else (0 if len(set([x])&country_re)>0 else 2))
    df['economic region'] = df[config.XZZ_XZQH2].apply(lambda x: 'cd' if len(set([x]) & set_cd) > 0 else
                                                                                                        ('cdb' if len(set([x]) & set_cdb) > 0 else
                                                                                                          ('cn' if len( set([x]) & set_cn) > 0 else
                                                                                                           ('cxb' if len(set([x]) & set_cxb) > 0 else
                                                                                                           ('px' if len(set([x]) & set_px) > 0 else -1)))))
     # 城乡分层; 经济区分层
    dic_gender_popu = dict()
    dic_eco=dict()
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_tmp = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & (
                    df[config.CY_DATE] <= pd.to_datetime(year_f))]
        print('! df_tmp.shape',df_tmp.shape)
        df_xb = df_tmp[[config.SFZH,'is_city']].drop_duplicates()
        print('! df_xb.shape', df_xb.shape)

        city = df_xb[df_xb['is_city'] == 1].shape[0]
        country = df_xb[df_xb['is_city'] ==0].shape[0]
        dic_gender_popu[i] = [city, country]

        cd=df_tmp[df_tmp['economic region']=='cd'].shape[0]
        cdb = df_tmp[df_tmp['economic region'] == 'cdb'].shape[0]
        cn = df_tmp[df_tmp['economic region'] == 'cn'].shape[0]
        cxb = df_tmp[df_tmp['economic region'] == 'cxb'].shape[0]
        px = df_tmp[df_tmp['economic region'] == 'px'].shape[0]
        dic_eco[i]=[cd,cdb,cn,cxb,px]

    print(dic_gender_popu)
    print('dic_eco',dic_eco)

    padding=[-1 for i in range(len(dic_gender_popu))]
    dic_table32 =dict()

    # 性别人数和比例
    nda_gender = np.array(list(dic_gender_popu.values()))
    male = nda_gender[:, 0]
    female = nda_gender[:, 1]
    sum_=male+female
    male_r=male/sum_*100
    female_r=female/sum_*100

    dic_table32['城乡分布[例(%)]']=padding
    dic_table32['城市 数值']=male
    dic_table32['农村 数值']=female
    dic_table32['城市 比例'] = male_r
    dic_table32['农村 比例'] = female_r

    # 经济区人数和比例
    nda_gender = np.array(list(dic_eco.values()))   # 行：年份  ； 列： 经济区人数

    sum_col = nda_gender.sum(axis=0)
    rates_nda = nda_gender / sum_col * 100

    dic_table32['五大经济区[例(%)]'] = padding
    for index,eco_r in enumerate(['成都经济区','川东北经济区','川南经济区','川西北经济区','攀西经济区']):
        dic_table32['%s 数值'%(eco_r)] = nda_gender[:, index]
        dic_table32['%s 比例' % (eco_r)] = rates_nda[:, index]


    table32=pd.DataFrame(dic_table32.values(),columns= ['2015', '2016', '2017', '2018', '2019', '2020'])
    table32['项目']=list(dic_table32.keys())
    print(table32)
    table32=table32.round(2)
    table32.to_csv(save_file)
    return table32


def data_popu_flow(df, dic_code_region):
    """ 构建一个矩阵： 行为流出，列为流入 """
    flow_mtxs=[]
    dic_code_row = dict(zip(list(dic_code_region.keys()), range(len(dic_code_region))))
    sum_=0
    for i in ['2015', '2016', '2017', '2018', '2019', '2020']:
        year_s = i + '-01-01'
        year_f = i + '-12-31'
        df_year = df[(df[config.CY_DATE] >= pd.to_datetime(year_s)) & ( df[config.CY_DATE] <= pd.to_datetime(year_f))][[config.XZZ_XZQH2,config.DEPT_ADDRESSCODE2]]
        df_year=list(zip(df_year[config.XZZ_XZQH2],df_year[config.DEPT_ADDRESSCODE2]))

        mt_flow = np.zeros((len(dic_code_region), len(dic_code_region)))

        for j in df_year:
            # if j[0] in dic_code_row and j[1] in dic_code_row:
            mt_flow[dic_code_row[j[0]]][dic_code_row[j[1]]]+=1
        flow_mtxs.append(mt_flow)
        sum_+=mt_flow.sum()

    print(i, '--------------------------------')
    print('sum----------------',sum_)
    print(mt_flow)

    return flow_mtxs,dic_code_row



def data_statistic_intercountry_popu_rate(flow_mtxs,dic_code_row, dic_code_country,dic_code_citycountry,dic_country_city,dic_regioncode_popu,dic_code_economy):
    """"""
    mt_6years = np.zeros((len(dic_code_row), len(dic_code_row)))
    for i in flow_mtxs:
        mt_6years+=i
    print('mt_6years.sum()',mt_6years.sum())

    lst_tmp=[]
    for i in range(mt_6years.shape[0]):
        lst_tmp.append(mt_6years[i][i])    # 对角线表示：不跨市州住院人次
    print(lst_tmp)
    print('sum(list(dic_regioncode_popu.values()))',sum(list(dic_regioncode_popu.values())))
    print('sum(lst_tmp)',sum(lst_tmp))

    refresh_is_city=[]
    city = []
    country = []
    total_cere_popu=[]
    economy=[]
    for i in dic_code_row:
        refresh_is_city.append(dic_code_citycountry[i])
        city.append(dic_country_city[i])
        country.append(dic_code_country[i])
        total_cere_popu.append(dic_regioncode_popu[i])
        economy.append(dic_code_economy[i])
    print('dic_regioncode_popu',dic_regioncode_popu)
    # print('total_cere_popu',total_cere_popu)
    print('dict(zip(country,total_cere_popu))',dict(zip(dic_code_row.keys(),total_cere_popu)))

    # 流出人数/比例
    flow_out = mt_6years.sum(axis=1)  # 行和
    # print('flow_out', flow_out)
    print('flow_out.sum()',flow_out.sum())
    flow_out-=lst_tmp
    flow_out_rate=flow_out/total_cere_popu*100
    # 流入人数/比例
    flow_in=mt_6years.sum(axis=0)  # 列和
    # print('flow_in',list(flow_in))
    print('flow_in.sum()', flow_in.sum())
    flow_in-=lst_tmp
    flow_in_rate = flow_in / total_cere_popu*100

    # 整理数据
    df_flow_popu=pd.DataFrame(zip(list(dic_code_row.keys()),country,flow_out,flow_in,refresh_is_city,city,flow_out_rate,flow_in_rate,economy),columns=['区县','区县名称','流出人次','流入人次','城乡划分','地级市','流出比例','流入比例','五大经济区'])
    df_flow_popu=df_flow_popu.round(1)

    print('df_flow_popu',df_flow_popu.head(20))
    return df_flow_popu


def data_net_flow_city(flow_mtxs,dic_code_row,region_code):
    """"""
    # 构建一个分配矩阵s ,维度为：21*183
    row=region_code['市州'].drop_duplicates().values
    dic_code_city=dict(zip(region_code['code6'],region_code['市州']))
    dic_row=dict(zip(row,range(row.shape[0])))
    dic_row_city=dict(zip(range(row.shape[0]),row))

    s=np.zeros((len(dic_row),len(dic_code_row)))
    for i in dic_code_city:
        s[dic_row[dic_code_city[i]]][dic_code_row[i]]=1
    s=sp.coo_matrix(s)  # 分配矩阵
    # print('s.todense()',list(s.todense()))

    net_flows=[]
    city_popus=[]
    dic_city_flows=[]
    sum_=0
    for mtx in flow_mtxs:
        mtx = sp.coo_matrix(mtx)
        city_city_popu=s*mtx*(s.T)

        # 先取对角线 —— 各城市的本地就医的患病人次数
        city_popu=city_city_popu.diagonal()
        dict_city_popu=dict(zip(dic_row.keys(),city_popu))
        sum_+=city_city_popu.sum()


        # 算净流量
        net_flow_intercity=city_city_popu-city_city_popu.T
        # print('net_flow_intercity',net_flow_intercity)
        # print('type(net_flow_intercity)',type(net_flow_intercity))
        net_flow_intercity=net_flow_intercity.tocoo()
        tmp=pd.DataFrame(list(zip(net_flow_intercity.row,net_flow_intercity.col,net_flow_intercity.data)),columns=['row','col','net_flow'])
        tmp=tmp[tmp['net_flow']>=0]
        tmp['source city']=tmp['row'].apply(lambda x: dic_row_city[x])
        tmp['terminal city']=tmp['col'].apply(lambda x: dic_row_city[x])

        net_flows.append(tmp)
        city_popus.append(dict_city_popu)  # 本地就医人数

        # print('123-----------------------')
        print(tmp)
        print(dict_city_popu)
    print('citysum',sum_)

    # 各个城市的净流量
    city_flow=np.array(net_flow_intercity.sum(axis=0)).reshape(-1)
    print('city_flow',city_flow)
    dic_city_flow=dict(zip(dic_row.keys(),city_flow))
    print('dic_city_flow',dic_city_flow)
    dic_city_flows.append(dic_city_flow)

    return net_flows,city_popus,dic_city_flows   # 六年分层的净流量、本地就医人数


def data_net_flow_country(flow_mtxs,dic_code_row,region_code):
    """"""
    dic_code_country=dict(zip(region_code['code6'],region_code['县市区']))
    dic_tmp=dict(zip(dic_code_row.values(),dic_code_row.keys()))
    dic_row_country=dict()
    for i in dic_tmp:
        dic_row_country[i]=dic_code_country[dic_tmp[i]]
    dic_country_row=dict(zip(dic_row_country.values(),dic_row_country.keys()))

    net_flows=[]
    city_popus=[]
    dic_city_flows=[]
    sum_=0
    for mtx in flow_mtxs:
        mtx = sp.coo_matrix(mtx)
        # city_city_popu=s*mtx*(s.T)

        # 先取对角线 —— 各城市的本地就医的患病人次数
        city_popu=mtx.diagonal()
        dict_city_popu=dict(zip(dic_country_row.keys(),city_popu))
        sum_+=mtx.sum()

        # 算净流量
        net_flow_intercity=mtx-mtx.T
        net_flow_intercity=net_flow_intercity.tocoo()
        tmp=pd.DataFrame(list(zip(net_flow_intercity.row,net_flow_intercity.col,net_flow_intercity.data)),columns=['row','col','net_flow'])
        tmp=tmp[tmp['net_flow']>=0]
        tmp['source city']=tmp['row'].apply(lambda x: dic_row_country[x])
        tmp['terminal city']=tmp['col'].apply(lambda x: dic_row_country[x])

        net_flows.append(tmp)
        city_popus.append(dict_city_popu)  # 本地就医人数

        # print('123-----------------------')
        print(tmp)
        print(dict_city_popu)
        print('citysum',sum_)

        # 各个城市的净流量
        city_flow=np.array(net_flow_intercity.sum(axis=0)).reshape(-1)
        print('city_flow',city_flow)
        dic_city_flow=dict(zip(dic_country_row.keys(),city_flow))
        print('dic_city_flow',dic_city_flow)

        dic_city_flows.append(dic_city_flow)

    return net_flows,city_popus,dic_city_flows   # 六年分层的净流量、本地就医人数


def interCountries_flow(net_flows,dic_country_flows, flag):
    """
    net_flow: columns=['row','col','net_flow','source city','terminal city']
    作图 - 跨市州住院净流量 - 有向流向图

    :return:
    """

    test = True
    # flag = 'city'

    if flag=='city':
        maptype="四川"
    elif flag=='country':
        maptype = "四川-区县"
    for index, year in enumerate(['2019']):   #['2015', '2016', '2017', '2018', '2019', '2020']

        region_count=list(zip(dic_country_flows[index].keys(),dic_country_flows[index].values()))
        in_rgion=[]
        out_rgion=[]
        for t in region_count:
            if t[1]>=0:
                in_rgion.append(t)
            else:
                out_rgion.append((t[0],-1*t[1]))
        # 排序  (升序)
        in_rgion=pd.DataFrame(in_rgion,columns=['region','popu'])
        out_rgion=pd.DataFrame(out_rgion,columns=['region','popu'])
        in_rgion=in_rgion.sort_values(by=["popu"]).reset_index(drop=True)
        out_rgion = out_rgion.sort_values(by=["popu"]).reset_index(drop=True)
        in_rgion=list(zip(in_rgion['region'],in_rgion['popu']))
        out_rgion = list(zip(out_rgion['region'], out_rgion['popu']))
        print('out_rgion', out_rgion)


        net_flow=net_flows[index]
        # 排序 ——升序
        net_flow=net_flow.sort_values(by=["net_flow"]).reset_index(drop=True)
        flow_strt_end=list(zip(net_flow['source city'],net_flow['terminal city']))


        print('flow_strt_end',flow_strt_end)
        if test == True:
            region_count = [("成都", 55), ("乐山", 66), ("", 77), ("南充", 88), ("内江", 90)]
            in_rgion=[("成都", 55), ("乐山", 66), ("绵阳", 77), ("南充", 88), ("内江", 90), ("攀枝花", 66), ("泸州", 77), ("德阳", 88), ("广元", 90)]
            out_rgion=[("宜宾", 55), ("达州", 66), ("雅安", 77), ("巴中", 88), ("凉山彝族自治州", 90), ("阿坝藏族羌族自治州", 66), ("凉山彝族自治州", 77)]
            flow_strt_end = [("乐山", "成都"), ("内江", "成都"), ("绵阳", "成都"), ("南充", "成都"),("乐山", "绵阳"), ("内江", "泸州"), ("绵阳", "广元"), ("南充", "阿坝藏族羌族自治州")]

        # c=['#b2bec3','#636e72','#2d3436','#b2bec3','#636e72','#2d3436',]
        # c_line=['#FFC312','#F79F1F','#EE5A24','#EA2027']
        c = [ '#636e72','#2d3436', 'black', '#636e72','#2d3436', 'black']
        c_line=[YlGn[5][1],YlGn[5][2],YlGn[5][3],YlGn[5][4]]

        geo= (
        Geo(init_opts=opts.InitOpts(width="2700px", height="2700px"))
        .add_schema(maptype=maptype)   # "四川-区县"  or  "四川"

        # 流向
        .add(
            "2.1",
            flow_strt_end[0:int(len(flow_strt_end)/4)],
            color=c_line[0],
            type_=ChartType.LINES,
            # color='#16c79a',  # 标记颜色
            # itemstyle_opts=opts.ItemStyleOpts(color=c_line[0]),
            linestyle_opts=opts.LineStyleOpts(curve=0.2,width=2),
            symbol_size=15,
            label_opts=opts.LabelOpts(is_show=False),
        )

        .add(
            "2.2",
            flow_strt_end[int(len(flow_strt_end)/4):int(2*len(flow_strt_end)/4)],
            color=c_line[1],
            type_=ChartType.LINES,
            # color='#f6c065',

            # itemstyle_opts=opts.ItemStyleOpts(color=c_line[1]),
            linestyle_opts=opts.LineStyleOpts(curve=0.2,width=5),
            symbol_size=25,
            label_opts=opts.LabelOpts(is_show=False)
        )
        .add(
            "2.3",
            flow_strt_end[int(2*len(flow_strt_end) / 4):int(3 * len(flow_strt_end) / 4)],
            color=c_line[2],
            type_=ChartType.LINES,
            # color='#16c79a',  # 标记颜色

            # itemstyle_opts=opts.ItemStyleOpts(color=c_line[2]),
            linestyle_opts=opts.LineStyleOpts(curve=0.2, width=10),
            symbol_size=60,
            label_opts=opts.LabelOpts(is_show=False),
        )

        .add(
            "2.4",
            flow_strt_end[int(3*len(flow_strt_end) / 4):len(flow_strt_end)+1],
            color=c_line[3],
            type_=ChartType.LINES,
            # color='#f6c065',
            # itemstyle_opts=opts.ItemStyleOpts(color=c_line[3]),
            linestyle_opts=opts.LineStyleOpts(curve=0.2, width=15),
            symbol_size=70,
            label_opts=opts.LabelOpts(is_show=False)
        )

            .add(
            "1.1.1-流入型",
            in_rgion[0:int(len(in_rgion) / 3)],
            type_=ChartType.SCATTER,
            # color=c[0],
            itemstyle_opts=opts.ItemStyleOpts(color=c[0]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=15,
            symbol='triangle',
        )
            .add(
            "1.1.2-流入型",
            in_rgion[int(len(in_rgion) / 3):int(2 * len(in_rgion) / 3)],
            type_=ChartType.SCATTER,
            # color=c[1],
            itemstyle_opts=opts.ItemStyleOpts(color=c[1]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=35,
            symbol='triangle',
        )
            .add(
            "1.1.3-流入型",
            in_rgion[int(2 * len(in_rgion) / 3):int(len(in_rgion) + 1)],
            type_=ChartType.SCATTER,
            # color=c[2],
            itemstyle_opts=opts.ItemStyleOpts(color=c[2]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=60,
            symbol='triangle',
        )
            # 流出型
            .add(
            "1.2.1-流出型",
            out_rgion[0:int(len(out_rgion) / 3)],
            type_=ChartType.SCATTER,
            # color=c[3],
            itemstyle_opts=opts.ItemStyleOpts(color=c[3]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=25,
            # symbol='triangle',
        )
            .add(
            "1.2.2-流出型",
            out_rgion[int(len(out_rgion) / 3):int(2 * len(out_rgion) / 3)],
            type_=ChartType.SCATTER,
            # color=c[4],
            itemstyle_opts=opts.ItemStyleOpts(color=c[4]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=35,
            # symbol='triangle',
        )
            .add(
            "1.2.3-流出型",
            out_rgion[int(2 * len(out_rgion) / 3):int(len(out_rgion) + 1)],
            type_=ChartType.SCATTER,
            # color=c[5],
            itemstyle_opts=opts.ItemStyleOpts(color=c[5]),
            label_opts=opts.LabelOpts(formatter="{b}", font_size=40),
            symbol_size=50,
            # symbol='triangle',
        )

        # .set_series_opts(label_opts=opts.LabelOpts(is_show=True))
        .set_global_opts(title_opts=opts.TitleOpts(title=""),legend_opts=opts.LegendOpts(pos_right='right',pos_bottom='bottom',textstyle_opts=opts.TextStyleOpts(font_size=50)))
    )

        # geo.render("geo_lines_%s.html"%(year))
        make_snapshot(snapshot, geo.render(), "geos_png/flowmap%s.png" % (year))



def compute_total_num_redion(df_celebro,dic_code_region):
    cy_xzz_count = df_celebro[[config.SFZH, config.XZZ_XZQH2]].groupby([config.XZZ_XZQH2]).count()
    print(cy_xzz_count)

    c = pd.DataFrame(list(zip(list(cy_xzz_count.index), cy_xzz_count[config.SFZH].values)), columns=[config.XZZ_XZQH2, 'count'])
    print('c',c)

    dic_regioncode_popu=dict()
    for i in dic_code_region:
        if i in c[config.XZZ_XZQH2].values:
            dic_regioncode_popu[i]=c[c[config.XZZ_XZQH2]==i]['count'].values[0]
        else:
            dic_regioncode_popu[i]=0

    print('dic_regioncode_popu',dic_regioncode_popu)
    print(len(dic_regioncode_popu))
    return dic_regioncode_popu


def compute_total_num_hosptial_redion(df_celebro,dic_code_region):
    cy_xzz_count = df_celebro[[config.SFZH, config.DEPT_ADDRESSCODE2]].groupby([config.DEPT_ADDRESSCODE2]).count()
    print(cy_xzz_count)

    c = pd.DataFrame(list(zip(list(cy_xzz_count.index), cy_xzz_count[config.SFZH].values)), columns=[config.DEPT_ADDRESSCODE2, 'count'])
    print('c',c)

    dic_regioncode_popu=dict()
    for i in dic_code_region:
        if i in c[config.DEPT_ADDRESSCODE2].values:
            dic_regioncode_popu[i]=c[c[config.DEPT_ADDRESSCODE2]==i]['count'].values[0]
        else:
            dic_regioncode_popu[i]=0

    print('dic_regioncode_popu',dic_regioncode_popu)
    print(len(dic_regioncode_popu))
    print('hosptial sum((dic_regioncode_popu.values()))',sum((dic_regioncode_popu.values())))
    # return dic_regioncode_popu


def plot_45_bubble(df_flow_popu,flag):

    #  区县  区县名称      流出人次      流入人次  城乡划分 地级市  流出比例  流入比例

    if flag=='城乡划分':
        df_flow_popu['index']=list(df_flow_popu.index)

        # 城市
        df_city = df_flow_popu[df_flow_popu['城乡划分'] == '城市地区'].reset_index(drop=True)
        df_city['index']=list(df_city.index)
        for i in list(zip(df_city["index"], df_city["流出人次"], df_city["流入人次"])):
            plt.plot([i[0], i[0]], [i[1], i[2]], color='black')

        plt.scatter(df_city["index"], df_city["流出人次"], s=80, alpha=0.9,c=sns.color_palette('Paired')[1], marker='D',label='流出')
        plt.scatter(df_city["index"], df_city["流入人次"], s=80, alpha=0.9, c=sns.color_palette('Paired')[5], marker='D',label='流入')

        orig_x=[i for i in range(df_city.shape[0])]
        new_x=list(df_city['区县名称'].values)
        plt.xticks(orig_x, new_x)

        plt.rcParams["figure.edgecolor"] = "none"
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.xlabel("城市地区", fontsize=20)
        plt.ylabel("跨区县住院人次", fontsize=20)
        plt.tick_params(labelsize=15)
        plt.legend(fontsize=20)
        plt.xticks(rotation=90)
        plt.show()

        # 农村
        df_city = df_flow_popu[df_flow_popu['城乡划分'] == '农村地区'].reset_index(drop=True)
        df_city['index'] = list(df_city.index)
        for i in list(zip(df_city["index"], df_city["流出人次"], df_city["流入人次"])):
            plt.plot([i[0], i[0]], [i[1], i[2]], color='black')

        plt.scatter(df_city["index"], df_city["流出人次"], s=70, alpha=0.9, c=sns.color_palette('Paired')[1],
                    label='流出')
        plt.scatter(df_city["index"], df_city["流入人次"], s=70, alpha=0.9, c=sns.color_palette('Paired')[5],
                    label='流入')

        orig_x = [i for i in range(df_city.shape[0])]
        new_x = list(df_city['区县名称'].values)
        plt.xticks(orig_x, new_x)

        plt.rcParams["figure.edgecolor"] = "none"
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.xlabel("农村地区", fontsize=20)
        plt.ylabel("跨区县住院人次", fontsize=20)
        plt.tick_params(labelsize=8)
        ax = plt.gca()
        ax.set_yticklabels(fontsize=15)

        plt.legend(fontsize=20)
        plt.xticks(rotation=90)
        plt.show()

    elif flag == '五大经济区':
        for e in df_flow_popu['五大经济区'].drop_duplicates().values:
            df_flow_popu['index'] = list(df_flow_popu.index)

            # 城市
            df_city = df_flow_popu[df_flow_popu[flag] == e].reset_index(drop=True)
            df_city['index'] = list(df_city.index)
            for i in list(zip(df_city["index"], df_city["流出人次"], df_city["流入人次"])):
                plt.plot([i[0], i[0]], [i[1], i[2]], color='black')

            plt.scatter(df_city["index"], df_city["流出人次"], s=80, alpha=0.9, c=sns.color_palette('Paired')[1], marker='D',
                        label='流出')
            plt.scatter(df_city["index"], df_city["流入人次"], s=80, alpha=0.9, c=sns.color_palette('Paired')[5], marker='D',
                        label='流入')

            orig_x = [i for i in range(df_city.shape[0])]
            new_x = list(df_city['区县名称'].values)
            plt.xticks(orig_x, new_x)

            plt.rcParams["figure.edgecolor"] = "none"
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.xlabel(e, fontsize=20)
            plt.ylabel("跨区县住院人次", fontsize=20)
            plt.tick_params(labelsize=15)
            plt.legend(fontsize=20)
            plt.xticks(rotation=90)
            plt.show()


def plot_46_bucket(df_flow_popu, flag='五大经济区'):
    df_flow_popu['index'] = list(df_flow_popu.index)
    if flag == '五大经济区':
        for e in df_flow_popu['五大经济区'].drop_duplicates().values:
            df_city = df_flow_popu[df_flow_popu[flag] == e].reset_index(drop=True)
            df_city['index'] = list(df_city.index)

            total_width, n = 0.6, 2
            width = total_width / n

            plt.rcParams["figure.edgecolor"] = "none"
            plt.rcParams['font.sans-serif'] = ['SimHei']

            plt.bar(df_city["index"], df_city["流出比例"], width=width, alpha=0.9, color=sns.color_palette('Paired')[1],  label='流出')
            for a, b in zip(df_city["index"], df_city["流出比例"]):  # 柱子上的数字显示
                plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=10)

            x = list(range(df_city["index"].shape[0]))
            for i in range(len(x)):
                x[i] = x[i] + width

            plt.bar(x, df_city["流入比例"],alpha=0.9, width=width,color=sns.color_palette('Paired')[5],  label='流入')
            for a, b in zip(x,  df_city["流入比例"]):  # 柱子上的数字显示
                plt.text(a, b, '%.1f' % b, ha='center', va='bottom', fontsize=10)

            orig_x = [i for i in range(df_city.shape[0])]
            new_x = list(df_city['区县名称'].values)
            plt.xticks(orig_x, new_x)

            plt.xlabel(e, fontsize=20)
            plt.ylabel("跨区县住院比例(%)", fontsize=20)
            plt.tick_params(labelsize=15)
            plt.legend(fontsize=20,loc='upper right')
            plt.xticks(rotation=90)
            plt.show()


def get_hotMAP_DATA_thousand(df_celebro, flag):
    """
    获取用于画183个区县前人口住院率地图热力图的数据
    :param df_celebro:
    :param sichuan_city_country:
    :param dic_code_region:
    :param flag: '2015-2020', '2015'
    :return:
    """
    # MAP_DATA示例       MAP_DATA = [["青羊区", 20057.34], ["成华区", 1547007.48], ['井研县', 111992.6], ]
    if flag in ['2015','2016','2017','2018','2019','2020']:
        year_s = flag + '-01-01'
        year_f = flag + '-12-31'
        df_tmp = df_celebro[(df_celebro[config.CY_DATE] >= pd.to_datetime(year_s)) & (df_celebro[config.CY_DATE] <= pd.to_datetime(year_f))]

    elif flag=='2015-2020':
        # 总的分布情况
        df_tmp=df_celebro
    # print('dic_code_region',dic_code_region)

    # 算人口
    df_tmp=df_tmp.sort_values(by=[config.CY_DATE])
    df_tmp=df_tmp.drop_duplicates(subset=[config.SFZH],keep='first')    # 一个患者只保留一条记录

    region_count=df_tmp[[config.XZZ_XZQH2,config.SFZH]].groupby([config.XZZ_XZQH2],as_index=False)[config.SFZH].count()  # return a Series
    # print(region_count)

    # region_df=pd.DataFrame(zip(region_count.index,region_count.values),columns=['region_code','count'])
    # region_df['region_name']=region_df['region_code'].apply(lambda x: dic_code_region[x] if x in dic_code_region else 'none')
    # print('region_df[region_df[region_name]!=none].shape[0]',region_df[region_df['region_name']!='none'].shape[0])
    # region_df=region_df[region_df['region_name']!='none']

    # MAP_DATA=list(zip(region_df['region_name'], region_df['count']))

    # MAP_DATA=[list(i) for i in MAP_DATA]
    # print(MAP_DATA)

    region_count['year']=flag
    print(region_count)

    return region_count



if __name__ == "__main__":
    """
    """
    # process = [40]
    process = [48]

    region_code_path = '../data/dic_183region_code.xlsx'
    region_code = pd.read_excel(region_code_path)

    region_code['code6']=region_code['code6'].apply(lambda x: str(x))   # int 转换为str

    dic_code_region = dict(zip(region_code['code6'], region_code['县市区']))  # 地区编码和区县名称字典
    dic_code_citycountry=dict(zip(region_code['code6'], region_code['城乡划分']))  # 地区编码和城乡划分字典
    dic_country_city=dict(zip(region_code['code6'], region_code['市州']))
    dic_code_economy=dict(zip(region_code['code6'], region_code['五大经济区']))

    sichuan_city_country_echarts_path='../data/sichuan_city_country.pkl'
    sichuan_city_country = pickle.load(open(sichuan_city_country_echarts_path, 'rb'))  # {市:[县1,县2...]}字典

    record_path = config.pdir +'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(record_path, 'rb'))  # 数据
    print('df_celebro.shape',df_celebro.shape)

    # 40- 检查字典和地图库中的地名对应关系  ; 若有不一致的对应关系，记得检查和更改
    if 40 in process:
        modified_sichuan_city_country=check_geo_info(sichuan_city_country,dic_code_region,sichuan_city_country_echarts_path,flag='modify')
        buyizhi_region=check_geo_info(modified_sichuan_city_country,dic_code_region,sichuan_city_country_echarts_path,flag='check')
        if len(buyizhi_region)==0:
            pickle.dump(modified_sichuan_city_country, open(sichuan_city_country_echarts_path, 'wb'), protocol=4)
        print(sichuan_city_country)


    # 41- 获取数据：分年画地图热力图-183个县市区
    if 41 in process:
        hotMapData=dict()
        for flag in ['2015-2020','2015','2016','2017','2018','2019','2020']:
            MAP_DATA=get_hotMAP_DATA(df_celebro, sichuan_city_country, dic_code_region,flag)
            hotMapData[flag]=MAP_DATA

        save_path =config.pdir + "Project_Cerebrovascular_data/median_data/dic_year__hotmap_MAP_DATA.pkl"
        pickle.dump(hotMapData, open(save_path, 'wb'), protocol=4)

        # 获取统计数据
        # get_region_statistics(hotMapData)


    # 42- 可视化：分年画地图热力图-183个县市区
    if 42 in process:
        read_path = config.pdir +"Project_Cerebrovascular_data/median_data/dic_year__hotmap_MAP_DATA.pkl"
        hotMapData=pickle.load(open(read_path, 'rb'))
        print(hotMapData)
        num_values=[]
        flag_y=['2015-2020','year'][0]

        if flag_y!='2015-2020':
            for i in hotMapData:
                # print(hotMapData[i])
                if len(hotMapData[i])>0 and i!='2015-2020':
                    _,nums=zip(*hotMapData[i])
                    num_values.extend(list(nums))

            max_value_year=max(num_values)
            min_value_year=min(num_values)
            min_value_year=0
            print(max_value_year)

            for i in hotMapData:
                if len(hotMapData[i])!=0 and i!='2015-2020':
                    year=i
                    MAP_DATA=hotMapData[i]
                    print('MAP_DATA',MAP_DATA)
                    hotmap_sicuhan_counties(year,MAP_DATA,min_value_year,max_value_year)
                    # "../../map_js/si4_chuan1_countries_yangping.js"

        elif flag_y=='2015-2020':
            _, nums = zip(*hotMapData[flag_y])
            print('nums',nums)
            max_value=max(list(nums))
            print('max_value',max_value)
            min_value=min(list(nums))
            min_value=0
            hotmap_sicuhan_counties(flag_y, hotMapData[flag_y], min_value, max_value)


    # 统计城乡 和 经济区
    if 43 in process:
        save_file = config.pdir +"Project_Cerebrovascular_data/results_tables/43- 2015-2020四川省脑血管住院患者城乡分布.csv"
        table43 = generate_table43(df_celebro, save_file,region_code)


    # 44- 统计跨市州住院人次
    if 44 in process:
        flow_mtxs,dic_code_row = data_popu_flow(df_celebro, dic_code_region)
        save_path1 = config.pdir +"Project_Cerebrovascular_data/median_data/44-flow_mtxs_years.pkl"
        pickle.dump(flow_mtxs, open(save_path1, 'wb'), protocol=4)
        save_path2 = config.pdir +"Project_Cerebrovascular_data/median_data/44-flow_mtxs_years_rowsDict.pkl"
        pickle.dump(dic_code_row, open(save_path2, 'wb'), protocol=4)

        dic_regioncode_popu=compute_total_num_redion(df_celebro,dic_code_region)  # 统计每个州的脑血管总人数
        compute_total_num_hosptial_redion(df_celebro, dic_code_region)


        # 区县的流入流出——用于画柱状图、散点图
        df_flow_popu=data_statistic_intercountry_popu_rate(flow_mtxs,dic_code_row,dic_code_region, dic_code_citycountry,dic_country_city,dic_regioncode_popu,dic_code_economy) # 45- 跨市州住院人次/比例
        #  df_flow_popu的字段： 区县  区县名称      流出人次   流入人次  城乡划分 地级市  流出比例  流入比例
        save_path3 = config.pdir +"Project_Cerebrovascular_data/median_data/44-df_flow_popu.pkl"
        pickle.dump(df_flow_popu, open(save_path3, 'wb'), protocol=4)

        # 每年的净流入流出—细化到”地级市“粒度- 用于画地图流向图
        net_flows, city_popus,dic_city_flow= data_net_flow_city(flow_mtxs,dic_code_row,region_code)
        save_path4 =config.pdir + "Project_Cerebrovascular_data/median_data/44-dfs_NetFlow_cities.pkl"   #6个df:  cols=[row  col  net_flow source city terminal city]
        pickle.dump(net_flows, open(save_path4, 'wb'), protocol=4)
        save_path5 = config.pdir +"Project_Cerebrovascular_data/median_data/44-city_hostLocal_popus.pkl"  # 六个字典
        pickle.dump(city_popus, open(save_path5, 'wb'), protocol=4)
        save_path6 =config.pdir + "Project_Cerebrovascular_data/median_data/44-dic_city_flow.pkl"  # 六个字典
        pickle.dump(dic_city_flow, open(save_path6, 'wb'), protocol=4)

        # 每年的净流入流出—细化到”区县“粒度- 用于画地图流向图
        net_flows_countries, countries_popus, dic_country_flow = data_net_flow_country(flow_mtxs, dic_code_row, region_code)
        save_path7 = config.pdir +"Project_Cerebrovascular_data/median_data/44-dfs_NetFlow_countries.pkl"  # 6个df:  cols=[row  col  net_flow source city terminal city]
        pickle.dump(net_flows_countries, open(save_path7, 'wb'), protocol=4)
        save_path8 = config.pdir +"Project_Cerebrovascular_data/median_data/44-country_hostLocal_popus.pkl"  # 六个字典
        pickle.dump(countries_popus, open(save_path8, 'wb'), protocol=4)
        save_path9 = config.pdir +"Project_Cerebrovascular_data/median_data/44-dic_country_flow.pkl"  # 六个字典
        pickle.dump(dic_country_flow, open(save_path9, 'wb'), protocol=4)

    if 45 in process:
    # 画流入流出情况散点图-183
        save_path3 = config.pdir +"Project_Cerebrovascular_data/median_data/44-df_flow_popu.pkl"
        df_flow_popu = pickle.load(open(save_path3, 'rb'))
        plot_45_bubble(df_flow_popu, flag = '五大经济区')

    if 46 in process:
        # 画流入流出情况柱状图-183
        save_path3 = config.pdir +"Project_Cerebrovascular_data/median_data/44-df_flow_popu.pkl"
        df_flow_popu = pickle.load(open(save_path3, 'rb'))
        plot_46_bucket(df_flow_popu, flag='五大经济区')

    if 47 in process:
        flag='city'
        if flag=='city':
            # 画流向图-21个地级市
            save_path4 = config.pdir +"Project_Cerebrovascular_data/median_data/44-dfs_NetFlow_cities.pkl"  # 6个df:  cols=[row  col  net_flow source city terminal city]
            save_path6 = config.pdir +"Project_Cerebrovascular_data/median_data/44-dic_city_flow.pkl"  # 六个字典
            net_flows = pickle.load(open(save_path4, 'rb'))
            dic_city_flows = pickle.load(open(save_path6, 'rb'))
            print('net_flows',net_flows)
            print('dic_city_flows',dic_city_flows)
            interCountries_flow(net_flows,dic_city_flows,flag=flag)

        elif flag=='country':
            save_path7 =config.pdir + "Project_Cerebrovascular_data/median_data/44-dfs_NetFlow_countries.pkl"
            save_path9 =config.pdir + "Project_Cerebrovascular_data/median_data/44-dic_country_flow.pkl"  # 六个字典
            net_flows = pickle.load(open(save_path7, 'rb'))
            dic_country_flows = pickle.load(open(save_path9, 'rb'))
            interCountries_flow(net_flows,dic_country_flows,flag=flag)

    if 48 in process:
        # 统计流入武侯区的患者来源；和流出成华区的患者去向
        # record_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
        # df_celebro = pickle.load(open(record_path, 'rb'))  # 数据
        # print('df_celebro.shape', df_celebro.shape)

        # 成华区代码：510108； 武侯区代码：510107   ；dic_code_region
        in_wuhou=df_celebro[df_celebro[config.DEPT_ADDRESSCODE2]=='513201'][[config.XZZ_XZQH2,config.SFZH]].groupby([config.XZZ_XZQH2],as_index=False).count()
        in_wuhou['区县']=in_wuhou[config.XZZ_XZQH2].apply(lambda x: dic_code_region[x])
        in_wuhou.to_csv(config.pdir + "Project_Cerebrovascular_data/median_data/48-in_maerkang.csv")

        # out_chenghua=df_celebro[df_celebro[config.XZZ_XZQH2]=='510108'][[config.DEPT_ADDRESSCODE2,config.SFZH]].groupby([config.DEPT_ADDRESSCODE2],as_index=False).count()
        # out_chenghua['区县']=out_chenghua[config.DEPT_ADDRESSCODE2].apply(lambda x: dic_code_region[x])
        # out_chenghua.to_csv(config.pdir + "Project_Cerebrovascular_data/median_data/48-out_chenghua.csv")

    # 49- 获取数据：分年画地图热力图-183个县市区前人口住院率
    if 49 in process:
        hotMapData = []
        for flag in ['2015-2020', '2015', '2016', '2017', '2018', '2019', '2020']:
            MAP_DATA = get_hotMAP_DATA_thousand(df_celebro,  flag)
            hotMapData.append(MAP_DATA)
        hotMapData = pd.concat(hotMapData, axis=0)
        print("hotMapData", hotMapData)  # 区县脑血管人数
        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/49_for_thousand_rate.csv"
        hotMapData.to_csv(save_path)
        # pickle.dump(hotMapData, open(save_path, 'wb'), protocol=4)

    # 50- 获取数据：各区县收治的CBVD住院人数
    if 50 in process:
        # 算人口
        df_celebro = df_celebro.sort_values(by=[config.CY_DATE])
        df_celebro = df_celebro.drop_duplicates(subset=[config.SFZH], keep='first')  # 一个患者只保留一条记录

        region_count = df_celebro[[config.DEPT_ADDRESSCODE2, config.SFZH]].groupby([config.DEPT_ADDRESSCODE2], as_index=False)[
            config.SFZH].count()  # return a Series

        save_path = config.pdir + "Project_Cerebrovascular_data/median_data/49_DEPTregion_population.csv"
        region_count.to_csv(save_path)
