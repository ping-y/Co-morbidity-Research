import pickle
import pandas as pd
import numpy as np
import config
# import jieba
# import wordcloud
# import Image
# import matplotlib.pyplot as plt



def get_sichuan_countries():
    sichuan=dict()
    sichuan['凉山彝族自治州']='会东县 会理县 冕宁县 喜德县 宁南县 布拖县 德昌县 昭觉县 普格县 木里藏族自治县 甘洛县 盐源县 美姑县 西昌市 越西县 金阳县 雷波县'
    sichuan['南充市']='仪陇县 南部县 嘉陵区 营山县 蓬安县 西充县 阆中市 顺庆区 高坪区'
    sichuan['乐山市']='五通桥区 井研县 夹江县 峨眉山市 峨边彝族自治县 乐山市市中区 沐川县 沙湾区 犍为县 金口河区 马边彝族自治县'
    sichuan['德阳市']='中江县 什邡市 广汉市 旌阳区 绵竹市 罗江县'
    sichuan['绵阳市']='三台县 北川羌族自治县 安州区 平武县 梓潼县 江油市 涪城区 游仙区 盐亭县'
    sichuan['成都市']='双流区 大邑县 崇州市 彭州市 成华区 新津县 新都区 武侯区 温江区 简阳市 蒲江县 邛崃市 郫都区 都江堰市 金堂县 金牛区 锦江区 青白江区 青羊区 龙泉驿区'
    sichuan['眉山市']='东坡区 丹棱县 仁寿县 彭山区 洪雅县 青神县'
    sichuan['雅安市']='名山区 天全县 宝兴县 汉源县 石棉县 芦山县 荥经县 雨城区'
    sichuan['资阳市']='乐至县 安岳县 雁江区'
    sichuan['阿坝藏族羌族自治州']='九寨沟县 壤塘县 小金县 松潘县 汶川县 理县 红原县 若尔盖县 茂县 金川县 阿坝县 马尔康市 黑水县'
    sichuan['达州市']='万源市 大竹县 宣汉县 开江县 渠县 达川区 通川区'
    sichuan['甘孜藏族自治州']='丹巴县 九龙县 乡城县 巴塘县 康定市 得荣县 德格县 新龙县 泸定县 炉霍县 理塘县 甘孜县 白玉县 石渠县 稻城县 色达县 道孚县 雅江县'
    sichuan['内江市']='东兴区 威远县 内江市市中区 资中县 隆昌县'
    sichuan['巴中市']='南江县 巴州区 平昌县 恩阳区 通江县'
    sichuan['泸州市']='叙永县 古蔺县 合江县 江阳区 泸县 纳溪区 龙马潭区'
    sichuan['攀枝花市']='东区 仁和区 盐边县 米易县 西区'
    sichuan['遂宁市']='大英县 安居区 射洪县 船山区 蓬溪县'
    sichuan['广元市']='利州区 剑阁县 旺苍县 昭化区 朝天区 苍溪县 青川县'
    sichuan['自贡市']='大安区 富顺县 沿滩区 自流井区 荣县 贡井区'
    sichuan['广安市']='前锋区 华蓥市 岳池县 广安区 武胜县 邻水县'
    sichuan['宜宾市']='兴文县 南溪区 宜宾县 屏山县 江安县 珙县 筠连县 翠屏区 长宁县 高县'
    print(sichuan)


    sichuan2=dict()
    num=0
    for i in sichuan:
        lst_xian = []
        for sep in sichuan[i].split(' '):
            lst_xian.append(sep)
        sichuan2[i]=lst_xian
    print(sichuan2)

    pickle.dump(sichuan2,open('data/sichuan_city_country.pkl','wb'))


def check_age_18():
    read_path = 'F:/Project_Cerebrovascular_data/cerebro_data.pkl'
    df = pickle.load(open(read_path, 'rb'))
    df = df[df['NL'] < 18]
    print(df.shape)
    print(df['SFZH'].drop_duplicates().shape)

def clean_df_celebro():
    load_path = 'F:/Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df = pickle.load(open(load_path, 'rb'))
    print('1. df.shape', df.shape)

    # 获取四川省区县编码
    region_code_path = 'data/dic_183region_code.xlsx'
    region_code = pd.read_excel(region_code_path)
    region_code['code6'] = region_code['code6'].apply(lambda x: str(x))  # int 转换为str
    code6_nd = region_code['code6'].values
    print('code6_lst', code6_nd.shape[0], code6_nd)

    print('region_code',list(region_code['县市区'].values))

    print('XZZ_XZQH2', list(df[config.XZZ_XZQH2].drop_duplicates().values))
    print('DEPT_ADDRESSCODE2',list(df[config.DEPT_ADDRESSCODE2].drop_duplicates().values))
    # # 过一遍现住址 ： 不在四川省内的人的所有记录都删掉
    # df['is_sichuanren'] = df[config.XZZ_XZQH2].apply(lambda x: 1 if x in code6_nd else 0)
    # del_id = df[df['is_sichuanren'] == 0][config.SFZH].drop_duplicates().values
    # save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)),
    #                        columns=[config.SFZH])  # 要保留的id
    # print("现住址不在四川的脑血管疾病患者人数：", del_id.shape[0])
    # old_dflen = df.shape[0]
    # df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    # print("现住址不在四川的脑血管疾病住院记录数：", old_dflen - df.shape[0])
    # del df['is_sichuanren']
    # print('2. df.shape', df.shape)
    #
    # # 过一遍医院地址编码：不在四川省内的人的所有记录都删除
    # df['is_sichuanren'] = df[config.DEPT_ADDRESSCODE2].apply(lambda x: 1 if x in code6_nd else 0)
    # del_id = df[df['is_sichuanren'] == 0][config.SFZH].drop_duplicates().values
    # save_id = pd.DataFrame(list(set(df[config.SFZH].drop_duplicates().values) - set(del_id)),
    #                        columns=[config.SFZH])  # 要保留的id
    # print("医院住址不在四川的脑血管疾病患者人数：", del_id.shape[0])
    # old_dflen = df.shape[0]
    # df = pd.merge(df, save_id, on=[config.SFZH]).reset_index(drop=True)
    # print("医院住址不在四川的脑血管疾病住院记录数：", old_dflen - df.shape[0])
    # del df['is_sichuanren']
    #
    # print('5. df.shape',df.shape)
    # save_path='F:/Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    # pickle.dump(df, open(save_path, 'wb'), protocol=4)



def jieba_cut():
    with open("data/test.txt", "r", encoding="utf-8") as f:
        mask = np.array(Image.open("data/sichuan_image.jpg"))
        text = f.read()
        w = wordcloud.WordCloud(background_color='white',
                                # font_path="C:/Windows/Fonts/simhei.ttf",
                                mask=mask)
        w.generate(text)
        w.to_file("test.png")
        plt.imshow(w)
        plt.show()

def check_rydate():
    load_path = config.pdir+'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df = pickle.load(open(load_path, 'rb'))
    tmp=df[(df[config.RY_DATE]<pd.to_datetime('2015-01-01'))& (df[config.CY_DATE]>=pd.to_datetime('2015-01-01'))]

    print(tmp[[config.SFZH,config.RY_DATE,config.RY_DATE]])
    print(tmp.shape)
    print()
    print(tmp[[config.RY_DATE,config.RY_DATE]].describe())


def test():
    load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
    df_celebro = pickle.load(open(load_path, 'rb'))
    # print('1. df_celebro.shape', df_celebro.shape)
    # print('1. list(df_celebro.columns)', list(df_celebro.columns))
    #
    # print(df_celebro[[config.FLAGS,config.SFZH]].groupby(config.FLAGS).count())
    # df_celebro=df_celebro[(df_celebro[config.FLAGS]=='4' )|(df_celebro[config.FLAGS]=='6')]
    # pickle.dump(df_celebro,open(config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl', 'wb'), protocol=4)

    print('1.1. df_celebro.shape', df_celebro.shape)
    print('1.1. list(df_celebro.columns)', list(df_celebro.columns))

    print('is_source', df_celebro[df_celebro['is_source']==1].shape)
    print('is source 2', df_celebro[df_celebro['is_source']==1][config.SFZH].drop_duplicates())

    print('df_celebro[config.SFZH].drop_duplicates().shape', df_celebro[config.SFZH].drop_duplicates().shape)

    df_celebro=df_celebro[['RY_DATE', 'NL','SFZH', 'CY_DATE']].drop_duplicates()
    print('2. df_celebro.shape', df_celebro.shape)
    print('2. df_celebro.drop_duplicates().shape', df_celebro[config.SFZH].drop_duplicates().shape)
    print('2. list(df_celebro.columns)', list(df_celebro.columns))

def test_cases():
    load_file = config.pdir + "Project_Cerebrovascular_data/89_df_final_case_netw.pkl"
    df_final_controlp = pickle.load(open(load_file, 'rb'))
    print(df_final_controlp)
    print('df_final_controlp.shape',df_final_controlp.shape)
    print('columns', list(df_final_controlp.columns))
    print('sfzh',df_final_controlp[config.SFZH].drop_duplicates().shape)
    print(df_final_controlp[config.ALL_DISEASE])

if __name__ == "__main__":
    # get_sichuan_countries()
    # clean_df_celebro()
    # jieba_cut()
    # check_rydate()
    test_cases()