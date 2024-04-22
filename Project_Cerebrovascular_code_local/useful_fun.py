from pyecharts import options as opts
from pyecharts.charts import Geo, Map
from pyecharts.faker import Faker
import pickle
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot
import numpy as np
from pyecharts.globals import ChartType, SymbolType


def scatter():


    c = (
        Geo()
            .add_schema(maptype="四川")
            .add("geo", [['井研县', 1000]])
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(), title_opts=opts.TitleOpts(title="Geo-基本示例")
        )
            .render("geo_base.html")
    )


def sichuan_quxian_hotmap():
    MAP_DATA = [
        ["青羊区", 20057.34],
        ["成华区", 1547007.48],
        # ["绵阳市", 31686.1],
        # ["德阳市", 6992.6],
        ["井研县", 111992.6],
    ]

    sichuan_city_country = pickle.load(open('data/sichuan_city_country.pkl', 'rb'))
    print(sichuan_city_country)
    countries = []
    for i in sichuan_city_country:
        countries.extend(sichuan_city_country[i])

    print(countries)
    print(len(countries))
    MAP_DATA = list(
        zip(countries, np.random.randint(low=800, high=50000, size=(len(countries)), dtype='int').astype(float)))
    MAP_DATA = [list(i) for i in MAP_DATA]
    print(MAP_DATA)

    map = (Map(init_opts=opts.InitOpts(width="1400px", height="800px"))
        .add(
        series_name="四川-区县",
        maptype="四川-区县",  # E:/python37/Lib/site-packages/echarts_china_cities_pypkg/resources/echarts-china-cities-js/
        data_pair=MAP_DATA,
        is_map_symbol_show=False,
    )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 不显示标签
        .set_global_opts(
        title_opts=opts.TitleOpts(
            title="四川-区县",
            # subtitle="人口密度数据来自Wikipedia"
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{b}<br/>{c} (p / km2)",

        ),
        visualmap_opts=opts.VisualMapOpts(
            min_=800,
            max_=50000,
            range_text=["High", "Low"],
            is_calculable=True,
            range_color=["lightskyblue", "yellow", "orangered"],
        ),
    )

    )

    map.render("sichuan_city_country_hotmap.html")

    # # 输出保存为图片
    # make_snapshot(snapshot, map.render(), "Options配置项_自定义样式_保存图片.png")


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

    lst_xian=[]
    sichuan2=dict()
    for i in sichuan:
        for sep in i.split(' '):
            lst_xian.append(sep)
        sichuan2[i]=lst_xian
    print(sichuan2)

    pickle.dump(sichuan2,open('data/sichuan_city_country.pkl','wb'))

def flow_graph():


    c = (
        Geo()
            .add_schema(maptype="四川")
            .add(
            "",
            [("成都", 55), ("乐山", 66), ("绵阳", 77), ("南充", 88)],
            type_=ChartType.EFFECT_SCATTER,
            color="white",
        )
            .add(
            "geo",
            # [("井研县", "青羊区"), ("高坪区", "青羊区"), ("三台县", "青羊区"), ("盐边县", "青羊区")],
            [("乐山", "成都"), ("内江", "成都"), ("绵阳", "成都"), ("南充", "成都")],
            type_=ChartType.LINES,
            effect_opts=opts.EffectOpts(
                symbol=SymbolType.ARROW, symbol_size=6, color="blue"
            ),
            linestyle_opts=opts.LineStyleOpts(curve=0.2),
        )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(title_opts=opts.TitleOpts(title="Geo-Lines"))
            .render("geo_lines.html")
    )


