
import ssl

import pyecharts.options as opts
from pyecharts.charts import Map
from pyecharts.datasets import register_url
from pyecharts.render import make_snapshot
# 使用snapshot-selenium 渲染图片
from snapshot_selenium import snapshot
import pickle
import numpy as np


def hotmap_sicuhan_counties():
    """
    Gallery 使用 pyecharts 1.1.0 和 echarts-china-cities-js
    参考地址: https://echarts.apache.org/examples/editor.html?c=map-HK
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    # 与 pyecharts 注册，当画香港地图的时候，用 echarts-china-cities-js
    register_url("https://echarts-maps.github.io/echarts-china-cities-js")


    MAP_DATA = [
        ["青羊区", 20057.34],
        ["成华区", 1547007.48],
        # ["绵阳市", 31686.1],
        # ["德阳市", 6992.6],
        ['井研县', 111992.6],
    ]
    print(type(MAP_DATA))
    print(type(MAP_DATA[0]))
    print(type(MAP_DATA[0][1]))


    sichuan_city_country=pickle.load(open('data/sichuan_city_country.pkl','rb'))
    print(sichuan_city_country)
    countries = []
    for i in sichuan_city_country:
        countries.extend(sichuan_city_country[i])

    print(countries)
    print(len(countries))
    MAP_DATA=list(zip(countries,np.random.randint(low=800,high=50000,size=(len(countries)),dtype='int').astype(float)))
    MAP_DATA=[list(i) for i in MAP_DATA]
    print(MAP_DATA)

    print(type(MAP_DATA))
    print(type(MAP_DATA[0]))
    print(type(MAP_DATA[0][1]))


    map=(Map(init_opts=opts.InitOpts(width="1400px", height="800px"))
        .add(
            series_name="四川-区县",
            maptype="四川-区县",    # E:/python37/Lib/site-packages/echarts_china_cities_pypkg/resources/echarts-china-cities-js/
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

    map.render("population_density_of_HongKong_v3.html")

    # 输出保存为图片
    # make_snapshot(snapshot, map.render(), "Options配置项_自定义样式_保存图片.png")



def inter_countried_flow():
    from pyecharts import options as opts
    from pyecharts.charts import Geo
    from pyecharts.globals import ChartType, SymbolType

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


#TODO 检查一下有两个县的名字是否没有对上。
sichuan_city_country=pickle.load(open('data/sichuan_city_country.pkl','rb'))
print(sichuan_city_country)