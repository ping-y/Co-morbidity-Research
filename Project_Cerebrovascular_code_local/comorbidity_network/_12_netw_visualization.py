from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ChartType, SymbolType, ThemeType
from pyecharts.render import make_snapshot
import config
import networkx as nx
import json
import numpy as np



def draw_netw(nodes,links,categories,render_html):
    c = (
        Graph(init_opts=opts.InitOpts(width="1000px", height="800px",theme=ThemeType.SHINE))
        .add(
            "",
            nodes=nodes,
            links=links,
            categories=categories,
            layout="circular",
            is_rotate_label=True,
            linestyle_opts=opts.LineStyleOpts(color="source", curve=0.3),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Graph-Les Miserables"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
        )
        .render(render_html)
    )



def data_for_draw_test():
    with open("les-miserables.json", "r", encoding="utf-8") as f:
        j = json.load(f)
        nodes = j["nodes"]
        links = j["links"]
        categories = j["categories"]

    print('nodes', nodes)
    print('links', links)
    print('categories', categories)

    return nodes,links,categories


def data_for_draw(g, edge_color):

    # 用于构建categories的json数据的前置步骤
    g.vs["dis_system"] = [ord(i[0]) - ord('A') for i in g.vs["name"]]
    dis_category = set(g.vs["dis_system"])
    dic_category = dict(zip(dis_category, range(len(dis_category))))

    # categories的json数据
    categories = [{"name": "%s00-%s99" % (chr(ord('A') + i), chr(ord('A') + i))}  for i in dic_category.keys()]


    # 用于构建节点json的前置步骤：vertex_weight and dic_category
    vertex_weight = [int(x + 1) for x in g.vs["prevalence"]]

    # 节点json数据
    nodes_data=[]
    for index, name in enumerate(g.vs["name"]):
        nodes_data.append({
                                              "name": name,
                                              "symbolSize": vertex_weight[index],
                                              "category": dic_category.get(g.vs["dis_system"][index]),
                                              "itemStyle":{ },   # "color": node_color
                                        })

    # 用于构建边json的前置步骤
    nodes_name = g.vs["name"]
    edges_weight = g.es["weight"]

    # 1. 边宽度
    xmin = min(edges_weight)
    xmax = max(edges_weight)
    edges_width = []  # 边宽度
    if max(edges_weight) > 8:
        MIN ,MAX= 1,8
        edges_width = [int(MIN + (MAX - MIN) / (xmax - xmin) * (x - xmin)) for x in edges_weight]
    else:
        edges_width = [int(x - 0.5) for x in edges_weight]  # 边宽度 向下取证

    # 2. 边透明度
    edges_opacity = []  # 边透明度
    x1, x2, x3 = np.percentile(edges_weight, [30, 60, 90])
    for each_weight in edges_weight:
        if each_weight > x3:
            edges_opacity.append(0.9)
        elif each_weight > x2:
            edges_opacity.append(0.6)
        else:
            edges_opacity.append(0.3)

    # 3. 边颜色
    color = 'black'  # 默认为黑色
    if edge_color is not None:
        color = edge_color


    # 边json数据
    links_data=[]
    for index, edge in enumerate(g.get_edgelist()):
        # print(index,edge[0],edge[1])
        links_data.append({
                                        "source": nodes_name[edge[0]],
                                        "target": nodes_name[edge[1]],
                                        "value": edges_weight[index],
                                        "lineStyle": {
                                                      "color": color,
                                                      "width": edges_width[index],
                                                      "opacity": edges_opacity[index]
                                                         }
                                      })

    return nodes, links, categories,lowest_corr


def get_color(key):
    colors = {"blue": "#ff0033",
              "orange": "#FFA500",
              "green": "#ff0033",
              "purple": "#800080",
              "cyan": "#ff0033",
              "red": "#ff0033",
              "yellow": "#ffff00",
              "DarkOrange": "#FF8C00",
              "DarkRed": "#8B0000",
              "DeepSkyBlue": "#00BFFF",
              "Firebrick": "#B22222",
              "Magenta": "#FF00FF",
              "Navy": "#000080",
              "DarkTurquoise": "#00CED1"}
    return colors[key]


if __name__=="__main__":

    participant = ['case', 'ctr', 'gender=1', 'gender=2', 'age_group=0', 'age_group=1', 'age_group=2', 'city','country']
    layer_ecolor = dict(zip(participant, ['blue', 'orange', 'green', 'purple', 'cyan', 'red', 'yellow', 'DarkOrange', 'DarkRed']))
    graph_htmls = 'graph_htmls/'
    layer = -1
    type_ = 'phi'

    for i in participant:
        gml_path = config.pdir + "Project_Cerebrovascular_data/results_tables/111_CD_%s_%s_graph_%s_layer%s.gml" % ( i, type_, str(0), str(layer))
        g = nx.read_gml(gml_path)

        edge_color=get_color(layer_ecolor[i])
        nodes, links, categories,lowest_corr=data_for_draw(g,edge_color)

        render_html=graph_htmls+"graph_%s_%s_%s_%s.html"% (i, type_, str(lowest_corr), str(layer))

        draw_netw(nodes, links, categories, render_html)

