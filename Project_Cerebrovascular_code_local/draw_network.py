from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.globals import ChartType, SymbolType, ThemeType
from pyecharts.render import make_snapshot


for layer, dic_layer in zip(
            ["total", "XB", "NL_GROUP", "MZ_2", "POOR", "URBAN", "DOWNTOWN", "I20", "I21", "I24", "I25"],
            [[None], dic_gender, dic_age_segment, dic_nation, dic_poor_city, dic_urban, dic_downtown, dic_i20, dic_i21,
             dic_i24, dic_i25]):
        for key in dic_layer:  # 遍历每个分层的值
            if key == 9 or key == -1:
                continue  # 跳过 不详 的代号
            if layer in ["I20", "I21", "I24", "I25"] and case_or_control in [1, 3]:
                continue  # 跳过单个疾病对照组
            if case_or_control in [2, 3] and layer == "POOR" and key == 1:
                continue  # 成都POOR=1分层无人
            print("开始分层画图  %s %s" % (layer, key))
            file_path = "COMOR_2IHD/networks/gml/%s_%s_flag=%s_%d_%s.gml" % (
                            layer, key, 1, int(p_value * 100), filename)
            result_file_path = "COMOR_2IHD/figure/networks/%s_%s_flag=%s_%d_%s.html" % (
                layer, key, 1, int(p_value * 100), filename)
            draw_comornetwork_core(key, file_path, result_file_path) # key 是每个分层的值，file_path是

def draw_comornetwork_core(key, file_path, result_file_path):
    g = load(file_path)

    color = 'black'  # 默认为黑色
    colors = {"blue": "#ff0033", "orange": "#FFA500", "green": "#ff0033", "purple": "#800080","cyan": "#ff0033",
              "red": "#ff0033", "yellow": "#ffff00", "DarkOrange": "#FF8C00","DarkRed": "#8B0000", "DeepSkyBlue": "#00BFFF",
              "Firebrick": "#B22222","Magenta":"#FF00FF","Navy":"#000080","DarkTurquoise":"#00CED1"}
    # 设置边颜色
    if key is not None:
        color = colors.get(list(colors.keys())[key])
    # 疾病系统
    categories = []
    g.vs["dis_system"] = [ord(i[0]) - ord('A') for i in g.vs["name"]]
    dis_category = set(g.vs["dis_system"])
    dic_category = dict([x for x in zip(dis_category, range(len(dis_category)))])
    for i in dic_category.keys():
        categories.append({"name": "%s00-%s99" % (chr(ord('A') + i), chr(ord('A') + i))})
    # 设置点颜色，根据疾病系统
    color_num = [dic_category.get(x) for x in g.vs["dis_system"]]
    node_color = [list(colors.values())[x] for x in color_num]
    #print(node_color)

    nodes_data = []
    nodes_name = g.vs["name"]
    vertex_weight = [int(x + 1) for x in g.vs["prevalence"]]
    for index, name in enumerate(nodes_name):
        nodes_data.append({"name": name,
                           "symbolSize": vertex_weight[index],
                           "category": dic_category.get(g.vs["dis_system"][index]),
                           "itemStyle":{
                               # "color": node_color
                           },

                           })

    links_data = []
    edges_weight = g.es["weight"]  # 边权重
    xmin = min(edges_weight);
    xmax = max(edges_weight);
    x1, x2, x3 = np.percentile(edges_weight, [30, 60, 90])
    edges_width = []  # 边宽度
    # edges_width = [int(x-0.5) for x in edges_weight]    # 边宽度
    if max(edges_weight) > 8:
        MIN = 1;
        MAX = 8;
        edges_width = [int(MIN + (MAX - MIN) / (xmax - xmin) * (x - xmin)) for x in edges_weight]
    else:
        edges_width = [int(x - 0.5) for x in edges_weight]  # 边宽度 向下取证

    edges_opacity = []  # 边透明度
    for each_weight in edges_weight:
        if each_weight > x3:
            edges_opacity.append(0.9)
        elif each_weight > x2:
            edges_opacity.append(0.6)
        else:
            edges_opacity.append(0.3)

    for index, edge in enumerate(g.get_edgelist()):
        # print(index,edge[0],edge[1])
        links_data.append(
            {"source": nodes_name[edge[0]],
             "target": nodes_name[edge[1]],
             "value": edges_weight[index],
             "lineStyle": {
                 "color": color,
                 "width": edges_width[index],
                 "opacity": edges_opacity[index]
             }
             })
        # if edges_weight[index]<x3:  # 关系小于某个阈值的边不显示
        #     links_data[index]["show"]="false"

    def chart() -> Graph:
        c = (
            Graph(init_opts=opts.InitOpts(width="1000px", height="800px", theme=ThemeType.SHINE))
                # ThemeType。 LIGHT\INFOGRAPHIC\MACARONS\SHINE\WALDEN
                .add(
                "",
                nodes=nodes_data,
                links=links_data,
                categories=categories,
                layout="circular",
                is_rotate_label=True,
                linestyle_opts=opts.LineStyleOpts(color="black", curve=0.3),
                label_opts=opts.LabelOpts(position="right"),
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="Graph-IHD-Cormorbidity-Network"),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="2%", pos_top="20%"),
            )
                .render(result_file_path)
        )
        return c

    chart()
    # make_snapshot(snapshot, chart().render(result_file_path), "bar.png")