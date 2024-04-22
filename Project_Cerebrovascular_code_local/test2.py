# 添加城乡字段
# todo check 'XZZ_XZQH2'是否是六位编码，以及是否有空值
region_code = pickle.load(open("dic_183region_code.xlsx", 'rb'))
dic_code2city = dict(zip(region_code['code6'], region_code['城乡划分']))
df_patient_case['is_city'] = df_patient_case['XZZ_XZQH2'].apply(
    lambda x: dic_code2city[x[:6]] if x[:6] in dic_code2city else 'nan')
df_patient_ctrl['is_city'] = df_patient_ctrl['XZZ_XZQH2'].apply(
    lambda x: dic_code2city[x[:6]] if x[:6] in dic_code2city else 'nan')
df_patient_case = df_patient_case[~(df_patient_case['is_city'] == 'nan')].reset_index(drop=True)
df_patient_ctrl = df_patient_ctrl[~(df_patient_ctrl['is_city'] == 'nan')].reset_index(drop=True)

# 得到住院年月
# todo check CY_DATE的类型，以及是否有空值
df_patient_case['CY_YEAR'] = df_patient_case['CY_DATE'].dt.year
df_patient_case['CY_MONTH'] = df_patient_case['CY_DATE'].dt.month
df_patient_ctrl['CY_YEAR'] = df_patient_ctrl['CY_DATE'].dt.year
df_patient_ctrl['CY_MONTH'] = df_patient_ctrl['CY_DATE'].dt.month

# 按年龄（±2）、相同性别、城/乡一致、相同出院年月 匹配对照组
df_casep = df_patient_case[['SFZH', 'XB', 'NL', 'CY_YEAR', 'CY_MONTH', 'is_city']]
df_controlp = df_patient_ctrl[['SFZH', 'XB', 'NL', 'CY_YEAR', 'CY_MONTH', 'is_city']]


def get_sfzhs(df_group):
    return df_group['NL'].values[0], df_group['XB'].values[0], df_group['CY_YEAR'].values[0], \
        df_group['CY_MONTH'].values[0], df_group['is_city'].values[0], list(df_group['SFZH'])


df_casep = df_casep.groupby(['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city']).apply(get_sfzhs)
df_casep = pd.DataFrame([list(i) for i in df_casep], columns=['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city', 'SFZHs'])
df_controlp = df_controlp.groupby(['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city']).apply(get_sfzhs)
df_controlp = pd.DataFrame([list(i) for i in df_controlp],
                           columns=['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city', 'SFZHs'])
df_controlp.rename(columns={"SFZHs": "SFZHs_ctrl"}, inplace=True)

# todo 如果内存不够大，则对df_casep进行分块
ls_blocks = []
block_num = 500
for block_i in tqdm(range(block_num)):  # 切分case df，分治；不然merge后得到的df_match所需存储空间会溢出内存

    df_casep_portion = df_casep.iloc[
                       int(block_i / block_num * df_casep.shape[0]): int((block_i + 1) / block_num * df_casep.shape[0]),
                       :]

    caseid_ctrlid_0 = pd.merge(df_casep_portion, df_controlp, on=['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city'],
                               how='left')
    caseid_ctrlid_0 = caseid_ctrlid_0[['SFZHs', 'SFZHs_ctrl']]

    ls_caseid_ctrlid = []
    nl_ranges = [-2, -1, 1, 2]  # 年龄（±2）
    for nl_range in nl_ranges:
        df_casep_portion['NL'] = df_casep_portion['NL'] + nl_range
        caseid_ctrlid = pd.merge(df_casep_portion, df_controlp, on=['NL', 'XB', 'CY_YEAR', 'CY_MONTH', 'is_city'],
                                 how='left')
        df_casep_portion['NL'] = df_casep_portion['NL'] - nl_range
        caseid_ctrlid.rename(columns={"SFZHs_ctrl": "SFZHs_ctrl" + str(nl_range)}, inplace=True)
        ls_caseid_ctrlid.append(caseid_ctrlid[["SFZHs_ctrl" + str(nl_range)]])
    caseid_ctrlid = pd.concat([caseid_ctrlid_0] + ls_caseid_ctrlid, axis=1)

    for nl_range in nl_ranges:
        caseid_ctrlid['SFZHs_ctrl'] = caseid_ctrlid. \
            apply(lambda row: row['SFZHs_ctrl'] + row['SFZHs_ctrl' + str(nl_range)], axis=1)

    caseid_ctrlid = caseid_ctrlid[['SFZHs', 'SFZHs_ctrl']]  # 可进行匹配的case sfzh —— ctrl sfzh
    ls_blocks.append(caseid_ctrlid)

ttl_caseid_ctrlid = pd.concat(ls_blocks, axis=0).reset_index(drop=True)
pickle.dump(ttl_caseid_ctrlid, open("caseid_ctrlid_%s.pkl" % (index_dis), 'wb'), protocol=4)

# 统计case / candidate ctrl的数量比值。根据比值的分布，确定control样本数/case样本数的倍数
ttl_caseid_ctrlid['case_num'] = ttl_caseid_ctrlid['SFZHs'].apply(lambda x: len(x))
ttl_caseid_ctrlid['ctrl_num'] = ttl_caseid_ctrlid['SFZHs_ctrl'].apply(lambda x: len(x))
ttl_caseid_ctrlid['ctrl/case'] = ttl_caseid_ctrlid['ctrl_num'] / ttl_caseid_ctrlid['case_num']
print(ttl_caseid_ctrlid['ctrl/case'].describe())  # 根据比值的分布，确定control样本数/case样本数的倍数

# 提取ctrl, ctrl的数量是case的multiple倍
multiple = 10


def get_sampled_ctrls(df_group):
    sample_num = df_group['case_num'] * multiple
    random.seed(1)
    df_group['SFZHs_ctrl_sampled'] = random.sample(df_group['SFZHs_ctrl'], min(sample_num, df_group['ctrl_num']))
    return df_group


ttl_caseid_ctrlid = ttl_caseid_ctrlid.apply(get_sampled_ctrls, axis=1)

df_ctrl_sfzhs = pd.DataFrame([list(i) for i in ttl_caseid_ctrlid['SFZHs_ctrl_sampled']], columns=['SFZH'])

