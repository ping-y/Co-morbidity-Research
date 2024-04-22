import pickle
import config


load_path = config.pdir + 'Project_Cerebrovascular_data/cerebro_data_afterFirstCerebrov.pkl'
df_celebro = pickle.load(open(load_path, 'rb'))
print(df_celebro)
df_celebro=df_celebro[[config.SFZH,config.ALL_DISEASE]]