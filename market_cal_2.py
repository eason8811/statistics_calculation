import market_cal_1 as mc1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#mc1.get_all_symbol_info()
df_data_org = pd.read_csv('kline_data_org_market.csv', index_col=0, encoding='gb2312') # gb2312
symbols = list(df_data_org.columns)
df_data = df_data_org.copy()
for col in df_data.columns:
    data_org = np.array(df_data.loc[:,col])
    data_max = np.max(data_org)
    data_mean = np.mean(data_org)
    data_std = np.std(data_org)
    data_2one = (data_org-data_mean)/data_max
    #data_2one = (data_org-data_mean)/data_std
    df_data.loc[:,col] = data_2one

data_org_mean = {}
data_org_std = {}
data_mean = {}
data_std = {}
for i in range(len(symbols)):
    data_mean[symbols[i]] = np.mean(df_data.loc[:,symbols[i]])
    data_std[symbols[i]] = np.std(df_data.loc[:,symbols[i]])

plt.plot(pd.Series(data_mean))
plt.plot(pd.Series(data_std))
plt.show()