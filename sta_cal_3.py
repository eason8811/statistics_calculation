import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import DFGLS


long = 1
short = -1
empty = 0

class Position:
    def __init__(self,side= empty,position= 0,entry_time= 0,entry_price= 0,exit_time= 0,exit_price= 0):
        self.side = side
        self.position = position
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.exit_price = exit_price
    def open_pos(self,side,position,entry_time,entry_price):
        #检查是否有仓位
        if self.side == empty:
            self.side = side
            self.position = position
            self.entry_price = entry_price
            self.entry_time = entry_time
        else:
            #检查方向是否一致
            if side != self.side:
                print(f"仓位方向不一致，原仓位方向为{self.side}，开仓方向为{side}")
                return None
            else:
                self.entry_price = (entry_price*position + self.entry_price*self.position) / (position+self.position)
                self.position = self.position+position
                self.entry_time = entry_time
    def close_pos(self,position,exit_time,exit_price):
        self.position = self.position - position
        self.exit_time = exit_time
        self.exit_price = exit_price
        if self.side == short:
            profit = (self.entry_price - self.exit_price) / self.entry_price * position
            if self.position == 0.0:
                self.side = empty
            return profit
        elif self.side == long:
            profit = (self.exit_price - self.entry_price) / self.entry_price * position
            if self.position == 0.0:
                self.side = empty
            return profit
    def check_pos_profit(self,now_price):
        if self.entry_price != 0:
            if self.side == short:
                profit = (self.entry_price - now_price) / self.entry_price * self.position
                return profit
            elif self.side == long:
                profit = (now_price - self.entry_price) / self.entry_price * self.position
                return profit
        else:
            return 0

symbol_pairs = [['CHZUSDT', 'CTSIUSDT'], ['CFXUSDT', 'IDUSDT'], ['CTSIUSDT', 'ONTUSDT'], ['CTSIUSDT', 'BATUSDT'], ['CTSIUSDT', 'DARUSDT'], ['HOOKUSDT', 'RLCUSDT'], ['GTCUSDT', 'RLCUSDT'], ['ALPHAUSDT', 'IDUSDT'], ['ALPHAUSDT', '1000LUNCUSDT'], ['IDUSDT', 'MANAUSDT'], ['IDUSDT', 'FTMUSDT'], ['IDUSDT', 'CTKUSDT'], ['IDUSDT', 'SANDUSDT'], ['IDUSDT', 'ENJUSDT'], ['IDUSDT', 'BNXUSDT'], ['RLCUSDT', 'ALICEUSDT'], ['RLCUSDT', 'LUNA2USDT'], ['ZILUSDT', 'XEMUSDT'], ['PEOPLEUSDT', 'XEMUSDT'], ['SUSHIUSDT', 'API3USDT']]

#symbol_pairs = [['IDUSDT', 'SANDUSDT']]
period = 1000

#处理symbols
symbol_pairs_after = []
for i in range(len(symbol_pairs)):
    symbol_pairs_after.extend(symbol_pairs[i])
    
symbol_pairs_after = list(set(symbol_pairs_after))
print(symbol_pairs_after)
print('\n\n')

#csv读取所有数据
df_data = pd.read_csv('kline_data_symbol_close_rate2one.csv', encoding='gb2312').loc[:,symbol_pairs_after] # gb2312
df_data_org = pd.read_csv('kline_data_org.csv', encoding='gb2312').loc[:,symbol_pairs_after] # gb2312
symbols = list(df_data.columns)
data = df_data.to_dict('list')
data_matric = df_data.values
data_org = df_data.to_dict('list')
data_org_2_one = {}
for i in range(len(data_org.keys())):
    data_symbol = data_org[list(data_org.keys())[i]]
    data_org_2_one[list(data_org.keys())[i]] = (data_symbol - np.mean(data_symbol)) / max(data_symbol)
df_data_org_2_one = pd.DataFrame(data_org_2_one)

symbols_params = []
for i in range(len(symbol_pairs)):
    y = df_data_org_2_one.loc[:,symbol_pairs[i][0]].values
    x = df_data_org_2_one.loc[:,symbol_pairs[i][1]].values
    model = sm.OLS(y, x)  # 生成模型
    result = model.fit()  # 模型拟合
    print(f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]} 的相关系数 R^2 =  {result.rsquared}')
    after_adjust_result_params = 1
    print(f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]} 的拟合直线斜率 k =  {result.params[0]}')
    #print(result.summary())
    symbols_params.append(result.params)
    consq = df_data_org.loc[:,symbol_pairs[i][0]].values - after_adjust_result_params * df_data_org.loc[:,symbol_pairs[i][1]].values
    #consq = y - x * result.params[0]
    consq_2_one = (consq - np.mean(consq)) / np.std(consq)
    adf_consq = adfuller(consq,regression='c')
    dfgls_consq = DFGLS(consq)
    if adf_consq[0] < adf_consq[4]['1%']:
        print(f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]} 是平稳曲线的把握 99%')
        if adf_consq[1] < 0.05:
            print(f'P = {adf_consq[1]} < 0.05，拟合程度（好）')
        else:
            print(f'P = {adf_consq[1]} > 0.05，拟合程度（差）')
    elif adf_consq[0] < adf_consq[4]['5%']:
        print(f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]} 是平稳曲线的把握 95%')
        if adf_consq[1] < 0.05:
            print(f'P = {adf_consq[1]} < 0.05，拟合程度（好）')
        else:
            print(f'P = {adf_consq[1]} > 0.05，拟合程度（差）')
    elif adf_consq[0] < adf_consq[4]['10%']:
        print(f'{symbol_pairs[i][0]} - {symbol_pairs[i][1]} 是平稳曲线的把握 90%')
        if adf_consq[1] < 0.05:
            print(f'P = {adf_consq[1]} < 0.05，拟合程度（好）')
        else:
            print(f'P = {adf_consq[1]} > 0.05，拟合程度（差）')
    else:
        print('曲线不平稳')


    #布林带及均线
    consq_std = np.std(consq)
    consq_mean = np.mean(consq)

    print('='*20)
    print(f"param = {after_adjust_result_params}")
    print(f'consq[0] = {consq[0]}')
    print(f'consq_mean = {consq_mean}')
    print((f'consq_std = {consq_std}'))
    print(f'result = {adf_consq}')
    print(f'dfgls = {dfgls_consq}')
    print('\n')

    plt.clf()
    plt.scatter(x,y,marker='.',s=10)
    plt.plot(x,result.params * x,color='r')
    plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbol_pairs[i][0]} - {symbol_pairs[i][1]} linear.png')
    plt.clf()
    plt.plot(consq,linewidth=1)
    #plt.plot(np.ones(2880) * consq_mean)
    plt.plot(np.ones(len(consq)) * consq_mean)
    plt.plot(np.ones(len(consq)) * 0.3 * consq_std + consq_mean)
    plt.plot(np.ones(len(consq)) * (-0.3) * consq_std + consq_mean)
    plt.plot(np.ones(len(consq)) * 9 * 0.3 * consq_std + consq_mean)
    plt.plot(np.ones(len(consq)) * 9 * (-0.3) * consq_std + consq_mean)
    #consq_mean = 0.00157777
    #plt.plot(np.arange(2880),(profit_line - np.mean(profit_line)) / np.std(profit_line)/100)
    #plt.plot(np.arange(2880), (amount_list_test - np.mean(amount_list_test)) / max(amount_list_test)/100)
    plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbol_pairs[i][0]} - {symbol_pairs[i][1]} coint.png')
plt.show()











