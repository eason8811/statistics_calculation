import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


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

symbol_pairs = [['EOSUSDT', 'MANAUSDT'], ['HOOKUSDT', 'REEFUSDT'], ['HOOKUSDT', 'ALICEUSDT'], ['REEFUSDT', 'ALICEUSDT'], ['DASHUSDT', 'HOTUSDT'], ['DASHUSDT', 'ALICEUSDT'], ['MANAUSDT', 'GMTUSDT'], ['MANAUSDT', 'SANDUSDT'], ['MANAUSDT', 'PEOPLEUSDT'], ['HOTUSDT', 'ALICEUSDT'], ['HOTUSDT', 'LRCUSDT'], ['HOTUSDT', 'LUNA2USDT'], ['MINAUSDT', 'ONEUSDT'], ['IOSTUSDT', 'ICXUSDT'], ['SPELLUSDT', 'DARUSDT'], ['ONEUSDT', 'LRCUSDT'], ['DARUSDT', 'LRCUSDT'], ['ALICEUSDT', 'PEOPLEUSDT'], ['ALICEUSDT', 'LRCUSDT'], ['LRCUSDT', 'LUNA2USDT']]

#symbol_pairs = [['ALGOUSDT', 'FLMUSDT']]#['ALGOUSDT', 'FLMUSDT'], ['MANAUSDT', 'KAVAUSDT'],['MAVUSDT', 'GMTUSDT'],['ZECUSDT', 'NEOUSDT']
period = 1000

#处理symbols
symbol_pairs_after = []
for i in range(len(symbol_pairs)):
    symbol_pairs_after.extend(symbol_pairs[i])
    
symbol_pairs_after = list(set(symbol_pairs_after))
print(symbol_pairs_after)
print('\n\n')

#csv读取所有数据
df_data = pd.read_csv('kline_data_symbol_close_rate2one.csv', index_col=0, encoding='gb2312').loc[:,symbol_pairs_after] # gb2312
df_data_org = pd.read_csv('kline_data_org.csv', index_col=0, encoding='gb2312').loc[:,symbol_pairs_after] # gb2312
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
    adf_consq = adfuller(consq)
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
    print('\n')

    amount_list_test = []
    profit_line = []
    initial_fund = 1
    position_long = Position()
    position_short = Position()
    param_ratio = after_adjust_result_params
    initial_price = consq[0]
    interval_ratio = 0.3
    each_position_ratio = 0.05
    each_position = initial_fund * each_position_ratio

    #当前仓位数量
    now_amount = 0
    y_past_price = 0
    x_past_price = 0
    for n in range(len(consq_2_one)):
        y_price = df_data_org.loc[:, symbol_pairs[i][0]].values[n]
        x_price = df_data_org.loc[:, symbol_pairs[i][1]].values[n]
        #计算应有仓位
            #计算余数
        remain = ((consq[n] - initial_price) / consq_std)%interval_ratio
        amount = 0
            #计算仓位个数
        if consq[n] > initial_price:
            #价格高于初始价格，做空组合
            amount = int(((consq[n] - initial_price)/consq_std - remain) / interval_ratio)
        elif consq[n] < initial_price:
            #价格低于初始价格，做多组合
            amount = int(((consq[n] - initial_price)/consq_std + remain) / interval_ratio)
        amount_list_test.append(amount)
        if abs(amount) < abs(now_amount) or abs(amount) < abs(now_amount) :
            #平仓
            if amount > 0 :
                #平 y 空仓，平 x 多仓
                roi = (x_price * y_past_price - x_past_price * y_price) / x_past_price / y_past_price
                if len(profit_line) == 0:
                    profit_line.append(roi)
                else:
                    profit_line.append(profit_line[-1] + roi)
                now_amount = amount
            elif amount < 0 :
                #平 y 多仓，平 x 空仓
                roi = (x_price * y_past_price - x_past_price * y_price) / x_past_price / y_past_price
                if len(profit_line) == 0:
                    profit_line.append(roi)
                else:
                    profit_line.append(profit_line[-1] + roi)
                now_amount = amount
            else:
                if amount_list_test[-2] > 0:
                    # 平 y 空仓，平 x 多仓
                    roi = (x_price * y_past_price - x_past_price * y_price) / x_past_price / y_past_price
                    if len(profit_line) == 0:
                        profit_line.append(roi)
                    else:
                        profit_line.append(profit_line[-1] + roi)
                    now_amount = amount
                elif amount_list_test[-2] < 0:
                    # 平 y 多仓，平 x 空仓
                    roi = (x_price * y_past_price - x_past_price * y_price) / x_past_price / y_past_price
                    if len(profit_line) == 0:
                        profit_line.append(roi)
                    else:
                        profit_line.append(profit_line[-1] + roi)
                    now_amount = amount
        elif abs(amount) > abs(now_amount) or abs(amount) > abs(now_amount) :
            #开仓
            if amount > 0:
                # 开 y 空仓，开 x 多仓
                y_past_price = y_price
                x_past_price = x_price
                now_amount = amount
                if len(profit_line) == 0:
                    profit_line.append(0)
                else:
                    profit_line.append(profit_line[-1])
            elif amount < 0:
                # 开 y 多仓，开 x 空仓
                y_past_price = y_price
                x_past_price = x_price
                now_amount = amount
                if len(profit_line) == 0:
                    profit_line.append(0)
                else:
                    profit_line.append(profit_line[-1])
        else:
            if len(profit_line) == 0:
                profit_line.append(0)
            else:
                profit_line.append(profit_line[-1])
        #更新initial_fund
        '''if amount > 0:
            profit_short = position_short.check_pos_profit(y_price)
            profit_long = position_long.check_pos_profit(x_price)
            initial_fund = initial_fund + profit_short + profit_long
            profit_line.append(initial_fund)
        elif amount < 0:
            profit_long = position_long.check_pos_profit(y_price)
            profit_short = position_short.check_pos_profit(x_price)
            initial_fund = initial_fund + profit_short + profit_long
            profit_line.append(initial_fund)
        elif amount == 0:
            profit_line.append(initial_fund)'''

    plt.clf()
    plt.scatter(x,y,marker='.',s=10)
    plt.plot(x,result.params * x,color='r')
    plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbol_pairs[i][0]} - {symbol_pairs[i][1]} linear.png')
    plt.clf()
    plt.plot(consq,linewidth=1)
    #plt.plot(np.ones(2880) * consq_mean)
    plt.plot(np.ones(len(consq)) * consq[0])
    plt.plot(np.ones(len(consq)) * 0.3 * consq_std + consq[0])
    plt.plot(np.ones(len(consq)) * (-0.3) * consq_std + consq[0])
    plt.plot(np.ones(len(consq)) * 2 * 0.3 * consq_std + consq[0])
    plt.plot(np.ones(len(consq)) * 2 * (-0.3) * consq_std + consq[0])
    #consq_mean = 0.00157777
    #plt.plot(np.arange(2880),(profit_line - np.mean(profit_line)) / np.std(profit_line)/100)
    #plt.plot(np.arange(2880), (amount_list_test - np.mean(amount_list_test)) / max(amount_list_test)/100)
    plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbol_pairs[i][0]} - {symbol_pairs[i][1]} coint.png')
plt.show()











