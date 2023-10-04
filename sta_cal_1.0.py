import matplotlib.pyplot as plt
from tqdm import tqdm
import requests
import urllib3
import csv
import time
from binance_API_USDT import BINANCE
from tenacity import *
import pandas as pd
import numpy as np

urllib3.disable_warnings()
s = requests.session()
s.keep_alive = False

class Position:
    def __init__(self, open_price :float, close_price :float = 0.0) -> None:
        self.open_price = open_price
        self.close_price = close_price
def get_all_position_ROI(now_price :float, pos_list :list, is_long :bool) -> float:
    roi = 0.0
    for pos in pos_list:
        roi += (now_price - pos.open_price)/pos.open_price if is_long else (pos.open_price - now_price)/pos.open_price
    return roi
#@retry(stop=stop_after_delay(15))
def get_info(symbol = 'BTCUSDT',limit = 1500,endTime = int(time.time()*1000)):
    global binance
    binance = BINANCE()
    interval = "15m"
    #limit = "73"
    output = []
    #klines = []
    title_name = ['date', 'open', 'high', 'low', 'close']
    while limit-1500 > 0:
        klines = []
        number = 0
        limit = limit-1500
        number = number+1500
        body = {
            "symbol": symbol,
            "interval": interval,
            "limit": number,
            "endTime": endTime
        }
        respond = binance.IO('GET', '/fapi/v1/klines', body)
        try:
            for i in range(len(respond)):
                kline = {}
                kline['date'] = respond[i][0]
                kline['open'] = respond[i][1]
                kline['high'] = respond[i][2]
                kline['low'] = respond[i][3]
                kline['close'] = respond[i][4]
                klines.append(kline)
            endTime = 2*klines[0]['date'] - klines[1]['date']
            klines = list(reversed(klines))
            output.extend(klines)
        except IndexError:
            print(limit)
    if limit > 0:
        klines = []
        body = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "endTime": endTime
        }
        respond = binance.IO('GET', '/fapi/v1/klines', body)
        try:
            for i in range(len(respond)):
                kline = {}
                kline['date'] = respond[i][0]
                kline['open'] = respond[i][1]
                kline['high'] = respond[i][2]
                kline['low'] = respond[i][3]
                kline['close'] = respond[i][4]
                klines.append(kline)
            klines = list(reversed(klines))
            output.extend(klines)
            output = list(reversed(output))
        except IndexError:
            print(limit)
    with open('kline_data.csv', 'w', encoding='utf-8', newline='') as file_obj:
        # 1.创建DicetWriter对象
        dictWriter = csv.DictWriter(file_obj, title_name)
        # 2.写表头
        dictWriter.writeheader()
        # 3.写入数据(一次性写入多行)
        dictWriter.writerows(output)

symbols = ['ETHUSDT','ETHUSDT_230929']

print(symbols)

data = {}
data_matric = []
data_org = {}
kline_num = int(90*24*4)
i = 0
#endTime = int(time.time()*1000)
endTime = 1695974400000
for n in tqdm(range(len(symbols))):
    #print(f'{round(i/len(symbols)*100,3)}%')
    get_info(symbols[i], kline_num,endTime)
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312') # gb2312
    data_symbol_close = data_symbol['close'].copy()

    data_org[symbols[i]] = list(data_symbol_close.values)
    i += 1
    time.sleep(0.3)

print('=================================================')

title_name = symbols
with open('kline_data_org.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 1.创建DicetWriter对象
    dictWriter = csv.DictWriter(file_obj, title_name)
    # 2.写表头
    dictWriter.writeheader()
    # 3.写入数据(一次性写入多行)
    output = []
    for i in range(kline_num):
        kline_of_symbol = {}
        for j in range(len(data_org.keys())):
            kline_of_symbol[list(data_org.keys())[j]] = data_org[list(data_org.keys())[j]][i]
        output.append(kline_of_symbol)
    dictWriter.writerows(output)

df_data_org = pd.read_csv('kline_data_org.csv', encoding='gb2312') # gb2312
symbols = list(df_data_org.columns)
y = np.array(df_data_org.loc[:,symbols[0]])
x = np.array(df_data_org.loc[:,symbols[1]])
MA_100 = []
std_100 = []
x_y = x-y
for i in range(99,len(x)):
    ma = np.mean(x_y[i-100+1:i])
    st = np.std(x_y[i-100+1:i])
    MA_100.append(ma)
    std_100.append(st)
MA_100 = np.array(MA_100)
std_100 = np.array(std_100)

revenew = 0
result = []
pos = 0
posx_list = []
posy_list = []
open_index = 0
close_index = 0
temp = []
total_pos_amount = 0
for i in range(99,len(x)):
    if x_y[i] > MA_100[i-99] + 1 * std_100[i-99]:
        open_index = i if pos == 0 else open_index
        posx = Position(x[i])
        posy = Position(y[i])
        posx_list.append(posx)
        posy_list.append(posy)
        total_pos_amount += 1
        pos += 1
        temp.append(0.0)

    elif x_y[i] < MA_100[i-99] - 3 * std_100[i-99] and pos != 0:
        r_all = (get_all_position_ROI(now_price=x[i], pos_list=posx_list, is_long=False) +
                 get_all_position_ROI(now_price=y[i], pos_list=posy_list, is_long=True))
        if r_all < 0:
            continue
        close_index = i
        temp.append(r_all)
        posx_list.clear()
        posy_list.clear()
        pos = 0
    elif pos != 0 :
        r_all = (get_all_position_ROI(now_price= x[i],pos_list= posx_list,is_long= False) +
                 get_all_position_ROI(now_price= y[i],pos_list= posy_list,is_long= True))
        temp.append(r_all)
    elif pos == 0:
        temp.append(0.0)

roi = 0
for i in range(len(temp)):
    if temp[i] == 0 and temp[i-1] != 0:
        roi += temp[i-1]
    if temp[i] == 0:
        result.append(roi)
    else:
        result.append(roi+temp[i])


# plt.plot(y)
# plt.plot(x)
plt.plot(x_y)
plt.plot([i for i in range(99,len(x_y))],MA_100)
plt.plot([i for i in range(99,len(x_y))],MA_100 - std_100)
plt.plot([i for i in range(99,len(x_y))],MA_100 + std_100)
plt.plot([i for i in range(99,len(x_y))],MA_100 - 2*std_100)
plt.plot([i for i in range(99,len(x_y))],MA_100 + 2*std_100)
plt.plot([i for i in range(99,len(x_y))],np.array(result))
plt.plot([i for i in range(99,len(x_y))],np.array(temp))
print(total_pos_amount)
plt.show()
