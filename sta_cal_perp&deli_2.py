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
    def __init__(self, open_price: float, close_price: float = 0.0, amount: float = 0.0, direction: int = 1) -> None:
        self.open_price = open_price
        self.close_price = close_price
        self.amount = amount
        self.direction = direction

    def get_one_roi(self, now_price: float) -> float:
        if self.direction == 1:
            return (now_price - self.open_price) / self.open_price * self.amount - 0 * 0.0004 * self.amount
        elif self.direction == -1:
            return (self.open_price - now_price) / self.open_price * self.amount - 0 * 0.0004 * self.amount

def get_all_position_ROI(now_price: float, pos_list: list) -> float:
    roi = 0.0
    for pos in pos_list:
        roi += pos.get_one_roi(now_price)
    return roi

#@retry(stop=stop_after_delay(15))
def get_info(symbol='BTCUSDT', limit=1500, endTime=int(time.time() * 1000)):
    global binance
    binance = BINANCE()
    interval = "3m"
    # limit = "73"
    output = []
    # klines = []
    title_name = ['date', 'open', 'high', 'low', 'close']
    while limit - 1500 > 0:
        klines = []
        number = 0
        limit = limit - 1500
        number = number + 1500
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
            endTime = 2 * klines[0]['date'] - klines[1]['date']
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

symbol_perp = 'BTCUSDT'
symbols_deli = ['BTCUSDT_221230', 'BTCUSDT_230331', 'BTCUSDT_230630', 'BTCUSDT_230929', 'BTCUSDT_231229']
end_time = [1672387200000, 1680249600000, 1688112000000, 1695974400000, 1698112962866]
#1664524800000  2022-09-30  1656057600000   2022-06-24
kline_num = int(90 * 24 * 4 * 5)
kline_num_list_perp = [int((1672387200000-1664524800000)/1000/180),int((1680249600000-1672387200000)/1000/180),
                       int((1688112000000-1680249600000)/1000/180),int((1695974400000-1688112000000)/1000/180),
                       int((1698112962866-1695974400000)/1000/180)]
# kline_num_list_deli = [int((1672387200000-1664524800000)/1000/180)-37860,int((1680249600000-1664524800000)/1000/180)-39360,
#                        int((1688112000000-1672387200000)/1000/180)-39360,int((1695974400000-1680249600000)/1000/180)-39360]
kline_num_list_deli = [int((1672387200000-1664524800000)/1000/180),int((1680249600000-1672387200000)/1000/180),
                       int((1688112000000-1680249600000)/1000/180),int((1695974400000-1688112000000)/1000/180),
                       int((1698112962866-1695974400000)/1000/180)]


data = {}
b = False
for i in tqdm(range(len(symbols_deli))):
    get_info(symbol_perp, kline_num_list_perp[i], end_time[i])
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312')  # gb2312
    data_symbol_perp_close = data_symbol['close'].copy()

    if b :
        data[symbol_perp] = data[symbol_perp].append(data_symbol_perp_close)
    else:
        data[symbol_perp] = data_symbol_perp_close
        b = True
    get_info(symbols_deli[i], kline_num_list_deli[i], end_time[i])
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312')  # gb2312
    data_symbol_deli_close = data_symbol['close'].copy()
    data[symbols_deli[i]] = data_symbol_deli_close

for key in data.keys():
    #plt.plot(data[key])
    ind = data[key].index
    y_x = data[key]-data[symbol_perp].loc[ind]
    plt.plot(y_x)
plt.show()


