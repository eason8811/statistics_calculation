import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3
from tenacity import *
from tqdm import tqdm
from arch.unitroot import DFGLS

from binance_API_USDT import BINANCE

urllib3.disable_warnings()
s = requests.session()
s.keep_alive = False

@retry(stop=stop_after_delay(15))
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

symbols = ['ENJUSDT', 'RDNTUSDT']
print(symbols)
print(f"total calculate times = {len(symbols)*(len(symbols)-1)/2}")
check_times = 1
while 1:
    data = {}
    data_matric = []
    data_org = {}
    kline_num = int(2880*5)
    i = 0
    endTime = int(time.time()*1000)
    print(f'第{check_times}次检查平稳性')
    for n in tqdm(range(len(symbols))):
        #print(f'{round(i/len(symbols)*100,3)}%')
        get_info(symbols[i], kline_num,endTime)
        data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312') # gb2312
        data_symbol_close = data_symbol['close'].copy()
        if len(data_symbol_close) == kline_num:
            data_org[symbols[i]] = list(data_symbol_close.values)
        i = i + 1
        time.sleep(0.3)
    y = np.array(data_org[symbols[0]])
    x = np.array(data_org[symbols[1]])
    y_x = (y-x)
    dfgls_consq = DFGLS(y_x)
    print('='*60)
    print(f'T-value = {dfgls_consq.stat}')
    print(f'P-value = {dfgls_consq.pvalue}')
    print(f'critical value = {dfgls_consq.critical_values}')
    print('='*60)
    if dfgls_consq.stat > dfgls_consq.critical_values['1%'] or dfgls_consq.pvalue >= 0.01:
        print('警戒！平稳性即将越界！请及时查看！'*6)
    # plt.plot(y_x)
    # plt.show()
    for j in tqdm(range(3*60)):
        time.sleep(1)
    check_times += 1