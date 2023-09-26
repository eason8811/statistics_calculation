import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3
from statsmodels.tsa.stattools import adfuller
from tenacity import *
from tqdm import tqdm

from binance_API_USDT import BINANCE

urllib3.disable_warnings()
s = requests.session()
s.keep_alive = False

headers_list = [
    {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G955U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/87.0.4280.77 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.188 Safari/537.36 CrKey/1.54.250320',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (BB10; Touch) AppleWebKit/537.10+ (KHTML, like Gecko) Version/10.0.9.2372 Mobile Safari/537.10+',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (PlayBook; U; RIM Tablet OS 2.1.0; en-US) AppleWebKit/536.2+ (KHTML like Gecko) Version/7.2.1.0 Safari/536.2+',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.3; en-us; SM-N900T Build/JSS15J) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.1; en-us; GT-N7100 Build/JRO03C) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.0; en-us; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.0; SM-G950U Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G965U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.111 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.1.0; SM-T837A) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.80 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; en-us; KFAPWI Build/JDQ39) AppleWebKit/535.19 (KHTML, like Gecko) Silk/3.13 Safari/535.19 Silk-Accelerated=true',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.4.2; en-us; LGMS323 Build/KOT49I.MS32310c) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 550) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2486.0 Mobile Safari/537.36 Edge/14.14263',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Moto G (4)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 10 Build/MOB31T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 5X Build/OPR4.170623.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.1.1; Nexus 6 Build/N6F26U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 7 Build/MOB30X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows Phone 8.0; Trident/6.0; IEMobile/10.0; ARM; Touch; NOKIA; Lumia 520)',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (MeeGo; NokiaN9) AppleWebKit/534.13 (KHTML, like Gecko) NokiaBrowser/8.5.0 Mobile Safari/534.13',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 9; Pixel 3 Build/PQ1A.181105.017.A1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.158 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; Pixel 4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 11; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Pixel 2 XL Build/OPD1.170816.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'content-type': 'application/json'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1',
        'content-type': 'application/json'
    }
]
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

def minus(symbols,i):
    for j in range(i+1,len(symbols)):
        data_symbol_minus_list = []
        for n in range(len(data[symbols[i]])):
            data_symbol_minus_list.append(data[symbols[i]][n] - data[symbols[j]][n])
        avg = np.ones(len(data_symbol_minus_list))*np.average(data_symbol_minus_list)
        var = np.ones(len(data_symbol_minus_list))*np.var(data_symbol_minus_list, ddof = 1)
        data_symbol_minus[f'{symbols[i]} - {symbols[j]}'] = data_symbol_minus_list
        plt.clf()
        plt.plot(data_symbol_minus_list)
        plt.plot(avg,"-")
        plt.plot(var,"*")
        plt.savefig(f'D:\\学校文件\\Python\\fig\\{symbols[i]} - {symbols[j]}.png')
        #plt.pause(0.2)

binance = BINANCE()
exchanges_info = binance.IO('GET','/fapi/v1/exchangeInfo',{})

symbols = []
print("Check symbols")
for i in range(len(exchanges_info['symbols'])):
    if (exchanges_info['symbols'][i]['status'] == 'TRADING') and (exchanges_info['symbols'][i]['symbol'][-4:] != 'BUSD') :
        symbols.append(exchanges_info['symbols'][i]['symbol'])
print("Symbols check finished")
symbols = symbols.copy()
print(symbols)
print(f"total calculate times = {len(symbols)*(len(symbols)-1)/2}")

data = {}
data_matric = []
data_org = {}
kline_num = int(2880*5)
i = 0
endTime = int(time.time()*1000)
for n in tqdm(range(len(symbols))):
    #print(f'{round(i/len(symbols)*100,3)}%')
    get_info(symbols[i], kline_num,endTime)
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312') # gb2312
    data_symbol_close = data_symbol['close'].copy()
    if (len(data_symbol_close) < kline_num) and (i < len(symbols)):
        symbols.remove(symbols[i])
        continue
    data_symbol_close_rate = []
    data_symbol_close_rate2one = []
    for j in range(len(data_symbol_close)):
        if j == 0:
            data_symbol_close_rate.append(0.0)
        else:
            data_symbol_close_rate.append((data_symbol_close[data_symbol_close.index[j]]-data_symbol_close[data_symbol_close.index[j-1]])
                                          / data_symbol_close[data_symbol_close.index[j-1]] + data_symbol_close_rate[-1])

    '''#归一化
    j = 0
    print("数据归一化......")
    for j in tqdm(range(len(data_symbol_close_rate))):
        Max = max(data_symbol_close_rate)
        Min = min(data_symbol_close_rate)
        if Max-Min == 0 :
            print(data_symbol_close_rate)
            print(data_symbol_close)
            print(symbols[i])
        mean = sum(data_symbol_close_rate) / len(data_symbol_close_rate)
        Max1 = np.max(np.abs(data_symbol_close_rate))
        data_symbol_close_rate2one.append((data_symbol_close_rate[j] - mean) / Max1)'''
    result = adfuller(data_symbol_close)
    if result[0] < result[4]['1%']:
        symbols.remove(symbols[i])
        continue
    if len(data_symbol_close_rate2one) == kline_num:
        data[symbols[i]] = data_symbol_close_rate2one
        data_org[symbols[i]] = list(data_symbol_close.values)
        data_matric.append(data_symbol_close_rate2one)
    i = i + 1
    time.sleep(0.3)

#plt.ion()

data_symbol_minus = {}
print('=================================================')

#print(data_symbol_minus)
#plt.ioff()


title_name = symbols
with open('kline_data_symbol_close_rate2one.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 1.创建DicetWriter对象
    dictWriter = csv.DictWriter(file_obj, title_name)
    # 2.写表头
    dictWriter.writeheader()
    # 3.写入数据(一次性写入多行)
    output = []
    for i in range(kline_num):
        kline_of_symbol = {}
        for j in range(len(data.keys())):
            kline_of_symbol[list(data.keys())[j]] = data[list(data.keys())[j]][i]
        output.append(kline_of_symbol)
    dictWriter.writerows(output)
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
