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

symbols = ['ENJUSDT', 'RDNTUSDT']#['MANAUSDT', 'MINAUSDT']
print(symbols)
print(f"total calculate times = {len(symbols)*(len(symbols)-1)/2}")
check_times = 1
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
    if len(data_symbol_close) == kline_num:
        data_org[symbols[i]] = list(data_symbol_close.values)
    i = i + 1
    time.sleep(0.3)
def back_test(data_org,check_times,grid):
    print(f'第{check_times}次检查平稳性')
    '''for n in tqdm(range(len(symbols))):
        #print(f'{round(i/len(symbols)*100,3)}%')
        get_info(symbols[i], kline_num,endTime)
        data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312') # gb2312
        data_symbol_close = data_symbol['close'].copy()
        if len(data_symbol_close) == kline_num:
            data_org[symbols[i]] = list(data_symbol_close.values)
        i = i + 1
        time.sleep(0.3)'''
    y = np.array(data_org[symbols[0]])
    x = np.array(data_org[symbols[1]])
    y_x = np.array(y-x)
    dfgls_consq = DFGLS(y_x)
    print('='*60)
    print(f'T-value = {dfgls_consq.stat}')
    print(f'P-value = {dfgls_consq.pvalue}')
    print(f'critical value = {dfgls_consq.critical_values}')
    print('='*60)
    if dfgls_consq.stat > dfgls_consq.critical_values['1%'] or dfgls_consq.pvalue >= 0.01:
        print('警戒！平稳性即将越界！请及时查看！'*6)
    m = np.mean(y_x[:-1000])
    s = np.std(y_x[:-1000])
    # plt.plot(y_x)
    # plt.plot([m for i in range(len(y_x))])
    # plt.plot([m + 3*s for i in range(len(y_x))])
    # plt.plot([m - 3*s for i in range(len(y_x))])
    # plt.plot([m + 4*s for i in range(len(y_x))])
    # plt.plot([m - 4*s for i in range(len(y_x))])
    # plt.plot([m + s for i in range(len(y_x))])
    # plt.plot([m - s for i in range(len(y_x))])
    # plt.show()

    # for j in tqdm(range(3*60)):
    #     time.sleep(1)
    check_times += 1
    m = np.mean(y_x)
    st = np.std(y_x)
    each_roi = abs(grid*st/m)
    consq_amount = m
    touch_amount = 0
    pair_list = []
    total_pair_amount = 0
    rev = 0
    fee = 0
    init_marg = 100
    usable_marg = round(100 / (3 + (4 * st - m) / 0.2357 * 50), 4)*3
    print(f'最大可用保证金（4std)${usable_marg}')
    print(f'最大可用保证金比率（4std){round(usable_marg/init_marg,4)*100}%')
    print(f'止损后剩余保证金（4std)${init_marg - usable_marg / 3 * (4 * st - m) / 0.2357 * 50}')
    for n in range(len(y_x)):
        if y_x[n]-consq_amount > grid*st:
            consq_amount += grid*st
            touch_amount += 1
            if consq_amount not in pair_list:
                pair_list.append(consq_amount)
            elif consq_amount in pair_list:
                pair_list.remove(consq_amount)
                total_pair_amount += 1
                rev += abs(each_roi*-1/(consq_amount-m-1)) if consq_amount < m else abs(each_roi*-1/(consq_amount-m+1))
                fee += abs(each_roi*-1/(consq_amount-m-1))*0.0005*4 \
                    if consq_amount < m else abs(each_roi*-1/(consq_amount-m+1))*0.0005*4
                # rev += abs(each_roi) if consq_amount < m else abs(each_roi)
                # fee += 0.0005 * 4
        elif consq_amount-y_x[n] > grid*st:
            consq_amount -= grid*st
            touch_amount += 1
            if consq_amount not in pair_list:
                pair_list.append(consq_amount)
            elif consq_amount in pair_list:
                pair_list.remove(consq_amount)
                total_pair_amount += 1
                rev += abs(each_roi*-1/(consq_amount-m-1)) if consq_amount < m else abs(each_roi*-1/(consq_amount-m+1))
                fee += abs(-1 / (consq_amount - m - 1)) * 0.0005 * 4 \
                    if consq_amount < m else abs(-1 / (consq_amount - m + 1)) * 0.0005 * 4
                # rev += abs(each_roi) if consq_amount < m else abs(each_roi)
                # fee += 0.0005 * 4
    return touch_amount,total_pair_amount,each_roi,rev,fee

result = []
result_pair = []
result_each_roi = []
result_rev = []
result_fee = []
for i in range(4500,6500):
    touch_amount,total_pair_amount,each_roi,rev,fee = back_test(data_org,1,i/100000)
    result.append(touch_amount)
    result_pair.append(total_pair_amount)
    result_each_roi.append(each_roi)
    result_rev.append(rev)
    result_fee.append(fee)
    print('='*35)
    print(f'i = {i}')
    print(f'touch_amount = {touch_amount}')
    print(f'total_pair_amount = {total_pair_amount}')
    print(f'each_roi = {each_roi}')
    print('='*35)
result = np.array(result)
result_pair = np.array(result_pair)
result_each_roi = np.array(result_each_roi)
result_rev = np.array(result_rev)
# plt.plot(np.array([i for i in range(1,500)])/1000,result,label='result')
# plt.plot(np.array([i for i in range(1,500)])/1000,result_pair,label='result_pair')
# plt.plot(np.array([i for i in range(1,1000)])/1000,result_each_roi,label='result_each_roi')
plt.plot(np.array([i for i in range(4500,6500)])/100000,result_rev,label='result_rev')
# plt.plot(np.array([i for i in range(1,1000)])/1000,result_each_roi*0.0005*4,label='result')
# plt.plot(np.array([i for i in range(1,1000)])/1000,-result_pair*0.0005*4+result_rev,label='result')
plt.plot(np.array([i for i in range(4500,6500)])/100000,-result_pair*0.0005*4+result_pair*result_each_roi,label='linear pos with fee')
plt.plot(np.array([i for i in range(4500,6500)])/100000,result_pair*result_each_roi,label='linear pos without fee')
plt.plot(np.array([i for i in range(4500,6500)])/100000,result_fee,label='fee')
plt.plot(np.array([i for i in range(4500,6500)])/100000,result_rev-result_fee,label='result_rev with fee')
print((list(-result_pair*0.0005*4+result_pair*result_each_roi).index(max(-result_pair*0.0005*4+result_pair*result_each_roi))+1)/1000)
plt.legend()
print((list(-result_pair*(0.0005+0.000347465)*4+result_pair*result_each_roi)
       .index(max(-result_pair*(0.0005+0.000347465)*4+result_pair*result_each_roi))+1)/1000)
plt.grid(True)
plt.show()

#0.05948



