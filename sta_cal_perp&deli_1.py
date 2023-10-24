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


symbols = ['BTCUSDT', 'BTCUSDT_230929']

print(symbols)

data = {}
data_matric = []
data_org = {}
#kline_num = int(12 * 24 * 4*3)
kline_num = int(90 * 24 * 4*5)
i = 0
# endTime = int(time.time() * 1000)
endTime = 1695974400000     #230929
# endTime = 1688112000000     #230630
# endTime = 1680249600000     #230331
# endTime = 1672387200000     #221230
for n in tqdm(range(len(symbols))):
    # print(f'{round(i/len(symbols)*100,3)}%')
    get_info(symbols[i], kline_num, endTime)
    data_symbol = pd.read_csv('kline_data.csv', index_col=0, encoding='gb2312')  # gb2312
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


def back_test(long_ma_period: int,pos_amount: int,initial_amount :int,ret :list[float]) -> list[float]:
    df_data_org = pd.read_csv('kline_data_org.csv', encoding='gb2312')  # gb2312
    symbols = list(df_data_org.columns)
    y = np.array(df_data_org.loc[:, symbols[0]])
    x = np.array(df_data_org.loc[:, symbols[1]])
    MA_100 = []
    MA_long = []
    # long_ma_period = 200
    std_100 = []
    x_y = x - y
    for i in range(499, len(x)):
        ma = np.mean(x_y[i - 500 + 1:i + 1])
        st = np.std(x_y[i - 500 + 1:i + 1])
        MA_100.append(ma)
        std_100.append(st)
    MA_100 = np.array(MA_100)
    std_100 = np.array(std_100)

    for i in range(long_ma_period - 1, len(x)):
        ma = np.mean(x_y[i - long_ma_period + 1:i + 1])
        MA_long.append(ma)
    MA_long = np.array(MA_long)

    revenew = 0
    result = []
    pos = 0
    posx_list = []
    posy_list = []
    open_index = 0
    close_index = 0
    temp = []
    total_pos_amount = 0
    max_pos_amount = 0
    can_open = True
    fee = 0
    total_fee = 0
    result_last_value = 0
    for i in range(long_ma_period - 1, len(x)):
        if x_y[i] >= 0:  # and MA_long[i-long_ma_period] >= MA_100[i-long_ma_period+len(MA_100)-len(MA_long)]
            if x_y[i] > MA_100[i - 499] + 3 * std_100[i - 499] and can_open and len(posx_list) < pos_amount:
                open_index = i if pos == 0 else open_index
                posx = Position(x[i], amount=initial_amount/pos_amount, direction=-1)
                posy = Position(y[i], amount=initial_amount/pos_amount, direction=1)
                posx_list.append(posx)
                posy_list.append(posy)
                total_pos_amount += 1
                pos += 1
                temp.append(0.0)
                result.append(result_last_value+temp[-1])
                max_pos_amount = max(max_pos_amount, len(posx_list))
                can_open = False
                print(f'Open! i = {i}')

            elif x_y[i] < MA_100[i - 499] - 3 * std_100[i - 499] and pos != 0:
                r_all = (get_all_position_ROI(now_price=x[i], pos_list=posx_list) +
                         get_all_position_ROI(now_price=y[i], pos_list=posy_list))
                amount = posx_list[0].amount
                if r_all < (len(posx_list) + len(posy_list)) * amount * 0.0005 * 2:
                    temp.append(r_all)
                    result.append(result_last_value + temp[-1])
                    continue
                fee = (len(posx_list) + len(posy_list)) * amount * 0.0005 * 2
                total_fee -= fee
                print(f'fee = {fee}')
                temp.append(r_all)
                result.append(result_last_value+temp[-1])
                result_last_value = result[-1]-fee
                posx_list.clear()
                posy_list.clear()
                pos = 0
                can_open = True
                print(f'Close! i = {i}')
            elif pos != 0:
                r_all = (get_all_position_ROI(now_price=x[i], pos_list=posx_list) +
                         get_all_position_ROI(now_price=y[i], pos_list=posy_list))
                temp.append(r_all)
                result.append(result_last_value+temp[-1])
            elif pos == 0:
                temp.append(0.0)
                result.append(result_last_value+temp[-1])
            if x_y[i] < MA_100[i - 499] + 1 * std_100[i - 499] and not can_open:
                can_open = True
        elif 0:
            if x_y[i] < MA_100[i - 99] - 1 * std_100[i - 99] and can_open:
                open_index = i if pos == 0 else open_index
                posx = Position(x[i])
                posy = Position(y[i])
                posx_list.append(posx)
                posy_list.append(posy)
                total_pos_amount += 1
                pos += 1
                temp.append(0.0)
                max_pos_amount = max(max_pos_amount, len(posx_list))
                can_open = False

            elif x_y[i] > MA_100[i - 99] + 3 * std_100[i - 99] and pos != 0:
                r_all = (get_all_position_ROI(now_price=x[i], pos_list=posx_list) +
                         get_all_position_ROI(now_price=y[i], pos_list=posy_list))
                if r_all < 0:
                    temp.append(r_all)
                    continue
                close_index = i
                temp.append(r_all)
                posx_list.clear()
                posy_list.clear()
                pos = 0
                can_open = True
            elif pos != 0:
                r_all = (get_all_position_ROI(now_price=x[i], pos_list=posx_list) +
                         get_all_position_ROI(now_price=y[i], pos_list=posy_list))
                temp.append(r_all)
            elif pos == 0:
                temp.append(0.0)
            if x_y[i] < MA_100[i - 99] + 1 * std_100[i - 99] and not can_open:
                can_open = True
        else:
            temp.append(0.0)
            result.append(result_last_value + temp[-1])

    '''roi = 0
    for i in range(len(temp)):
        if i == 0:
            result.append(temp[i])
        else:
            if temp[i] == 0 and temp[i - 1] != 0:
                roi += temp[i - 1]
            if temp[i] == 0:
                result.append(roi)
            else:
                result.append(roi + temp[i])'''

    # plt.plot(y)
    # plt.plot(x)
    plt.plot(x_y)
    plt.plot([i for i in range(499, len(x_y))], MA_100)
    plt.plot([i for i in range(499, len(x_y))], MA_100 - std_100)
    plt.plot([i for i in range(499, len(x_y))], MA_100 + std_100)
    plt.plot([i for i in range(499, len(x_y))], MA_100 - 3 * std_100)
    plt.plot([i for i in range(499, len(x_y))], MA_100 + 3 * std_100)
    plt.plot([i for i in range(long_ma_period - 1, len(x_y))], np.array(result))
    plt.plot([i for i in range(long_ma_period - 1, len(x_y))], np.array(temp))
    # plt.plot([i for i in range(long_ma_period-1,len(x_y))],MA_long)
    print(total_pos_amount)
    print(max_pos_amount)
    print('=' * 35)
    print(f'pos_amount = {pos_amount}')
    print(f'${round(result[-1], 3)}')
    print(f'${round(result[-1]+total_fee, 3)}')
    print(f'fee ${round(total_fee, 3)}')
    print(f'${round(result[-1] * amount/pos_amount, 3)}')
    print(f'{round(result[-1] * amount/pos_amount / 34 / 6 / 2 * 100, 3)}%')
    ret.append(round(result[-1] * amount/pos_amount / 34 / 6 / 2 * 100, 3))
    print(f'${round(result[-1] * amount/pos_amount + total_fee, 3)}')
    print(f'{round((result[-1] * amount/pos_amount + total_fee) / 34 / 6 / 2 * 100, 3)}%')
    print(f'{round(result[-1] * amount/pos_amount / 10 * 100, 3)}%')
    print(f'{round((result[-1] * amount/pos_amount + total_fee) / 10 * 100, 3)}%')
    print('=' * 35)
    plt.show()
    # pos_amount = 24
    # $467.14
    # $37.108
    return ret

result = []
x = []
for pos_amount in range(24, 25):
    result.append(back_test(500,pos_amount,int(100/7*125),[]))
    x.append(pos_amount)

# plt.clf()
# plt.plot(x,result)
# plt.show()

# 15m
# 504
# 50

# 5m
# 1356
# 129

# 3m
# 2166
# 201
