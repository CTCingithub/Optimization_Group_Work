'''
Author: CTC 2801320287@qq.com
Date: 2023-11-24 12:38:56
LastEditors: CTC 2801320287@qq.com
LastEditTime: 2023-11-24 12:45:05
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from datetime import datetime, timedelta


def Time2SerialNum(
    YEAR, MONTH, DAY, HOUR, MIN, SEC, START=datetime(2017, 1, 1, 0, 0, 0)
):
    # 构造一个将时间转化为按照小时的时间顺序的函数
    # ? 默认2017.01.01.0.0.0为起始时间
    return int(
        (datetime(YEAR, MONTH, DAY, HOUR, MIN, SEC) - START).total_seconds() / 3600
    )


def SerialNum2Time(SERIAL_NUM, START=datetime(2017, 1, 1, 0, 0, 0)):
    # 构造一个将按照小时的时间顺序转换为时间的函数
    # ? 默认2017.01.01.0.0.0为起始时间
    return START + timedelta(hours=SERIAL_NUM)
