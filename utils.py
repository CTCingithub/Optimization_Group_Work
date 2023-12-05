'''
Author: CTC 2801320287@qq.com
Date: 2023-11-24 12:38:56
LastEditors: CTC_322 2310227@tongji.edu.cn
LastEditTime: 2023-12-05 16:58:24
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from datetime import datetime, timedelta


def Time2SerialNum(
    YEAR, MONTH, DAY, HOUR, MIN=0, SEC=0, START=datetime(2017, 1, 1, 0, 0, 0)
):
    # ����һ����ʱ��ת��Ϊ����Сʱ��ʱ��˳��ĺ���
    # ? Ĭ��2017.01.01.0.0.0Ϊ��ʼʱ��
    return int(
        (datetime(YEAR, MONTH, DAY, HOUR, MIN, SEC) - START).total_seconds() / 3600
    )


def SerialNum2Time(SERIAL_NUM, START=datetime(2017, 1, 1, 0, 0, 0)):
    # ����һ��������Сʱ��ʱ��˳��ת��Ϊʱ��ĺ���
    # ? Ĭ��2017.01.01.0.0.0Ϊ��ʼʱ��
    return START + timedelta(hours=SERIAL_NUM)
