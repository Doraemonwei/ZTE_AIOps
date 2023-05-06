# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:00
# @Author : Lanpangzi
# @File : test.py
import pandas as pd

train_csv_1 = r'data/一阶段/train.csv'

train_dataframe_1 = pd.read_csv(train_csv_1, header=0)  # 表头是第一行