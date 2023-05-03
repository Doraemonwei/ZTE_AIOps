# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:00
# @Author : Lanpangzi
# @File : test.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets._base import Bunch
import numpy as np

train_csv_1 = r'data/一阶段/train.csv'

train_dataframe_1 = pd.read_csv(train_csv_1, header=0)  # 表头是第一行

al_features = ["feature{}".format(i) for i in range(0, 105)]
for f_name in al_features:
    f = train_dataframe_1[f_name].unique()
    if 0 in f:
        print(f_name)
        print(set(f))

labels = train_dataframe_1['label'].unique()
print(set(labels))