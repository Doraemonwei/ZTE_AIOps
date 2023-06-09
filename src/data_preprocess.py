# -*- coding: utf-8 -*-
# @Time : 2023/5/3 15:03
# @Author : Lanpangzi
# @File : data_preprocess.py

import numpy as np
# 包括数据分析 以及 读取 csv表格文件并以pandas格式返回
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._base import Bunch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_csv_1 = r'data/一阶段/train.csv'

train_dataframe_1 = pd.read_csv(train_csv_1, header=0)  # 表头是第一行
# train_dataframe_1.fillna(0, inplace=True)  # 将所有的nan替换成0

test_csv1 = r'data/一阶段/pre_contest_test1.csv'
test_dataframe_1 = pd.read_csv(test_csv1, header=0)  # 表头是第一行

test_csv2 = r'data/二阶段/pre_contest_test2.csv'
test_dataframe_2 = pd.read_csv(test_csv2, header=0)  # 表头是第一行

features_fill_speacial_num = ["feature{}".format(i) for i in range(0, 105) if
                              i not in [2, 4, 23, 30, 36, 38, 41, 48, 58, 64, 79]]

features_not_delete = ["feature{}".format(i) for i in range(0, 105) if
                       i not in [17, 70, 86]]

# 填充缺失数据
for column in list(train_dataframe_1.columns[train_dataframe_1.isnull().sum() > 0]):
    # 对于大部分都是相同的数的列，缺失的值使用众数填充，其他的用平均数填充
    mean_val = train_dataframe_1[column].mean()  # 平均数
    mode_val = train_dataframe_1[column].median()  # 众数,这里使用中位数替代众数，他们是一样的
    if column in features_fill_speacial_num:
        train_dataframe_1[column].fillna(mode_val, inplace=True)
    else:
        train_dataframe_1[column].fillna(mean_val, inplace=True)

for column in list(test_dataframe_1.columns[test_dataframe_1.isnull().sum() > 0]):
    mean_val = test_dataframe_1[column].mean()
    mode_val = test_dataframe_1[column].median()# 众数
    if column in features_fill_speacial_num:
        test_dataframe_1[column].fillna(mode_val, inplace=True)
    else:
        test_dataframe_1[column].fillna(mean_val, inplace=True)

for column in list(test_dataframe_2.columns[test_dataframe_2.isnull().sum() > 0]):
    # 用均值填充
    mean_val = test_dataframe_2[column].mean()
    mode_val = test_dataframe_2[column].median()  # 众数
    if column in features_fill_speacial_num:
        test_dataframe_2[column].fillna(mode_val, inplace=True)
    else:
        test_dataframe_2[column].fillna(mean_val, inplace=True)


# 一二阶段的训练数据集合是一样的
# train_csv_2 = r'data/二阶段/train.csv'
# train_dataframe_2 = pd.read_csv(train_csv_2, header=0)  # 表头是第一行


def load_rbb(df):
    """
    获取训练
    :return:
    """
    data_csv = df
    rbb = Bunch()
    rbb.data = _get_rbbdata(data_csv)
    rbb.target = _get_rbbtarget(data_csv)
    rbb.DESCR = _get_rbbdescr(data_csv)
    rbb.feature_names = _get_feature_names()
    rbb.target_names = _get_target_names()
    return rbb


def _get_rbbdata(data):
    """
    获取双色球特征值
    :return:
    """
    # 直接不使用那3个全是0的特征
    data_r = data.loc[:, features_not_delete]
    data_np = np.array(data_r)
    return data_np


def _get_rbbtarget(data):
    """
    获取双色球目标值
    :return:
    """
    data_b = data.iloc[:, 106:107]
    data_np = np.array(data_b)
    return data_np


def _get_rbbdescr(data):
    """
    获取数据集描述
    :return:
    """
    text = "本数据集为服务器，样本数量：{}；" \
           "特征数量：{}；目标值数量：{}；无缺失数据" \
           "".format(data.index.size, data.columns.size - 2, 1)
    return text


def _get_feature_names():
    """
    获取特征名字
    :return:
    """
    fnames = features_not_delete
    return fnames


def _get_target_names():
    """
    获取目标值名称
    :return:
    """
    tnames = ["label"]
    return tnames


if __name__ == '__main__':
    data = load_rbb(train_dataframe_1)
    # print(data)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    score = []
    max_f1 = 0
    k_value = -1
    for K in range(40):
        K_value = K + 1
        knn = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
        knn.fit(X_train, y_train.ravel())
        y_pred = knn.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro') * 100
        score.append(f1)
        if f1 > max_f1:
            max_f1 = f1
            k_value = K_value
    print('最好的k:{}，此时f1={}'.format(k_value, max_f1))
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 41), score, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('The Learning curve')
    plt.xlabel('K Value')
    plt.ylabel('Score')
    plt.show()
