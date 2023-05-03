# -*- coding: utf-8 -*-
# @Time : 2023/5/3 17:16
# @Author : Lanpangzi
# @File : 朴素贝叶斯.py
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from data_preprocess import *


data = load_rbb(train_dataframe_1)
# print(data)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy of SVM using optimized parameters ", accuracy_score(y_test, y_pred) * 100)
print("Report : ", classification_report(y_test, y_pred))
