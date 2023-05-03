# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:55
# @Author : Lanpangzi
# @File : 支持向量机.py
import json
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # "Support Vector Classifier"
from data_preprocess import *

data = load_rbb(train_dataframe_1)
# print(data)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='linear', C=10, gamma='scale', decision_function_shape='ovr')
model.fit(X_train, y_train)
svc_predictions = model.predict(X_test)
print("Accuracy of SVM using optimized parameters ", accuracy_score(y_test, svc_predictions) * 100)
print("Report : ", classification_report(y_test, svc_predictions))


# 阶段1提交
test_data = load_rbb(test_dataframe_1)
X = test_data.data
scaler.fit(X)
X_train = scaler.transform(X)
test_y_pre = model.predict(X_train)
ans1 = dict()
for i in range(0, 1005):
    ans1[str(i)] = int(test_y_pre[i])
with open("submit.json", "w") as f:
    json.dump(ans1, f)
    print("加载入文件完成...")

cnt = Counter(test_y_pre)
print(cnt)

# 阶段2提交
test_data = load_rbb(test_dataframe_2)
X = test_data.data
scaler.fit(X)
X_train = scaler.transform(X)
test_y_pre = model.predict(X_train)
ans1 = dict()
for i in range(0, 1047):
    ans1[str(i)] = int(test_y_pre[i])
with open("submit2.json", "w") as f:
    json.dump(ans1, f)
    print("加载入文件完成...")

cnt = Counter(test_y_pre)
print(cnt)



