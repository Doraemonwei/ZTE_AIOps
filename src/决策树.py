# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:52
# @Author : Lanpangzi
# @File : 决策树.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from data_preprocess import *
import json
from collections import defaultdict, Counter

data = load_rbb(train_dataframe_1)
# print(data)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=6)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# 阶段2提交
test_data = load_rbb(test_dataframe_2)
X = test_data.data
scaler.fit(X)
X_train = scaler.transform(X)
test_y_pre = classifier.predict(X_train)
ans1 = dict()
for i in range(0, 1047):
    ans1[str(i)] = int(test_y_pre[i])
with open("submit.json", "w") as f:
    json.dump(ans1, f)
    print("加载入文件完成...")

cnt = Counter(test_y_pre)
print(cnt)

