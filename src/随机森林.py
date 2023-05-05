# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:52
# @Author : Lanpangzi
# @File : 随机森林.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json
from data_preprocess import *
from collections import defaultdict,Counter

data = load_rbb(train_dataframe_1)
# print(data)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

rfc = RandomForestClassifier(random_state=0,
                             criterion='entropy',
                             n_estimators=25,
                             )
rfc.fit(X_train, y_train.ravel())

y_pred = rfc.predict(X_test)
print("Report : ", classification_report(y_test, y_pred))


def cal_n_estimators():
    superpa = []
    for i in range(150):
        rfc = RandomForestClassifier(random_state=0,
                                     criterion='entropy',
                                     n_estimators=i + 1,
                                     n_jobs=-1)
        rfc.fit(X_train, y_train.ravel())
        # y_pred = rfc.predict(X_test)
        # f1 = f1_score(y_test, y_pred, average='macro') * 100
        sc = rfc.score(X_test,y_test)
        superpa.append(sc)
        print(i)
    print(max(superpa), superpa.index(max(superpa)))
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 151), superpa)
    plt.show()

# cal_n_estimators()

def get_answer():
    # 阶段1提交
    test_data = load_rbb(test_dataframe_1)
    X = test_data.data
    scaler.fit(X)
    X_train = scaler.transform(X)
    test_y_pre = rfc.predict(X_train)
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
    test_y_pre = rfc.predict(X_train)
    ans1 = dict()
    for i in range(0, 1047):
        ans1[str(i)] = int(test_y_pre[i])
    with open("submit2.json", "w") as f:
        json.dump(ans1, f)
        print("加载入文件完成...")

    cnt = Counter(test_y_pre)
    print(cnt)
get_answer()


