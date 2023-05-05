# -*- coding: utf-8 -*-
# @Time : 2023/5/3 16:00
# @Author : Lanpangzi
# @File : test.py

features = ["feature{}".format(i) for i in range(0, 105) if i not in [2,4,23,30,36,38,41,48,58,64,79,17,70,86]]
print(features)