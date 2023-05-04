# -*- coding: utf-8 -*-
# @Time : 2023/5/3 19:06
# @Author : Lanpangzi
# @File : 神经网络.py
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class TrainDataset(Dataset):
    def __init__(self, filepath=r'data/一阶段/train.csv', use_mean_fill=True):
        """

        :param filepath: 导入的csv路径
        :param use_mean_fill: 是否使用均值填充缺失部分
        """
        print("reading {}".format(filepath))
        df = pd.read_csv(
            filepath, header=0, index_col=0,
            encoding='utf-8',
            names=["feature{}".format(i) for i in range(0, 105)] + ['label'],
            dtype=np.float32
        )
        if use_mean_fill:
            for column in list(df.columns[df.isnull().sum() > 0]):
                mean_val = df[column].mean()
                df[column].fillna(mean_val, inplace=True)

        # 对数据进行标准化
        names = ["feature{}".format(i) for i in range(0, 105)]
        for i in names:
            if i not in ['feature17', 'feature70', 'feature86']:
                df[i] = (df[i] - df[i].mean()) / df[i].std()

        print("the shape of data is {}".format(df.shape))
        features = df.iloc[:, :105].values
        labels = df.iloc[:, 105].values
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


al_dataset = TrainDataset()
train_size = int(len(al_dataset) * 0.8)
val_size = len(al_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(al_dataset, [train_size, val_size])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=False)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=512, shuffle=True, num_workers=0, drop_last=False)

net = nn.Sequential(
    nn.Linear(105, 50),  # 输入层与第一隐层结点数设置，全连接结构
    torch.nn.ReLU(),  # 第一隐层激活函数采用sigmoid
    nn.Linear(50, 50),  # 第一隐层与第二隐层结点数设置，全连接结构
    torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
    nn.Linear(50, 6),  # 第二隐层与输出层层结点数设置，全连接结构
    nn.Softmax(dim=1),  # 由于有两个概率输出，因此对其使用Softmax进行概率归一化，dim=1代表行归一化
)

net = net.to('cuda')
optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)  # 优化器使用adam，传入网络参数和学习率
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)
loss_func = torch.nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失函数

if __name__ == '__main__':
    # print(al_dataset[0])

    num_epoch = 10000  # 最大迭代更新次数
    for epoch in range(num_epoch):
        epoch_loss = 0
        for data in train_dataloader:
            net = net.to('cuda')
            features, labels = data
            features = features.to('cuda')
            labels = labels.to("cuda")
            y_p = net(features)  # 喂数据并前向传播
            loss = loss_func(y_p, labels.long())  # 计算损失
            epoch_loss += loss.data.item()
            '''
            PyTorch默认会对梯度进行累加，因此为了不使得之前计算的梯度影响到当前计算，需要手动清除梯度。
            pyTorch这样子设置也有许多好处，但是由于个人能力，还没完全弄懂。
            '''
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 计算梯度，误差回传
            optimizer.step()  # 根据计算的梯度，更新网络中的参数
        # print('第{}轮训练结束'.format(epoch))
        if epoch % 100 == 0:
            print('epoch: {}, loss: {}'.format(epoch, epoch_loss))
            # 检测在验证集上的表现
            val_epoch_loss = 0
            with torch.no_grad():
                for data in val_dataloader:
                    features, labels = data
                    features = features.to('cuda')
                    labels = labels.to("cuda")
                    y_p = net(features)  # 喂数据并前向传播
                    loss = loss_func(y_p, labels.long())  # 计算损失
                    val_epoch_loss += loss.data.item()
                print("-----------------------------")
                print("在验证集上的loss：{}".format(val_epoch_loss))
                net = net.to('cpu')
                pre = []
                labels = []
                for x, y in val_dataloader:
                    pre_ys = net(x)
                    pre += torch.max(pre_ys, dim=1)[1]
                    labels += y
                print("Report : ", classification_report(labels, pre))
        if epoch % 3000 == 0:
            print("************************************************************")
            print("lr: " + str(optimizer.state_dict()['param_groups'][0]['lr']))
            print("************************************************************")
        StepLR.step()  # 学习速率衰减
