# coding: UTF-8
import time
import pandas
import pandas as pd
import torch
import numpy as np
from train_eval import train, init_network,draw_loss,draw_accuracy,predict
from importlib import import_module
from models.bert import Config,Model
from utils import build_dataset, build_iterator, get_time_dif
from pandasrw import load

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    config = Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data,test_pred_data= build_dataset(config)  ##构建词表，转化训练，测试，验证，预测数据集
    train_iter = build_iterator(train_data, config)##构建训练集迭代器
    dev_iter = build_iterator(dev_data, config)##构建训练集迭代器
    test_iter = build_iterator(test_data, config)##构建训练集迭代器
    test_pred_iter = build_iterator(test_pred_data, config)  ##构建预测集迭代器
    time_dif = get_time_dif(start_time)##记录时间
    print("Time usage:", time_dif)
    # 训练
    model = Model(config).to(config.device)##放在GPU上加速
    Train_Acc, Train_loss, Val_Acc, Val_loss,batches = train(config, model, train_iter, dev_iter, test_iter)  ##模型训练，返回每次结果
    # 调用画图函数，画训练集和验证集准确率和损失变化曲线图
    draw_loss(config,Train_loss, Val_loss,batches,show=False)
    draw_accuracy(config,Train_Acc, Val_Acc,batches,show=False)
    ###预测结果
    results=predict(model,test_pred_iter)
    # ##读取数据，添加标签
    data=load('C:/Users/xc/Desktop/data/test.csv',engine='pandas')
    data['predict']=results
    data.to_csv('C:/Users/xc/Desktop/data/test.csv') ##保存数据


