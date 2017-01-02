#!/usr/bin/python
#coding:utf8

#我的包
from mltoolkits import *
import myneural_network as mymodel

#非我的包
import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
  logger.debug('start...')

  #
  trainfile,labelfile = sys.argv[1:3]

  train_data = np.loadtxt(trainfile, delimiter = ',', dtype = np.float)
  label_data = np.loadtxt(labelfile, delimiter = ',', dtype = np.float)
  train_data = np.column_stack((train_data, label_data))
  np.random.shuffle(train_data)

  train_x = train_data[0:4000,0:-1]
  train_y = train_data[0:4000, -1]
  test_x = train_data[4000:,0:-1]
  test_y = train_data[4000:,-1]

  #显示图片随机抽取100张
#  myplot.ShowPictures(train_x)

  #参数
  params = {}
  params['layers'] = 3
  params['layers_info'] = {1:400,2:300,3:10}
  params['iters'] = 500
  params['lambda'] = 3
  params['learning_rate'] = 6
  params['class_num'] = 10
  #模型
  model = mymodel.MyNNModel(params)
  #训练
  model.Train(train_x, train_y)

  #预测(训练数据)
  pre_y = model.Predict(train_x)
  logger.info('迭代次数:%d',params['iters'])
  logger.debug('pred_y:%s', pre_y)
  logger.debug('     y:%s', train_y)
  print '训练集准确度为:%f' %(float(np.sum(train_y == pre_y)) / len(train_y))
  pred_testy = model.Predict(test_x)
#  print test_y
#  print pred_testy
  print '测试集准确率为:%f' %(float(np.sum(test_y == pred_testy))/len(test_y))

  for i in range(0, len(train_x)):
    j = np.random.permutation(len(train_x))[0]
    pred = model.Predict(train_x[j,:].reshape(1,train_x.shape[1]))
    print '该图像算法预测的数字是:', pred
    myplot.ShowPicture(train_x, j)

