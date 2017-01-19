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

#adaboost使用的callback方法
def train_and_return_model_callback(x, y, params):
  model = mymodel.MyNNModel(params)
  model.Train(x, y)
  return model

if __name__ == '__main__':
  logger.debug('start...')

  #
  trainfile = sys.argv[1]
  logger.debug('file:%s', trainfile)
  train_data = np.loadtxt(trainfile, delimiter = ',', dtype = np.float)
#  label_data = np.loadtxt(labelfile, delimiter = ',', dtype = np.float)
#  train_data = np.column_stack((train_data, label_data))
  np.random.shuffle(train_data)
  logger.debug('file:%s', trainfile)

  train_x = train_data[0:500,0:-10]
  train_data_y = train_data[0:500, -10:]
  train_y = myfunc.HArgmax(train_data_y, len(train_data_y))
  
  test_x = train_data[500:,0:-10]
  test_data_y = train_data[500:,-10:]
  test_y = myfunc.HArgmax(test_data_y, len(test_data_y))

  logger.debug('test_y:%s', test_y)
  logger.debug('train_y:%s', train_y)
  logger.debug('train_x:%s', train_x.shape)
  logger.debug('test_x :%s', test_x.shape)

  #显示图片随机抽取100张
#  myplot.ShowSomePictures(train_x, 100, 16, 16)

  #参数
  params = {}
  params['layers'] = 3
  params['layers_info'] = {1:256,2:55,3:10}
  params['iters'] = 300
  params['lambda'] = 2
  params['learning_rate'] = 5
  params['class_num'] = 10
  #模型
  model = mymodel.MyNNModel(params)
  #训练
#  model.Train(train_x, train_y)

  #预测(训练数据)Holdout method
  rate = [i for i in range(1500, 100, -200)]
  train_acc = []
  test_acc = []
  '''
   验证集合
  '''
  ada_acc = []
  k = 10
  for num in rate:
    '''
     数据
    '''
    train_x = train_data[0:num,0:-10]
    train_data_y = train_data[0:num, -10:]
    train_y = myfunc.HArgmax(train_data_y, len(train_data_y))
    test_x = train_data[num:,0:-10]
    test_data_y = train_data[num:,-10:]
    test_y = myfunc.HArgmax(test_data_y, len(test_data_y))
    '''
     循环训练
    '''
    avg_acc_train = 0.
    avg_acc_test = 0.
    for i in range(0, k):
      model.Train(train_x, train_y)
      pre_y = model.Predict(train_x)
  #  logger.info('迭代次数:%d',params['iters'])
  #  logger.debug('pred_y:%s', pre_y)
  #  logger.debug('     y:%s', train_y)
  #  print '训练集准确度为:%f' %(float(np.sum(train_y == pre_y)) / len(train_y))
      pred_testy = model.Predict(test_x)
      avg_acc_train += float(np.sum(train_y == pre_y)) / len(train_y)
      avg_acc_test  += float(np.sum(test_y == pred_testy))/len(test_y)
    train_acc.append(avg_acc_train / k)
    test_acc.append(avg_acc_test/ k)

    '''
     adaboost
    '''
    acc = myada.Adaboost(train_x, train_y, test_x, test_y, k, train_and_return_model_callback, params)
    ada_acc.append(acc)
  #logger.info('%d个分类器的adaboost验证集准确率:%f', num_classifier, acc)

  '''
    画图
  '''
  print '训练集样本个数:%s'%rate
#  logger.info('训练集准确率  :%s', train_acc)
#  logger.info('验证集准确率  :%s', test_acc)
  print 'Holdout验证集准确率  :%s'%test_acc
  print 'Adaboost验证集准确率 :%s'%ada_acc

#  myplot.Plot2DLine(rate, train_acc, 'train examples', 'accuracy')
#  myplot.Plot2DLine(rate, test_acc, 'train examples', 'accuracy')
#  myplot.Legend(['training set','validation set'])
#  myplot.Title('Holdout method accuracy')

  myplot.Plot2DLine(rate, test_acc, 'validation set', 'accuracy')
  myplot.Plot2DLine(rate, ada_acc, 'validation set', 'accuracy')
  myplot.Legend(['Holdout Method','Adaboost'])
  myplot.Title('Holdout & Adaboost')
  myplot.Show()
#  print test_y
#  print pred_testy
#  print '测试集准确率为:%f' %(float(np.sum(test_y == pred_testy))/len(test_y))
#  logger.info('%d次训练,训练集,平均准确率:%f', k, avg_acc_train / k)
#  logger.info('%d次训练,验证集,平均准确率:%f', k, avg_acc_test/ k)

  import time
#  time.sleep(3)

#  '''
#    adaboost
#  '''
#  
#  num_classifier = 10
#  acc = myada.Adaboost(train_x, train_y, test_x, test_y, num_classifier, train_and_return_model_callback, params)
#  logger.info('%d个分类器的adaboost验证集准确率:%f', num_classifier, acc)
  time.sleep(3)

#  '''
#    交叉验证
#  '''
#  train_x = train_data[:,0:-10]
#  train_data_y = train_data[:, -10:]
#  train_y = myfunc.HArgmax(train_data_y, len(train_data_y))
#  folds = [i for i in range(2, 11, 2)]
#  cv_acc = []
#  logger.info('folds:%s', folds)
#  for fold in folds:
#    cv_accuracy = mycv.CrossValidation(model.Train, model.Predict, train_x, train_y, fold)
##    logger.info('%d折交叉验证准确率:%f', fold, cv_accuracy)
#    cv_acc.append(cv_accuracy)
#  print '折数  :%s'%str(folds)
#  print '准确率:%s'%str(cv_acc)
#  myplot.Plot2DLine(folds, cv_acc, 'fold', 'accuracy')
#  myplot.Legend(['CV Accuracy'])
#  myplot.Title('Cross Validation Accuracy')
#  myplot.Show()

#  sys.exit(0)

  for i in range(0, len(train_x)):
    j = np.random.permutation(len(train_x))[0]
    pred = model.Predict(train_x[j,:].reshape(1,train_x.shape[1]))
    print '该图像算法预测的数字是:', pred
    myplot.ShowPicture(train_x, j, 16, 16)

