#!/usr/bin/python
#coding:utf8

#我的包
from mltoolkits import *
#非我的包
import os
import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class MyNNModel(object):
  def __init__(self, params = {}):
    self.layers = params['layers']
    self.layers_info = params['layers_info']
    self.iters = params['iters']
    self.all_weights = {}
    logger.debug('神经网络有%d层', self.layers)

    #初始化权值
    for layer in range(1, self.layers):
      #例如401*25
      weights = myequation.RandomSample((self.layers_info[layer] + 1, self.layers_info[layer+1]), 0, 1)
      self.all_weights[layer] = weights
      logger.debug('初始化第%d层权值:%s', layer, weights)

    self.train_finish = False

  #all_weights类型:{}
  def LoadWeights(self, all_weights):
    #第二层开始有权值
    for layer in range(2, self.layers+1):
      self.all_weights[layer] = all_weights[layer]
    return self.all_weights

  def __error_function(self, x, y):
    #层数
    layers = self.layers
    #样例的条数
    m = len(x)
    #每一层的输出output
    outputs = {}
    #所有权值的梯度
    gradients = {}
    #当前层的x
    this_layer_x = x
    #第一层输出为x
    outputs[1] = np.column_stack((np.ones(len(this_layer_x)), self.__net(this_layer_x, 1)))
    #前向传播
    for layer in range(2, layers+1):
      net = self.__net(this_layer_x, layer)
      output = self.__output(net)
      outputs[layer] = output
      #下一层x为本层的output
      this_layer_x = output

    #反向传播
    deltas = {}
    #最后一层delta
    deltas[layers] = 

  def __gradient_descent(self, x, y):
    iters = self.iters
    logger.debug('迭代次数:%d', iters)
    
    errors_last = 0
    for i in range(0, iters):
      
      logger.info('iteration %d | Errors:%f | Loss:%f', i, errors_this, errors_this - errors_last)
  #
  def __net(self, x, layer):
    #加上一列
    #第一层输出为x本身加上偏置
    if layer is 1:
      return np.column_stack((np.ones(len(x)), x))
    #从第二层开始计算weight * x
    input_x = np.column_stack((np.ones(len(x)), x))
    net =  np.dot(input_x, self.all_weights[layer])
    return net

  def __output(self, x):
    output = self.__sigmoid(x)
    return np.column_stack((np.ones(len(output)), output))

  def Train(self, x, y):

  def Validate(self, x, y):
    pass
  def Predict(self, x):

  def __sigmoid_derivative(self, x):
    return myequation.SigmoidGradient(x)
  def __sigmoid(self, x):
    return myequation.Sigmoid(x)

if __name__ == '__main__':
  logger.debug('start...')

  params = {}
  params['layers'] = 3
  #第2层的单元个数为25个,是隐藏层
  params['layers_info'] = {1:400,2:25,3:10}
  params['iters'] = 50

