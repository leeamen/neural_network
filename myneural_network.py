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
    self.lamda = 1.0*params['lambda']
    self.learning_rate = params['learning_rate']
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

  def __gradient_descent(self, x, y):
    iters = self.iters
    learning_rate = self.learning_rate
    m = len(x)

    #获取误差和梯度
    last_errors, weights_grad = self.__error_function(x, y)
    for i in range(0, iters):
      errors, weights_grad = self.__error_function(x, y)
      for layer in layers:
        self.weights[layer] = self.weights[layer] - learning_rate * weights_grad[layer]
      logger.info('iteration %d | Errors:%f | Loss:%f', i+1, errors, last_errors - errors)
      #
      last_errors = errors
    logger.info('Iteration over! %f', last_errors)

  def __error_function(self, x, y):
    #层数
    layers = self.layers
    #样例的条数
    m = len(x)
    #每一层的输出output
    outputs = {}
    #每一层的net；wx,sigmoid之前的值
    nets = {}
    #所有权值的梯度
    gradients = {}
    #当前层的x
    this_layer_x = x
    #第一层输出为x,加上偏置x0 = 1
    outputs[1] = np.column_stack((np.ones(len(this_layer_x)), this_layer_x))
    #前向传播
    for layer in range(2, layers+1):
      net = self.__net(this_layer_x, layer)
      nets[layer] = net
      #__output已经添加偏置x0=1
      output = self.__output(net)
      outputs[layer] = output
      #下一层x为本层的output
      this_layer_x = output
    #误差
    errors = 1.0/(2*m) * np.sum((outputs[layers] - y) ** 2)
    for layer in range(1, layers):
      errors += self.lamda/(2*m)*np.sum(self.weights[layer][1:,:]**2)

    #反向传播
    deltas = {}
    #最后一层的delta
    deltas[layers] = (outputs[layers] - y) * self.__sigmoid_derivative(nets[layers])
    #反向传播计算delta
    for layer in range(layers - 1, 1):
      #self.weights[layer].T[:,1:],去除偏置的权值那一列
      deltas[layer] = self.__sigmoid_derivative(nets[layer]) * np.dot(deltas[layer+1], self.weights[layer].T[:,1:])
    ##求梯度
    weights_grad = {}
    for layer in range(1, layers):
      weight_grad = np.dot(outputs[layer].T, deltas[layer + 1])
      #1/m 并且加上正则项
      weight_grad = 1.0/m * weight_grad
      y_size = self.weights[layer].shape[2]
      weight_grad += self.lamda/m * np.vstack((np.zeros((1,y_size)), self.weights[layer][1:,:]))
      weights_grad[layer] = weight_grad
    return errors, weights_grad

  def __gradient_descent(self, x, y):
    iters = self.iters
    logger.debug('迭代次数:%d', iters)

    errors_last = 0
    for i in range(0, iters):
      logger.info('iteration %d | Errors:%f | Loss:%f', i, errors_this, errors_this - errors_last)
  #
  def __net(self, x, layer):
    #从第二层开始计算weight * x
    input_x = x
    net =  np.dot(input_x, self.all_weights[layer])
    return net

  def __output(self, x):
    output = self.__sigmoid(x)
    return np.column_stack((np.ones(len(output)), output))

  def Train(self, x, y):
    pass
  def Validate(self, x, y):
    pass
  def Predict(self, x):
    pass
  #对net求导:o(1-o)
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
  params['lambda'] = 1

