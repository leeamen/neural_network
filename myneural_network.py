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
logger.setLevel(logging.INFO)

class MyNNModel(object):
  def __init__(self, params = {}):
    self.class_num = params['class_num']
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
      logger.debug('权值%d,维度(%d,%d)',layer, self.layers_info[layer] + 1, self.layers_info[layer+1])
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
    layers = self.layers
    iters = self.iters
    learning_rate = self.learning_rate
    m = len(x)

    logger.debug('迭代次数:%d', iters)

    #获取误差和梯度
    last_errors, weights_grad = self.__error_function(x, y)
    for i in range(0, iters):
      errors, weights_grad = self.__error_function(x, y)
      for layer in range(1, layers):
        self.all_weights[layer] = self.all_weights[layer] - learning_rate * weights_grad[layer]
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
    #第一层输出为x,加上偏置x0 = 1
    outputs[1] = np.column_stack((np.ones(len(x)), x))
    #当前层的x
    this_layer_x = outputs[1]
    #前向传播
    for layer in range(2, layers):
      net = self.__net(this_layer_x, layer)
      nets[layer] = net
      #__output已经添加偏置x0=1
      output = self.__output(net)
      outputs[layer] = output
      #下一层x为本层的output
      this_layer_x = output
    #输出层不加偏置
    net = self.__net(this_layer_x, layers)
    nets[layers] = net
    output = self.__sigmoid(net)
    outputs[layers] = output

    #误差
    errors = 1.0/(2*m) * np.sum((outputs[layers] - y) ** 2)
    for layer in range(1, layers):
      errors += self.lamda/(2*m)*np.sum(self.all_weights[layer][1:,:]**2)

    #反向传播
    deltas = {}
    #最后一层的delta
#    logger.debug('nets[%d]:shape:%s,%s',layers, nets[layers].shape, nets[layers])
    logger.debug('nets[%d]:%s',layers, nets[layers])
    deltas[layers] = (outputs[layers] - y) * self.__sigmoid_derivative(nets[layers])
    #反向传播计算delta
    for layer in range(layers - 1, 1, -1):
      #self.all_weights[layer].T[:,1:],去除偏置的权值那一列
      deltas[layer] = self.__sigmoid_derivative(nets[layer]) * np.dot(deltas[layer+1], self.all_weights[layer].T[:,1:])
    ##求梯度
    weights_grad = {}
    for layer in range(1, layers):
      logger.debug('deltas[%d]:%s', layer + 1, deltas[layer + 1])
      weight_grad = np.dot(outputs[layer].T, deltas[layer + 1])
      #1/m 并且加上正则项
      weight_grad = 1.0/m * weight_grad
      y_size = self.all_weights[layer].shape[1]
      weight_grad += self.lamda/m * np.vstack((np.zeros((1,y_size)), self.all_weights[layer][1:,:]))
      weights_grad[layer] = weight_grad
    return errors, weights_grad

  def __net(self, x, layer):
    #从第二层开始计算weight * x
    input_x = x
    net =  np.dot(input_x, self.all_weights[layer-1])
    return net

  def __output(self, x):
    output = self.__sigmoid(x)
    return np.column_stack((np.ones(len(output)), output))

  def Train(self, x, y):
    train_x = x
    train_y = np.empty((len(y), 0))
    class_num = self.class_num
    for label in range(0, class_num):
      label_y = np.array(y == label, dtype = float)
      train_y = np.column_stack((train_y, label_y))
    self.__gradient_descent(train_x, train_y)
  def Predict(self, x):
    layers = self.layers
    output_last = np.column_stack((np.ones(len(x)), x))
    for layer in range(2, layers):
      net = self.__net(output_last, layer)
      output = self.__output(net)
      output_last = output
    #输出层
    net = self.__net(output_last, layers)
    output = self.__sigmoid(net)
    pred_y = myfunction.HArgmax(output, len(output))
    return pred_y

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

