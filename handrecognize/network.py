#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by paulpig on 17-10-23

import numpy as np
import read_data
#lossfunction为最大似然估计，输出层为softmax，三层神经网络
class nerualNetwork:
    #输入为list类型，内部转换为array类型
    def __init__(self,input_data,input_label,layer_numbers=3,hidden_layer_numbers=100,output_numbers=10):
        self.train_number=input_data.shape[0]
        self.feature_number=input_data.shape[1]
        self.hidden_layer_numbers=hidden_layer_numbers
        self.input_data=np.array(input_data)
        self.layer_numbers=layer_numbers
        self.input_label=np.array(input_label)
        self.hidden_layer=[]
        # self.weights1=np.array([[np.random.uniform(0.0,1.0) for i in range(hidden_layer_numbers)] for j in range(self.feature_number)])
        for i in range(layer_numbers-2):
            self.weights_hidden.append(np.array([[np.random.uniform(0.0,1.0) for i in range(hidden_layer_numbers)] for j in range(hidden_layer_numbers)]))
        self.weights_hidden=np.array(self.weights_hidden)
        self.output_weight=np.array([[np.random.uniform(0.0,1.0) for i in range(hidden_layer_numbers)] for j in range(output_numbers)])
        return

    #x的row为train_number,column为feature+1,1代表的是b
    #w的row为feature+1,column为下一层的number+1
    def sigmoid(self,x,w):
        return 1.0/(1.0+np.exp(-np.dot(x,w)))
    #sigmoid的导数
    def sigmoidDaoShu(self,x):
        return x*(1-x)
    #输出节点的函数,output为行向量
    def softmax(self,output):
        return np.exp(output*1.0)/(np.sum(np.exp(output),axis=1))
    #前向传播
    def forward(self,input_data):
        #第一层传播
        hidden_layer_1=self.sigmoid(self.input_data,self.weights_hidden[0])
        self.hidden_layer.append(hidden_layer_1)
        #隐藏层的传播
        hidden_layer_temp=hidden_layer_1
        for i in range(1,self.layer_numbers-2):
            hidden_layer_temp=self.sigmoid(hidden_layer_temp,self.weights_hidden[i])
            self.hidden_layer.append(hidden_layer_temp)
        self.hidden_layer=np.array(self.hidden_layer)
        #输出层传播
        output_layer=np.dot(hidden_layer_temp,self.output_weight)
        #防止数据过大，导致溢出
        output_layer=output_layer-np.max(output_layer)
        output_layer=self.softmax(output_layer)
        result_label=np.array(np.where(output_layer==np.max(output_layer,axis=1)))[:,1]
        #构造掩码keepProb,L3范式暂时不加
        keepProb = np.zeros_like(output_layer)
        keepProb[np.arange(self.train_number), self.input_label] = 1.0
        loss=-np.sum(keepProb*output_layer)*1.0/self.train_number
        return output_layer,result_label,loss
    #反向传播
    #
    def bp(self,fb_numbers):
        out_layer=self.forward()[0]
        keepProb=keepProb = np.zeros_like(out_layer)
        keepProb[np.arange(self.train_number),self.input_label]=1.0
        for i in range(fb_numbers):
            self.output_weight += -np.dot(self.hidden_layer[-1].T, out_layer - keepProb) / self.train_number

            # pre_tidu=1
            # if self.layer_numbers>3:
            #     pre_tidu*=np.dot(keepProb - out_layer,self.output_weight.T)*self.sigmoidDaoShu(self.hidden_layer[-1])  #表达式中的wji和之前的一大堆相乘
            #     self.weights_hidden[-1] += -np.dot(self.hidden_layer[-2].T, pre_tidu)
            # # pre_sigema= #
            pre_tidu=keepProb - out_layer;
            pre_weight=self.output_weight
            for i in range(0,self.layer_numbers-3):
                #隐藏层中的梯度表达式,以及隐藏层到输出层的表达式
                pre_tidu=np.dot(pre_tidu,pre_weight.T)*self.sigmoidDaoShu(self.hidden_layer[-i-1]),
                self.weights_hidden[-1-i]+=-np.dot(self.hidden_layer[-2-i].T,pre_tidu)
                pre_weight=self.weights_hidden[-1-i]
                # pre_weight=self.weights_hidden[-1-i]
            #隐藏层到输出层的表达式
            # pre_tidu = np.dot(pre_tidu, self.weights1.T) * self.sigmoidDaoShu(self.hidden_layer[1]),
            # self.weights1 += -np.dot(self.hidden_layer[0].T, pre_tidu)
        return

if __name__=="__main__":
    train_obj=read_data.readData()
    train_data,train_label=read_data.read_picture_data(True)
