# -*- coding: utf-8 -*-
import numpy
import random
import math

class FNN:
    #初始化
    def __init__(self,train_data,train_label,hidden_num,output_num):
        self.train_data=train_data
        # print 'train_data',self.train_data,train_label
        self.train_label=train_label
        self.hidden_num=hidden_num
        self.outpu_num=output_num
        #隐藏层的数据
        self.hidden_data=[0 for i in range(hidden_num)]
        #输出层的数据
        self.output_data=[0 for i in range(output_num)]
        #输入层到隐藏层的weight
        self.i_h_weight=[[ random.uniform(-1.0, 1.0) for j in range(hidden_num)] for i in range(len(train_data[0]))]
        #隐藏层到输出层的weight
        self.h_o_weight=[[ random.uniform(-1.0, 1.0) for i in range(output_num)] for j in range(hidden_num)]
        #隐藏层的b
        self.hidde_b=[0 for i in range(hidden_num)]
        #输出层的b
        self.output_b=[0 for j in range(output_num)]
        self.error=[0 for i in range(output_num)]

    #sigmod激活函数
    def sigmod(self,x):
        return 1.0/(1+math.exp(-x))

    def sigmoid_derivative(self,x):
        return x * (1 - x)


    #前馈网络
    def feedforward(self,data_index):
        # 隐藏层的数据
        self.hidden_data = [0 for i in range(self.hidden_num)]
        # 输出层的数据
        self.output_data = [0 for i in range(self.outpu_num)]
        # print self.hidden_data
        # 只对一条数据进行训练的
        # 得到隐藏层的数据
        for i in range(self.hidden_num):
            total=0.0
            for j in range(len(self.train_data[0])):
                total+=self.train_data[data_index][j]*self.i_h_weight[j][i]
            total+=self.hidde_b[i]
            self.hidden_data[i]=self.sigmod(total)
        #得到输出层的数据
        for i in range(self.outpu_num):
            total=0.0
            for j in range(self.hidden_num):
                total+=self.hidden_data[j]*self.h_o_weight[j][i]
            total+=self.output_b[i]
            # self.output_data[i]=self.output_data[i]
            self.output_data[i]=self.sigmod(total)
        print 'predict',self.output_data[0],self.train_data[data_index][0]
        return self.output_data[0]

    #BP反馈网络
    def feedback(self,MM,data_index):
        # 前馈网络
        self.feedforward(data_index)

        # #更新隐藏层到输出层的weight和b
        for i in range(len(self.output_data)):
            # print data_index,i
            self.error[i]=self.train_label[data_index][i]-self.output_data[i]
        # print 'error',self.error
        # print 'hidden_data',self.error

        for i in range(self.outpu_num):
            for j in range(self.hidden_num):
                self.h_o_weight[j][i]+=MM*self.hidden_data[j]*self.error[i]*self.output_data[i]*(1-self.output_data[i])
            self.output_b[i]+=MM*self.output_data[i]*(1-self.output_data[i])*self.error[i]
        #更行输入层到输出层的weight和b
        for i in range(self.hidden_num):
            sum_ek=0.0
            for k in range(self.outpu_num):
                sum_ek+=self.h_o_weight[i][k]*self.error[k]*self.output_data[k]*(1-self.output_data[k])
            for j in range(len(self.train_data[0])):
                self.i_h_weight[j][i]+=MM*self.hidden_data[i]*(1-self.hidden_data[i])*self.train_data[data_index][j]*sum_ek
            self.hidde_b[i]+=MM*self.hidden_data[i]*(1-self.hidden_data[i])*sum_ek


    #训练
    def train(self,train_num,MM,):
        for i in range(train_num):
            rand_int= i%4;
            # rand_int=random.randint(0,len(self.train_data)-1)
            # print rand_int
            # self.feedforward(rand_int)
            self.feedback(MM,rand_int)
            # if i%33==0:
            # print 'error',self.error[0]*self.error[0]


if __name__ == '__main__':
    train_data=[ [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],]
    train_label=[[0], [1], [1], [0]]
    object=FNN(train_data,train_label,5,1)
    object.train(100000,0.05)
    print 'predect',object.feedforward(3)
