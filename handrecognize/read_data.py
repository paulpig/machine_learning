#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by paulpig on 17-10-22

import struct
class readData:
    def __init__(self,train_pic_path="./data/train-images.idx3-ubyte",train_label_path="./data/train-labels.idx1-ubyte",test_pic_path="./data/t10k-images.idx3-ubyte",test_label_path="./data/t10k-labels.idx1-ubyte"):
        self.train_pic_path=train_pic_path
        self.train_label_path=train_label_path
        self.test_pic_path=test_pic_path
        self.test_label_path=test_label_path
        return
    def read_picture_data(self,train_or_test):
        if train_or_test==True:
            f_train_pic=open(self.train_pic_path,'rb')
            f_train_label = open(self.train_label_path, 'rb')
        else:
            f_train_pic = open(self.test_pic_path, 'rb')
            f_train_label = open(self.test_label_path, 'rb')
        # print "hello world"
        #按照大端方式读取
        magic_number=struct.unpack('>i',f_train_pic.read(4))
        image_number = struct.unpack('>i', f_train_pic.read(4))
        rows_pic=struct.unpack('>i',f_train_pic.read(4))
        column_pic = struct.unpack('>i', f_train_pic.read(4))
        # print magic_number,image_number,rows_pic,column_pic
        pic_data=[]
        for j in range(image_number[0]):
            single_data=[]
            for i in range(rows_pic[0]*column_pic[0]):
                single_data.append(struct.unpack('>B',f_train_pic.read(1)))
            pic_data.append(single_data)
        # print pic_data[0][159]


        pic_label=[]
        magic_number=struct.unpack('>i',f_train_label.read(4))
        item_number = struct.unpack('>i', f_train_label.read(4))
        for i in range(item_number[0]):
            pic_label.append(struct.unpack('>B',f_train_label.read(1)))
        # print pic_label[7]
        f_train_pic.close()
        f_train_label.close()
        return
    def read_label_data(self):
        return


if __name__=="__main__":
    read_data=readData()
    read_data.read_picture_data(True)