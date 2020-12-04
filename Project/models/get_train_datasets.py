# -*- coding: utf-8 -*-
#用来获取每一个batch的训练集
from PIL import Image
import numpy as np

#全局变量
IMAGE_ORDERING = 'channels_last'
NCLASSES = 2
HEIGHT = 1024
WIDTH = 1024

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i]
            # 从文件中读取图像
            
            num = int(name.split('.')[0])
            ind = (((num-1)//100)+1)*100
            path = '../input/certidatasets/train'+str(ind)+'/train'+str(ind) 
            img = Image.open(path + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            img = np.array(img)
            img = img/255
            X_train.append(img)

            name = lines[i].split('.')[0] + '.png'
            # 从文件中读取图像
            img = Image.open(r"../input/certidatasets/train_mask/train_mask" + '/' + name)
            img = img.resize((int(WIDTH/2),int(HEIGHT/2)))
            img = np.array(img)
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            for c in range(NCLASSES):
                if c == 0:
                    seg_labels[: , : , c ] = (img == c ).astype(int)
                else:
                    seg_labels[: , : , c ] = (img == 255 ).astype(int) * 255
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))
