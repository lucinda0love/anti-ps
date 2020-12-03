# -*- coding: utf-8 -*-
from segnet_mobile import mobilenet_segnet
from PIL import Image
import copy
from keras.models import *
from keras.layers import *
import numpy as np

#全局变量
IMAGE_ORDERING = 'channels_last'
NCLASSES = 2
HEIGHT = 1024
WIDTH = 1024

#-------测试集预测---------
def predict(path):
    #导入模型
    model = mobilenet_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    model.load_weights("logs/last1.h5")

    class_colors = [[0,0,0],[0,255,0]]
    img = Image.open(path)
    old_img = copy.deepcopy(img)
    #保存原始图片信息
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))  #转换成模型需要的
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]  #取出每个像素点对应的概率

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)  #重塑形状，并找出每个像素点概率最大的下标

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img,seg_img,0.3)
    image.save("datasets/predict/"+path)
    
    return 'ok'

predict('xinda2.jpg')