# -*- coding: utf-8 -*-
from segnet_mobile import mobilenet_segnet
from PIL import Image
import copy
from keras.models import *
from keras.layers import *
import numpy as np
from skimage import exposure, img_as_float, io

#全局变量
IMAGE_ORDERING = 'channels_last'
NCLASSES = 2
HEIGHT = 1024
WIDTH = 1024

#-------测试集预测---------
def predict(path, model):

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
    
    modif_num = list(pr.reshape(1,-1)[0]).count(1)
    
    if modif_num > 100:
        for c in range(NCLASSES):
            seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

        seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))
        back = Image.blend(old_img,seg_img,0.3)  #原图和检测完的mask图叠加，0.3是mask图的透明度
    else:
        back = "恭喜你，该证书未被修改过！"
    
    return back

#导入模型
model = mobilenet_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/weights.h5")

image = predict('datasets/4.jpg', model)
print(image)
#image.save("datasets/"+'22.png')
#展示结果
#io.imshow(np.array(image))
#print(image.size,image)
