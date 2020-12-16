# -*- coding: utf-8 -*-
from keras.models import *
from keras.layers import *
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from keras.applications.mobilenet import MobileNet

from get_train_datasets import generate_arrays_from_file

#全局变量
IMAGE_ORDERING = 'channels_last'
NCLASSES = 2
HEIGHT = 1024
WIDTH = 1024

#-------------解码层---------------
#参数含义：取编码后的第几层，是图像，分类数
def segnet_decoder(f, n_classes, n_up=4):

    #assert n_up >= 2  #断言
    o = f  #要解码的图像
    #先进行0填充，卷积，然后标准化；此时是为反卷积做准备
    o = ( ZeroPadding2D((1,1), data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = ( BatchNormalization())(o)
    
    #进入反卷积；此时图片已经进行了4次缩小，变为原来的1/16
    #进行一次UpSampling2D（上采样），此时hw变为原来的1/8，放大一倍
    #64,64,256
    o = ( UpSampling2D( (2,2), data_format=IMAGE_ORDERING))(o)
    o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
    o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)  #因为提前进行了填充，所以卷积后大小不变
    o = ( BatchNormalization())(o)
    # 进行一次UpSampling2D，此时hw变为原来的1/4
    # 128,128,128
    o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ) )(o)
    o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING ))(o)
    o = ( BatchNormalization())(o)
    # 进行一次UpSampling2D，此时hw变为原来的1/2
    # 256,256,64
    o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ))(o)
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
    o = ( BatchNormalization())(o)
    
    #512,512,32
    o = ( UpSampling2D((2,2)  , data_format=IMAGE_ORDERING ))(o)
    o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
    o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
    o = ( BatchNormalization())(o)
    # 此时输出为h_input/2,w_input/2,nclasses：
    #height/2,width/2,2
    o =  Conv2D( n_classes , (3, 3) , padding='same', data_format=IMAGE_ORDERING )( o ) #不填充，直接变为2维的
     
    return o 

#构建segnet网络
def _segnet(n_classes, input_height=416, input_width=608, encoder_level=4):
    #编码
    ## 将特征传入segnet网络，迁移学习
    model = MobileNet(include_top=False,weights='imagenet',input_shape=(1024,1024,3))
    feat = model.output
    
    #解码
    o = segnet_decoder(feat, n_classes, n_up=3)

    # 将结果进行reshape
    o = Reshape((int(input_height/2)*int(input_width/2), -1))(o)  #h/2 * w/2,2
    
    #经过预测函数，预测类别
    o = Softmax()(o)  #每一行代表一个像素点，两列分别为0的概率，1的概率
    
    img_input = model.input  #输入形式

    model = Model(img_input,o)  #keras里的函数

    return model

#构建网络
def mobilenet_segnet( n_classes, input_height=224, input_width=224 , encoder_level=3):

    model = _segnet(n_classes, input_height=input_height, input_width=input_width, encoder_level=encoder_level)
    model.model_name = "mobilenet_segnet"
    return model

#求损失
def loss(y_true, y_pred):
    loss = K.categorical_crossentropy(y_true,y_pred)
    return loss



#用来训练模型
if __name__ == "__main__":
    log_dir = "logs/"
    # 获取组好的model
    model = mobilenet_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)

    # 打开数据集
    lines = os.listdir('datasets/train')

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)  #估计
    num_train = len(lines) - num_val  #训练

    # 保存的方式，3世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=3
                                )
    # 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-4),
            metrics = ['accuracy'])
    batch_size = 4
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    #利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。即可在CPU，也可在GPU
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),  #生成器返回这么多次数据时，迭代结束，执行下一个epoch
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),  #生成验证集的生成器
            validation_steps=max(1, num_val//batch_size),  #指定验证集生成器执行的次数
            epochs=50,   #数据迭代的轮次
            initial_epoch=0,  #从该参数指定的epoch开始训练，在继续之前的训练时有用
            callbacks=[checkpoint_period, reduce_lr])  #在训练时调用的一系列回调函数

    model.save_weights(log_dir+'last1.h5')