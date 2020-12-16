#前后端交互所需包
from flask import Flask,render_template,url_for,request,json,flash
from io import BytesIO
import matplotlib.pyplot as plt
import base64
#模型所需包
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
def predict(img, model):

    class_colors = [[0,0,0],[0,255,0]]
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


###############上边是模型###############

"""
每次发版都有一个版本号
"""
app = Flask(__name__)


@app.route('/login/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        img = f.read()
        bytes_img = BytesIO(img)
        var_img = Image.open(bytes_img)
        
        back = predict(var_img, model)
        if type(back) == str:
            return back
        else:
            back.save("datasets/"+'result.png')
            with open("datasets/result.png",'rb') as f:
                base64_data = base64.b64encode(f.read())
                s = base64_data.decode()
            return s


if __name__ == '__main__':
    # 调试模式：服务器会在代码修改后自动重新载入，并在发生错误时提供一个相当有用的调试器
    app.run()


"""
目的：实现一个简单的登录的逻辑处理
1. 路由需要有get和post两种请求方式---需要判断请求方式
2. 获取请求的参数
3. 判断参数是否填写，以及密码是否相同
4. 如果判断都没有问题，就返回一个success
"""
"""
给模板传递消息
flash--需要求内容加密，因此需要设置secret_key，做加密消息的混淆
模板中需要遍历消息
"""
"""
使用WTF实现一个表单
自定义表单类
"""







