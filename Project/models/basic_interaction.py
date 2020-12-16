
from flask import Flask
from flask import request
from io import BytesIO
import matplotlib.pyplot as plt
import base64
from PIL import Image
import numpy as np


with open("datasets/1.jpg",'rb') as f:
    #base64_data = base64.b64encode(f.read())
    #s = base64_data.decode()
    s = f.read()
#print(s)
#@app.route('/')
#def hello_world():
#    return 'hello world'

app = Flask(__name__)  #作用是为了确定资源所在的路径


    
#装饰器
#定义路由，通过装饰器实现
@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        img = f.read()
        bytes_img = BytesIO(img)
        var_img = Image.open(bytes_img)
        
        var_img = predict(var_img, model)

        
        print(var_img)

#        plt.imsave("00.png",deal_img)
#        with open("deal_img",'rb') as f:
#            base64_data = base64.b64encode(f.read())
#            s = base64_data.decode()
    return var_img


#启动程序。运行起一个小型服务器
if __name__ == '__main__':
    #执行这就话，就会将程序运行在一个简易的服务器，是flask提供的
    app.run()
    