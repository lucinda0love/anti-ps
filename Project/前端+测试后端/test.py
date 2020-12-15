# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from aliyunsdkcore.client import AcsClient
import json
import numpy as np
from PIL import Image
import io
#from model import deal_Z
import matplotlib.pyplot as plt
import base64
import matplotlib.pyplot as plt
from aliyunsdkecs.request.v20140526 import DescribeInstancesRequest, DescribeInstanceStatusRequest

app = Flask(__name__)

accessKeyId = 'LTAI4G2XhMU4U5d9KYjNAu4r'
accessSecret = 'dcB9PgcVOYJp4cRqzSYmf0KU61YrcE'
region = 'cn-shenzhen'
client = AcsClient(accessKeyId, accessSecret, region)

# 在app.route装饰器中声明响应的URL和请求方法
@app.route('/ecs/getServerInfo', methods=['GET','POST'])


def get_image():
 
    f = request.files.get("file")
    a = request.values.get("data1")
    img = f.read()
    im1 = io.BytesIO(img) 
    im = Image.open(im1)
    im=im.convert("RGB")
    print(im)
    var_img=np.array(im)
    #deal_img = deal_Z(var_img)
    #plt.imsave("C:/AD/00.png",deal_img)
    #with open("C:/AD/00.png",'rb') as f:
     #   base64_data = base64.b64encode(f.read())
      #  s = base64_data.decode()
    #img_stream = base64.b64encode(im)
    a="Congratulations!"
    return a

if __name__ == "__main__":
    app.run()