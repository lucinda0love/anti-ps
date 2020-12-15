Page({
  data: {
    aimgurl: "", // //临时图片的路径
    countIndex: 1, // 可选图片剩余的数量
    imageData: {}, // 所选上传的图片数据 
    data1:null,
    data2:{}

   },
   
   onLoad: function (options) {
    var that = this
    that.data.data1=options
    },
  /*图片浏览及上传 */
    browse: function(e) {
    let that = this;
    wx.showActionSheet({
      itemList: ['从相册中选择', '拍照'],
      itemColor: "#CED63A",
      success: function(res) {
        if (!res.cancel) {
          if (res.tapIndex == 0) {
            that.chooseWxImage('album');
          } else if (res.tapIndex == 1) {
            that.chooseWxImage('camera');
          }
        }
      }
    })
  },

  /*打开相册、相机 */
  chooseWxImage: function(type) {
    let that = this;
    wx.chooseImage({
      count: that.data.countIndex,
      sizeType: ['original', 'compressed'],
      sourceType: [type],
      success: function(res) {
        const tempFilePaths = res.tempFilePaths
        that.setData({
          imagesList: tempFilePaths
        })
        wx.showToast({
          title:'正在上传',
          icon:'loading',
          mask:true,
          duration: 1000
        }),
        wx.uploadFile({
          url: 'http://127.0.0.1:5000/ecs/getServerInfo',
          filePath: tempFilePaths[0],
          name: "file",
          formData:{
            data1:that.data.data1['data1']
          },
          header: {"Content-Type": "multipart/form-data"},
          success:function(res){
            const data2 = res.data
            that.setData({
              img_stream:data2
            })
            var a = typeof that.data.img_stream
            if (a=="string") {
                wx.showModal({
                  title: res.data,
                  // content: '这是一个消息',
                  success: function (res) {
                    if (res.confirm) {//这里是点击了确定以后
                      console.log('用户点击确定')
                    } else {//这里是点击了取消以后
                      console.log('用户点击取消')
                    }

                  }
                })  
              
            } else {
              console.log(222)
            }
              
            }
           
        })
      }
    })
  },
 //点击保存图片
 save:function() {
  let that = this
  //若二维码未加载完毕，加个动画提高用户体验
  wx.showToast({
   icon: 'loading',
   title: '正在保存图片',
   duration: 1000
  })
  //判断用户是否授权"保存到相册"
  wx.getSetting({
   success (res) {
    //没有权限，发起授权
    if (!res.authSetting['scope.writePhotosAlbum']) {
     wx.authorize({
      scope: 'scope.writePhotosAlbum',
      success () {//用户允许授权，保存图片到相册
       that.savePhoto();
      },
      fail () {//用户点击拒绝授权，跳转到设置页，引导用户授权
       wx.openSetting({
        success () {
         wx.authorize({
          scope: 'scope.writePhotosAlbum',
          success() {
           that.savePhoto();
          }
         })
        }
       })
      }
     })
    } else {//用户已授权，保存到相册
     that.savePhoto()
    }
   }
  })
 },
//保存图片到相册，提示保存成功
 savePhoto() {
  let that = this
  wx.downloadFile({
   url: that.data.imagesList[0],
   success: function (res) {
    wx.saveImageToPhotosAlbum({
     filePath: res.tempFilePath,
     success(res) {
      wx.showToast({
       title: '保存成功',
       icon: "success",
       duration: 1000
      })
     }
    })
   }
  })
 },
  previewImage:function(e){
    var current = e.target.dataset.src;
    wx.previewImage({
      urls:this.data.imagesList,
      current: current,
      success:function(e){
        console.log("预览成功")
      }
   })
   }

})
