Page({
  data: {
    queryResult: null,
    showView: 'false',
  },


  bindblur: function(e) {
    let that = this;
    wx.request({
      url: 'http://127.0.0.1:5000/ecs/getServerInfo',
      method: 'GET',
      data: {
        instanceId: e.detail.value
      },
      success(res) {
        console.log(res)
        if(res.statusCode == 200){
          that.setData({
            queryResult: res.data,
            showView: !that.data.showView,
          });
        }else{
          that.setData({
            showView: 'false',
          });
          wx.showToast({
            title: '请输入正确的实例ID',
            duration: 1500,
            icon: 'none',
            mask: true
          })
          
   
        }
      }

    })
  }
})