var app = getApp();
Page({
  data: {
    keep: 1,
    /*写死部分*/
    date: "24",
    time: "2019/03",
    day: 0,
    motto: [
      "耐心和恒心总会得到报酬的。\n——爱因斯坦",
      "不积硅步无以至千里，\n不积小流无以成江海。",
      "\n宝剑锋从磨砺出，梅花香自苦寒来。",
      "千里始足下，高山起微尘，\n吾道亦如此，行之贵日新。\n——白居易",
      "锲而舍之，朽木不折;\n锲而不舍，金石可镂。\n——荀况",
      "明日复明日，明日何其多！\n我生待明日，万事成蹉跎。\n——《明日歌》",
      "劝君莫惜金缕衣，劝君须惜少年时。\n有花堪折直须折，莫待无花空折枝。\n——《金缕衣》"
    ]
  },
  to_set_logs: function(){
    wx.navigateTo({
      url: '/pages/set-logs/set-logs',
      success: function(res) {},
      fail: function(res) {},
      complete: function(res) {},
    })
  },
  to_test: function () {
    wx.navigateTo({
      url: '/pages/test/test',
      success: function (res) { },
      fail: function (res) { },
      complete: function (res) { },
    })
  },
  onLoad: function (options) {
    if (app.globalData.udata.countCard!=-1){
      this.setData({
        udata: app.globalData.udata
      })
      this.doMore()
    }else{
      app.udataCallback = res => {
        this.setData({
          udata: app.globalData.udata
        })
        this.doMore()
      }
    }
  },
  doMore: function(){
    var date = new Date();
    //计算持续天数
    var buildTime = new Date(this.data.udata.buildTime);
    var stringTime = '' + (buildTime.getMonth()+1) +'-'+ buildTime.getDate() +'-'+ buildTime.getFullYear();
    buildTime = new Date(stringTime);
    var keep = (date - buildTime) / 86400000 + 1;
    keep = parseInt(keep);
    this.setData({
      keep:keep
    })
    //计算当前时间，放入data
    var y = 2019;
    var m = 3;
    var d = 24;
    y = date.getFullYear();
    m = (date.getMonth() + 1 < 10 ? "0" + (date.getMonth() + 1) : date.getMonth() + 1);
    d = date.getDate();
    this.setData({ date: "" + d });
    this.setData({ time: "" + y + "/" + m });
    this.setData({ day: date.getDay() });
  },
  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
    
  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})
