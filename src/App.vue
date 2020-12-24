<template>
  <div id="app">
    <router-view/>
  </div>
</template>

<style lang="less">
body {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  background: #F79F9F;
  color: #ffffff;
}

.banner{
  text-align: center;
  margin: 50rpx auto;
  font-size: 50rpx;
  font-weight: bolder;
}

.image{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.image img{
  width: 60%;
  box-shadow: 0rpx 10rpx 30rpx 5rpx rgba( 128, 128 , 128, 0.1);
}

.btn-group button{
  margin-top: 20rpx;
  min-width: 500rpx;
  font-weight: 400;
  box-shadow: 0rpx 10rpx 30rpx 5rpx rgba( 128, 128 , 128, 0.1);
}

.btn-group .select{
  background: #005476;
  color: #E1EBF6;
}

.btn-group .save{
  background: #EC3150;
  color: #FFFFFF;
}

.statement{
  width: 100%;
  font-size: 18rpx;
  line-height: 28rpx;
  text-align: center;
  margin-top: 30rpx;
  color: #ffffff;
}

</style>
