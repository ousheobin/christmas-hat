<template>
  <div class="home">
    <h1>Merry Christmas</h1>
    <p>试下看 AI 能不能给你的图片戴上圣诞帽</p>
    <picView/>
    <div class="statement">
      <p>本功能采用开源技术 Tensorflow.js 实现</p>
      <p>我们不会存储任何图片数据</p>
      <p>&copy; Steve's 2020</p>
    </div>
  </div>
</template>

<script>
import picView from '@/components/PicView.vue'

export default {
  name: 'Home',
  components: {
    picView
  }
}
</script>

<style lang="less">
.home{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

  & .statement{
    font-size: 10px;
    margin-top: 30px;
    & p{
      margin: 0px;
      line-height: 20px;
    }
  }
}
</style>