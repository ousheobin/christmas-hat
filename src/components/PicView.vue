<template>
  <div class="pic-view">
    <div class="loading-mask" v-if="showLoading">
      <div class="sk-chase">
        <div class="sk-chase-dot"></div>
        <div class="sk-chase-dot"></div>
        <div class="sk-chase-dot"></div>
        <div class="sk-chase-dot"></div>
        <div class="sk-chase-dot"></div>
        <div class="sk-chase-dot"></div>
      </div>
      <p class="loading-hints">{{loadingHints}}</p>
    </div>

    <input type="file" @change="handleFileChange" ref="imgInput" style="display:none;" accept="image/*"/> 
    <img :src="dataURL" class="preview-img" v-if="!showCanvas"/>
    <canvas class="preview-img" ref="drawingCanvas" v-if="showCanvas"></canvas>
    <button class="btn btn-select-pic" @click="selectFile">选择图片</button>
    <button class="btn btn-save-pic"  @click="save">保存图片</button>
  </div>
</template>

<script>
import * as tf from '@tensorflow/tfjs-core';
import * as blazeface from '../model/model';

import '@tensorflow/tfjs-backend-webgl';

const windowURL = window.URL || window.webkitURL;
let model = null;

export default {
  name: 'PicView',
  mounted(){
    if(!this.$refs.drawingCanvas){
      alert('哎呀，你的浏览器不支持这个功能哦,要不换个浏览器？');
    }

    if(!this.$data.canvasCtx){
      this.$data.canvasCtx = this.$refs.drawingCanvas.getContext("2d");
    }
    this.handlePic();
  },
  methods:{
    handleFileChange(){
      let inputDOM = this.$refs.imgInput;
      if(inputDOM && inputDOM.files && inputDOM.files.length > 0){
        this.$data.filePath = windowURL.createObjectURL(inputDOM.files[0])
        this.handlePic();
        this.$data.showLoading = true;
        this.$data.showCanvas = true;
      }
    },
    selectFile(){
      let inputDOM = this.$refs.imgInput;
      inputDOM.click();
    },

    save(){
      alert('长按图片就可以保存啦~')
    },

    handlePic(){
      let ctx = this.$data.canvasCtx;
      let canvasElement = this.$refs.drawingCanvas;
      ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      this.drawImg(ctx,()=>{
        let imgData = ctx.getImageData(0,0,canvasElement.width, canvasElement.height);
        let rgbData = new Uint8Array(imgData.width * imgData.height * 3);
        let idx = 0;
        for (let i = 0; i < imgData.data.length; i +=4) {
          rgbData[idx] = imgData.data[i];
          rgbData[idx + 1] = imgData.data[i + 1];
          rgbData[idx + 2] = imgData.data[i + 2];
          idx += 3;
        }
        let imgTensor = tf.tensor3d(rgbData, [imgData.height, imgData.width , 3]);
        this.doPredict(imgTensor);
      })
    },

    drawImg(ctx,callback){
      let canvasElement = this.$refs.drawingCanvas;
      const image = new Image();
      image.src = this.$data.filePath;
      image.onload = ()=>{
        let widthOfImg = canvasElement.width;
        let scaledHeight = Math.round((canvasElement.width / image.width) * image.height);
        canvasElement.height = scaledHeight;
        ctx.drawImage(image, 0, 0, widthOfImg , scaledHeight );
        if(callback){
          callback();
        }
      }
    },

    async doPredict(imgTensor){
      if(!model){
        model = await blazeface.load();
        this.$data.loadingHints = '模型正在努力地研究，稍等...'
      }
      const returnTensors = false;
      const flipHorizontal = false;
      const annotateBoxes = true;
      const predictions = await model.estimateFaces(imgTensor, returnTensors, flipHorizontal, annotateBoxes);
      this.drawHat(predictions);
    },

    drawHat(predictions){
      const ctx = this.$data.canvasCtx;
      const element = this.$refs.drawingCanvas;
      ctx.clearRect(0,0,element.width,element.height);
      // console.log(predictions)
      this.drawImg(ctx, ()=>{
        let leftEye = [0,0];
        let rightEye = [0,0];
        let nose = [0,0];
        
        for (let i = 0; i < predictions.length; i++) {

          const landmarks = predictions[i].landmarks;
          leftEye = landmarks[0];
          rightEye = landmarks[1];
          nose = landmarks[2];

          let distanceBetweenEyes = this.distanceBtwPoints(leftEye,rightEye);
          let slopOfEyesLine = this.slope(leftEye,rightEye);
          let biasOfEyesLine = leftEye[1] - leftEye[0] * slopOfEyesLine;
          let slopOfNoseToHead = -1 / slopOfEyesLine;
          let biasOfNoseToHead = nose[1] - nose[0] * slopOfNoseToHead;
          let crossPointX = (biasOfEyesLine - biasOfNoseToHead) / (slopOfNoseToHead - slopOfEyesLine );
          let crossPointY = slopOfNoseToHead * crossPointX + biasOfNoseToHead;
          let hatCenterX = crossPointX - (nose[0] - crossPointX);
          let hatCenterY = crossPointY - (nose[1] - crossPointY);

          const start = predictions[i].topLeft;
          const end = predictions[i].bottomRight;
          const size = [end[0] - start[0], end[1] - start[1]];

          const image = new Image();
          image.src = 'img/common/hat.png';
          image.onload = ()=>{
            let width = Math.abs(end[0] - start[0]) * 1.2;
            let height = Math.round((width / image.width) * image.height);
            ctx.drawImage(image, hatCenterX - width * 0.75 , hatCenterY - height * 1.1, width , height );
            if( i == predictions.length - 1){
              this.$data.dataURL = element.toDataURL('img/png');
              this.$data.showLoading = false;
              this.$data.showCanvas = false;
            }
          }
        }
      });
    },
    slope(a,b){
      if(a[1] == b[1]){
        return 0;
      }
      return  (a[1] - b[1]) / (a[0] - b[0]); 
    },
    distanceBtwPoints(a,b){
      return Math.sqrt(Math.pow(a[0]-b[0], 2) + Math.pow(a[1]-b[1], 2))
    },
  },
  data(){
    return {
      showLoading:true,
      dataURL: '',
      showCanvas: true,
      loadingHints: 'AI 模型正在加载 ing ...',
      filePath: 'img/common/cat.jpg',
      imgElement: null,
      canvasCtx: null,
      showPreviewImg: true
    }
  }
}
</script>

<style lang="less">

.loading-mask {
  position: fixed;
  z-index: 2000;
  background-color: hsla(351, 71%, 51%, 0.897);
  margin: 0;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  transition: opacity .3s;

  .sk-chase {
    width: 40px;
    height: 40px;
    margin: 20% auto;
    position: relative;
    animation: sk-chase 2.5s infinite linear both;
  }

  .sk-chase-dot {
    width: 100%;
    height: 100%;
    position: absolute;
    left: 0;
    top: 0; 
    animation: sk-chase-dot 2.0s infinite ease-in-out both; 
  }

  .sk-chase-dot:before {
    content: '';
    display: block;
    width: 25%;
    height: 25%;
    background: #ffffff;
    border-radius: 100%;
    animation: sk-chase-dot-before 2.0s infinite ease-in-out both; 
  }

  .sk-chase-dot:nth-child(1) { animation-delay: -1.1s; }
  .sk-chase-dot:nth-child(2) { animation-delay: -1.0s; }
  .sk-chase-dot:nth-child(3) { animation-delay: -0.9s; }
  .sk-chase-dot:nth-child(4) { animation-delay: -0.8s; }
  .sk-chase-dot:nth-child(5) { animation-delay: -0.7s; }
  .sk-chase-dot:nth-child(6) { animation-delay: -0.6s; }
  .sk-chase-dot:nth-child(1):before { animation-delay: -1.1s; }
  .sk-chase-dot:nth-child(2):before { animation-delay: -1.0s; }
  .sk-chase-dot:nth-child(3):before { animation-delay: -0.9s; }
  .sk-chase-dot:nth-child(4):before { animation-delay: -0.8s; }
  .sk-chase-dot:nth-child(5):before { animation-delay: -0.7s; }
  .sk-chase-dot:nth-child(6):before { animation-delay: -0.6s; }

  @keyframes sk-chase {
    100% { transform: rotate(360deg); } 
  }

  @keyframes sk-chase-dot {
    80%, 100% { transform: rotate(360deg); } 
  }

  @keyframes sk-chase-dot-before {
    50% {
      transform: scale(0.4); 
    } 100%, 0% {
      transform: scale(1.0); 
    } 
  }
}


.pic-view{
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 20px;

  & .btn{
    width: 80%;
    height: 48px;
    margin: 10px 0px 0px 0px;
    font-size: 18px;
    color: #ffffff;
    border: none;
    font-weight: 400;
  }

  & .btn-select-pic{
    margin-top: 50px;
    background: #005476;
    color: #E1EBF6;
  }

  & .btn-save-pic{
    background: #EC3150;
  }

  & .preview-img{
    width: 80%;
    height: auto;
    border: 2px dotted #ffffff;
    padding: 10px;
    border-radius: 10px;
  }
}
</style>