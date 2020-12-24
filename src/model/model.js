import * as tfconv from "@tensorflow/tfjs-converter";
import * as face from "@tensorflow-models/blazeface/dist/face";

const BLAZEFACE_MODEL_URL = './model';
const BlazeFaceModel = face.BlazeFaceModel;
async function load({ maxFaces = 10, inputWidth = 128, inputHeight = 128, iouThreshold = 0.3, scoreThreshold = 0.75 } = {}) {
    const blazeface = await tfconv.loadGraphModel(BLAZEFACE_MODEL_URL, { fromTFHub: true });
    const model = new face.BlazeFaceModel(blazeface, inputWidth, inputHeight, maxFaces, iouThreshold, scoreThreshold);
    return model;
}

export{
    load,
    BlazeFaceModel
};