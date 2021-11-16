import * as tf from '@tensorflow/tfjs';

export async function serializeModel(model){
    let result = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
    result.weightData = Buffer.from(result.weightData).toString("base64");
    return JSON.stringify(result);
}
  
export async function deserializeModel(modelString){
    let json = JSON.parse(modelString);
    let weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
    let model = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));
    return model
  }