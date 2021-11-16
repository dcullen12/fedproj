const fedExtensionID = 'fgkpihplpeabjpgphfjiemimmmngodee';
import * as tf from '@tensorflow/tfjs';
import {Buffer} from 'buffer';

async function serializeModel(model){
    let result = await model.save(tf.io.withSaveHandler(async modelArtifacts => modelArtifacts));
    result.weightData = Buffer.from(result.weightData).toString("base64");
    return JSON.stringify(result);
}

async function deserializeModel(modelString){
    let json = JSON.parse(modelString);
    let weightData = new Uint8Array(Buffer.from(json.weightData, "base64")).buffer;
    let model = await tf.loadLayersModel(tf.io.fromMemory(json.modelTopology, json.weightSpecs, weightData));
    return model
}

export function requestTraining(req){
    return new Promise(async (resolve, reject) => {
        chrome.runtime.sendMessage(fedExtensionID, req, (response) => {
            if(response.ok){
                resolve(deserializeModel(response.model));
            }else{
                reject(response.err);
            }
        });
    })
}

export async function trainOnDocs(model, timesteps, compile){
    return requestTraining({
        train: true,
        model: await serializeModel(model),
        compile: compile,
        dataRequest: {
            source: "docs",
            timesteps: timesteps
        }
    });
}

export async function trainOnPhotos(model, width, height, compile){
    return requestTraining({
        train: true,
        model: await serializeModel(model),
        compile: compile,
        dataRequest: {
            source: "photos",
            width: 512,
            height: 1024
        },
    })
}