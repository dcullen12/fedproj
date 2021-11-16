import { trainOnDocs, trainOnPhotos } from "./client";
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

function testDocs(){
    let timesteps = 5;

    let model = tf.sequential();
    // model.add(tf.layers.lstm({units: 17, inputShape: [exampleLength, 17]}));
    // model.add(tf.layers.rnn({cell: cells, returnSequences: false, inputShape: [null, 17]}));
    model.add(tf.layers.lstm({units: 17, batchInputShape: [null, null, 17], returnSequences: false, activation: "sigmoid"}));
    model.summary();

    trainOnDocs(model, timesteps, {
        loss: 'meanSquaredError', 
        optimizer: 'sgd'
    })
    .then(updatedModel => {

        let r = "the university of michigan";
        let words = r.split(' ');
        chrome.runtime.sendMessage(fedExtensionID, {
            translate: true,
            words: words
        }, response => {
            let t = tf.tensor([response.bits]);
            t.print();
            updatedModel.predict(t).print();
        })
    })
}

async function testPhotos(){
    let width = 128;
    let height = 128;
    let model = tf.sequential();
    model.add(tf.layers.conv2d({filters: 32, kernelSize: 3, padding: 'same', activation: 'relu', inputShape: [null, null, 3]}));
    model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, padding: 'same', activation: 'relu',}));
    model.add(tf.layers.conv2d({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
    model.add(tf.layers.conv2dTranspose({filters: 8, kernelSize: 3, padding: 'same', activation: 'relu'}));
    model.add(tf.layers.conv2dTranspose({filters: 16, kernelSize: 3, padding: 'same', activation: 'relu'}));
    model.add(tf.layers.conv2dTranspose({filters: 32, kernelSize: 3, padding: 'same', activation: 'relu'}));
    model.add(tf.layers.conv2dTranspose({filters: 3, kernelSize: 3, padding: 'same', activation: 'relu'}));
    model.summary();

    let modelstr = await serializeModel(model);
    let cm = await deserializeModel(modelstr);
    cm.summary();

    trainOnPhotos(model, width, height, {
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    })
    .then(updatedModel => {
        console.log("Done");
    })
}

chrome.runtime.onInstalled.addListener(async () => {
    console.log("start");
    testPhotos();
    
})