import {serializeModel, deserializeModel} from './util';
import { trainOnDocs, binaryEncode, decodeBinary } from './docs';
import { trainOnPhotos } from './photos';

const sourcesToCallbacks = {
  "docs": trainOnDocs,
  "photos": trainOnPhotos
}

function handleTranslateRequest(request, sendResponse){
  if(!!request.words){
    let bits = []
    for(let word of request.words){
      bits.push(binaryEncode(word));
    }
    sendResponse({
      ok: true,
      bits: bits
    });
  }else if(!!request.bits){
    let word = decodeBinary(request.bits);
    sendResponse({
      ok: !!word,
      word: word
    });
  }
}

function handleTrainRequest(request, sendResponse){
  chrome.identity.getAuthToken({ interactive: true }, async function (token) {
    let model = await deserializeModel(request.model);
    model.compile(request.compile);

    let trainingCb = sourcesToCallbacks[request.dataRequest.source];
    if(!trainingCb){

      sendResponse({
        ok: false,
        err: "Invalid dataRequest source"
      });

    }else{

      console.log("Starting training");
      
      trainingCb(token, model, request)
      .then(async newModel => {
        console.log("Training done, serializing and sending model");
        let modelStr = await serializeModel(newModel);
        sendResponse({
          "ok": true,
          "model": modelStr
        })
      })
      .catch(err => {
        console.log(err);
        sendResponse({
          "ok": false,
          "err": JSON.stringify(err)
        });
      });
    }
  })
}


chrome.runtime.onMessageExternal.addListener(
  function(request, _, sendResponse){
    console.log(request);
    if(!!request.translate){
      console.log("Translating")
      handleTranslateRequest(request, sendResponse);
    }else if(!!request.train){
      handleTrainRequest(request, sendResponse);
    }
  }
);


// chrome.history.search({text: request.search, maxResults: request.numItems}, function(data) {
//   data.forEach(function(page) {
//       console.log(page.url);
//   });
// });