import * as tf from '@tensorflow/tfjs';

const driveURL = "https://www.googleapis.com/drive/v3/";

import dicts from '../../dicts';
const wordsToNums = dicts.wordsToNums;
const numsToWords = dicts.numsToWords;

console.log('dicts ready');

function binToInt(bits){
  let ret = 0;
  for(let bit of bits){
    ret = ret << 1;
    ret = ret | bit;
  }
  return ret;
}

function getDocs(token) {
    const url = new URL("/drive/v3/files", driveURL);
  
    return fetch(url, {
      headers: {
        Authorization: `Bearer ${token}`
      }
    }).then(res => {
      return res.json();
    })
    .then(res => {
      let ids = []
      for(let file of res.files){
        if(file.mimeType === 'application/vnd.google-apps.document'){
          ids.push(file.id);
        }
      }
  
      let fileContents = [];
      for(let id of ids){
        let fileURL = new URL(`/drive/v3/files/${id}/export?mimeType=text/plain`, driveURL);
        fileContents.push(fetch(fileURL, {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }).then(fres => {
          return fres.text();
        }));
      }
      return Promise.all(fileContents);
    });
  }

  export function binaryEncode(el){
    // Convert to binary string of number mapping
    let hashBits = wordsToNums[el].toString(2);
    // Pad 0s for uniform input size (17 bits for all words in english language)
    hashBits = '0'.repeat(Math.max(0, 17 - hashBits.length)) + hashBits;
    // Convert string to bit array and return
    return hashBits.split('').map(v => parseInt(v));
  }

  export function decodeBinary(bits){
      return numsToWords[binToInt(bits)];
  }


// Map each file to bit arrays
/*
eg: "Hello world!" => ["hello", "world", "!"] => [[1,0,1,0,0,...], [0,1,1,0,1,...], [0,0,0,0,1,...]]
*/
function transformFileContents(files) {
    return files.map(file => {
      return file
      .split(/\s|([\.\?!])/) // separate words and simple punctuation
      .reduce((res, el) => { // Convert word lists to bit arrays of numeric representations
        if(!el) return res;
        el = el.toLowerCase();
        // Skip unrecognized words
        if(el in wordsToNums){
          res.push(binaryEncode(el));
        }
        return res
      }, []);
    });
  }

  export function trainOnDocs(token, model, request){
    return new Promise((resolve, reject) => {
      getDocs(token)
      .then(async files => {
  
        let exampleLength = request['dataRequest']['timesteps']
        let transformedFiles = transformFileContents(files);
        let X = [];
        let Y = [];
  
        transformedFiles.forEach(file => {
          if(file.length <= exampleLength) return;
          for(let i = 0; i < file.length - exampleLength; i += 5){
            X.push(file.slice(i, i + 5));
            Y.push(file[i + 5]);
          }
        });
  
  
        X = tf.tensor(X);
        Y = tf.tensor(Y);
  
        await model.fit(X, Y);
        
        resolve(model);
  
      })
      .catch(err => reject(err));
    });
  }