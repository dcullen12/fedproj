import ExifReader from 'exifreader';
import {PNG} from 'pngjs';
import jpeg from 'jpeg-js';
import imageType from 'image-type';
import * as tf from '@tensorflow/tfjs';

const photosURL = "https://photoslibrary.googleapis.com";

function getPhotos(token, width, height){
  let url = new URL("v1/mediaItems", photosURL);

  return fetch(url, {
    headers: {
      Authorization: `Bearer ${token}`
    }
  })
  .then(res => res.json())
  .then(res => {
    let rgb_images = [];
    for(let image of res['mediaItems']){
      let dataUrl = new URL(image['baseUrl'] + `=w${width}-h${height}`);
      console.log(dataUrl.href);
      
      rgb_images.push(new Promise((resolve, reject) => {
        fetch(dataUrl)
        .then(res => res.blob())
        .then(res => res.arrayBuffer())
        .then(res => {

          let itype = imageType(res);

          if(itype.ext === 'png'){
            console.log("PNG");
            return new PNG({filterType: 4}).parse(new Uint8Array(res), (err, data) => {
              if(!err){
                console.log(data);
                resolve(data);
              }else{
                reject(err);
              }
            })
          }else if(itype.ext === 'jpg' || itype.ext === 'jpeg'){
            console.log("JPG");
            resolve(jpeg.decode(res));
          }
        })
      }));
    }
    return Promise.all(rgb_images);
  });

}

export function trainOnPhotos(token, model, request){
  return new Promise((resolve, reject) => {
    getPhotos(token, request.dataRequest.width, request.dataRequest.height)
    .then(images => {
      return images.map(data => {
        let ret = []
        let i = 0;
        for(let y = 0; y < data.height; y++){
          ret.push([]);
          for(let x = 0; x < data.width; x++){
            ret[y].push([]);
            for(let c = 0; c < 3; c++){
              ret[y][x].push(data.data[i]);
              i++;
            }
            // Skip alpha
            i++;
          }
        }
        return ret;
      });
    })
    .then(async rgbPhotos => {
      console.log(rgbPhotos);
      let X = tf.tensor(rgbPhotos);
      let Y = tf.tensor(rgbPhotos);

      X.print();
      console.log(X.shape, Y.shape);

      await model.fit(X, Y);

      resolve(model);
    })
    .catch(reject);
  });
}