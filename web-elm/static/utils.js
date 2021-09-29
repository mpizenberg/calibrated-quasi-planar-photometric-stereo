// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

const utils = (function () {
  // Download a file through an href temporary DOM element.
  // example use: download( str, "selection.json", "text/plain" )
  function download(data, name, mimeType) {
    const a = document.createElement("a");
    const blob = new Blob([data], { type: mimeType });
    const objectUrl = window.URL.createObjectURL(blob);
    a.href = objectUrl;
    a.download = name;
    click(a);
    window.URL.revokeObjectURL(objectUrl);
  }

  // Simple click emulation function.
  function click(node) {
    const event = new MouseEvent("click");
    node.dispatchEvent(event);
  }

  // Read JSON file as text
  function readJsonFile(file) {
    const promise = new Promise((resolve, reject) => {
      // if ( file.type === "application/json" ) { // ubuntu chrome returns "" ...
      if (file.name.match(/.*json/)) {
        const fileReader = new FileReader();
        fileReader.onload = () => resolve(fileReader.result);
        fileReader.readAsText(file);
      } else {
        reject("Incorrect file type, please load JSON file.");
      }
    });
    return promise;
  }

  // Provided an id and an image, returns an object
  // { id, url, width, height }
  // with the url corresponding to the image loaded
  function createImageObject(id, imageFile) {
    const promise = new Promise((resolve, reject) => {
      if (imageFile.type.startsWith("image")) {
        const img = document.createElement("img");
        img.onload = () => resolve({ id, img });
        img.src = window.URL.createObjectURL(imageFile);
      } else {
        reject("Not an image file: " + imageFile.type);
      }
    });
    return promise;
  }

  // Create an HTMLImageElement and return it when it is loaded.
  async function decodeImage(src) {
    let img = new Image();
    img.src = src;
    await img.decode();
    return img;
  }

  return {
    download: download,
    readJsonFile: readJsonFile,
    createImageObject: createImageObject,
    decodeImage: decodeImage,
  };
})();
