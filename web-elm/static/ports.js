// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

export function activatePorts(app, containerSize) {
  // Inform the Elm app when its container div gets resized.
  window.addEventListener("resize", () =>
    app.ports.resizes.send(containerSize())
  );

  let worker = new Worker("worker.js");

  // Global variable holding image ids
  let croppedImages = [];
  let registeredImages = [];
  let nb_images = 0;

  // Listen to worker messages.
  worker.onmessage = async function (event) {
    if (event.data.type == "log") {
      app.ports.log.send(event.data.data);
    } else if (event.data.type == "image-decoded") {
      const image = event.data.data;
      let img = await utils.decodeImage(image.url);
      app.ports.imageDecoded.send({ id: image.id, img });
    } else if (event.data.type == "cropped-image") {
      // Add the cropped image to the list of cropped images.
      const { id, arrayBuffer, imgCount } = event.data.data;
      console.log("Received cropped image in main:", id);
      const url = URL.createObjectURL(new Blob([arrayBuffer]));
      const decodedCropped = await utils.decodeImage(url);
      croppedImages.push({ id, img: decodedCropped });
      if (croppedImages.length == imgCount) {
        console.log(`Normal map computed, sending through port`);
        app.ports.receiveCroppedImages.send(croppedImages);
      }
    } else if (event.data.type == "registered-image") {
      // Add the registered image to the list of registered images.
      const { index, arrayBuffer, imgCount } = event.data.data;
      console.log("Received registered image in main thread:", index);
      registeredImages.push(arrayBuffer);
      if (registeredImages.length == imgCount) {
        console.log(`Warping and encoding of all ${imgCount} images done.`);
        for (let i = 0; i < imgCount; i++) {
          utils.download(
            registeredImages[i],
            `normal_map_for_${nb_images}.png`,
            "image/png"
          );
        }
        console.log("All images downloaded!");
      }
    } else if (event.data.type == "should-stop") {
      let { step, progress } = event.data.data;
      // Convert undefined to null to be a valid Elm "Maybe Int".
      progress = progress === undefined ? null : progress;
      app.ports.updateRunStep.send({ step, progress });
    } else {
      console.warn("Unknown message type:", event.data.type);
    }
  };

  // Listen for images to decode.
  app.ports.decodeImages.subscribe(async (imgs) => {
    console.log("Received images to decode");
    try {
      nb_images = 0;
      for (let img of imgs) {
        const url = URL.createObjectURL(img);
        worker.postMessage({
          type: "decode-image",
          data: { id: img.name, url },
        });
        nb_images += 1;
      }
    } catch (error) {
      console.error(error);
    }
  });

  // Listen for images to load and decode.
  app.ports.loadImagesFromUrls.subscribe(async (urls) => {
    console.log("Received images to load from urls");
    try {
      for (let url of urls) {
        const img = await utils.decodeImage(url);
        worker.postMessage({
          type: "decode-image",
          data: { id: url, url },
        });
      }
    } catch (error) {
      console.error(error);
    }
  });

  // Capture pointer events to detect a pointerup even outside the area.
  app.ports.capture.subscribe((event) => {
    event.target.setPointerCapture(event.pointerId);
  });

  function sendLog(lvl, content) {
    app.ports.log.send({ lvl, content });
  }

  // Run the registration algorithm with the provided parameters.
  app.ports.run.subscribe(async (params) => {
    croppedImages.length = 0; // reset associated cropped images
    worker.postMessage({ type: "run", data: params });
  });

  // Save registered images.
  app.ports.saveNMapPNG.subscribe(async (imgCount) => {
    registeredImages.length = 0; // reset associated registered images
    worker.postMessage({ type: "warp-encode", data: { imgCount } });
  });

  // Stop a running algorithm.
  app.ports.stop.subscribe(async () => {
    worker.postMessage({ type: "stop", data: null });
  });

  // Replace elm Browser.onAnimationFrameDelta that seems to have timing issues.
  // startAnimationFrameLoop(app.ports.animationFrame);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function startAnimationFrameLoop(port) {
  let timestamp = performance.now();
  let loop = (time) => {
    window.requestAnimationFrame(loop);
    let delta = time - timestamp;
    timestamp = time;
    port.send(delta);
  };
  loop();
}

// // Port to save dependencies to the store.
// app.ports.saveDependencies.subscribe((dep) => {
// 	db.put('dependencies', dep.value, dep.key);
// });

// // Use ports to communicate with local storage.
// app.ports.store.subscribe(function(value) {
// 	if (value === null) {
// 		localStorage.removeItem(storageKey);
// 	} else {
// 		localStorage.setItem(storageKey, JSON.stringify(value));
// 	}
// });
//
// // Whenever localStorage changes in another tab, report it if necessary.
// window.addEventListener("storage", function(event) {
// 	if (event.storageArea === localStorage && event.key === storageKey) {
// 		app.ports.onStoreChange.send(event.newValue);
// 	}
// }, false);
//
//

//
// // Read config file as text and send it back to the Elm app.
// app.ports.loadConfigFile.subscribe(file => {
//   utils
//     .readJsonFile(file)
//     .then(fileAsText => app.ports.configLoaded.send(fileAsText))
//     .catch(error => console.log(error));
// });
//
// // Export / save annotations
// app.ports.export.subscribe(value => {
//   utils.download(JSON.stringify(value), "annotations.json", "application/json");
// });
