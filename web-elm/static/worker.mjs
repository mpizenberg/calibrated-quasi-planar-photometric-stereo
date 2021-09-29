// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

// Import and initialize the WebAssembly module.
// Remark: ES modules are not supported in Web Workers,
// so you have to process this file with esbuild:
// esbuild worker.mjs --bundle --preserve-symlinks --outfile=worker.js
import { Stenm as StenmWasm, default as init } from "./pkg/cal_qp_ps_wasm.js";

// Initialize the wasm module.
// Let's hope this finishes before someone needs to call a Stenm method.
let Stenm;
(async function () {
  await init("./pkg/cal_qp_ps_wasm_bg.wasm");
  Stenm = StenmWasm.init();
})();

// Global module variable recording if the algorithm was asked to stop.
let stopOrder = false;

console.log("Hello from worker");

// Listener for messages containing data of the shape: { type, data }
// where type can be one of:
//   - "decode-image": decode an image provided with its url
//   - "run": run the algorithm on all images
//   - "stop": stop the alogorithm
onmessage = async function (event) {
  console.log(`worker message: ${event.data.type}`);
  if (event.data.type == "decode-image") {
    await decode(event.data.data);
    postMessage({ type: "image-decoded", data: event.data.data });
  } else if (event.data.type == "run") {
    await run(event.data.data);
  } else if (event.data.type == "warp-encode") {
    await warpEncode(event.data.data);
  } else if (event.data.type == "stop") {
    console.log("Received STOP in worker");
    stopOrder = true;
  }
};

// Load image into wasm memory and decode it.
async function decode({ id, url }) {
  console.log("Loading into wasm: " + id);
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  Stenm.load(id, new Uint8Array(arrayBuffer));
}

// Main algorithm with the parameters passed as arguments.
async function run(params) {
  console.log("worker running with parameters:", params);
  // Convert params to what is expected by the Rust code.
  const lights = params.lights;

  for (const light of lights) {
    Stenm.push_light(light.x, light.y, light.z);
  }

  const args = {
    config: {
      max_iterations: params.maxIterations,
      threshold: params.convergenceThreshold,
      verbosity: params.maxVerbosity,
      z_mean: params.z_mean,
    },
    crop: params.crop,
  };

  // Run stenm main registration algorithm.
  stopOrder = false;
  let motion = await Stenm.run(args);

  // Send back to main thread all cropped images.
  const image_ids = Stenm.image_ids();
  const imgCount = image_ids.length;
  console.log(`Encoding normal map:`);
  let NMu8 = Stenm.normal_map();
  postMessage(
    {
      type: "cropped-image",
      data: { id: "n_map", arrayBuffer: NMu8.buffer, imgCount: 1 },
    },
    [NMu8.buffer]
  );
  // for (let i = 0; i < imgCount; i++) {
  //   await shouldStop("encoding", i);
  //   const id = image_ids[i];
  //   console.log("   Encoding ", id, " ...");
  //   let croppedImgArrayU8 = Stenm.normal_map(i);
  //   // Transfer the array buffer back to main thread.
  //   postMessage(
  //     {
  //       type: "cropped-image",
  //       data: { id, arrayBuffer: croppedImgArrayU8.buffer, imgCount },
  //     },
  //     [croppedImgArrayU8.buffer]
  //   );
  // }
  await shouldStop("done", null);
}

// Warp and encode images that have just been registered.
async function warpEncode({ imgCount }) {
  stopOrder = false;
  console.log("Warping and encoding registered images");
  // Warp and encode image in wasm.
  let imgArrayU8 = Stenm.register_and_save();
  // Transfer the array buffer back to main thread.
  postMessage(
    {
      type: "registered-image",
      data: {
        index: "normal-map",
        arrayBuffer: imgArrayU8.buffer,
        imgCount: 1,
      },
    },
    [imgArrayU8.buffer]
  );
}

// Log something in the interface with the provided verbosity level.
export function appLog(lvl, content) {
  postMessage({ type: "log", data: { lvl, content } });
}

// Function regularly called in the algorithm to check if it should stop.
export async function shouldStop(step, progress) {
  postMessage({ type: "should-stop", data: { step, progress } });
  await sleep(0); // Force to give control back.
  return stopOrder;
}

// Small utility function.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
