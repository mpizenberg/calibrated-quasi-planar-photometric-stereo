# Calibrated quasi-planar photometric stereo

[![Watch the video][thumbnail]][video]

Slides of the video above: [pdf][slides]

This is a Rust implementation of a calibrated quasi-planar photometric stereo algorithm.
This repository is organized in four main directories:

- `cal-qp-ps-lib/`: the core parts of the algorithm presented as a Rust library.
- `cal-qp-ps-bin/`: an example CLI executable program.
- `cal-qp-ps-wasm/`: the WebAssembly modules exposing the algorithm in wasm.
- `web-elm/`: the frontend application, made in Elm.

[video]: https://youtu.be/oGjEF13Qmvs
[thumbnail]: https://img.youtube.com/vi/oGjEF13Qmvs/0.jpg
[slides]: https://mpizenberg.github.io/resources/calibrated-quasi-planar-photometric-stereo/photometric-stereo-web-rust-cv-sept-2021.pdf

## Example CLI application

To run the example CLI application, simply move into the `cal-...-bin/` directory
and run a command similar to this one:

```sh
cargo run --release -- --lights data-bayeux/lights.csv data-bayeux/*.jpg
```

Where `lights.csv` contains the directions of the lights for each image.
It is obtained from the light calibration step and looks like as follows:

```csv
-5572.41288647651,10223.2280248599,15623.3881625612
876.927634237602,11292.6574672864,17249.6879948994
12776.0813287204,8668.26040675236,13911.4380411348
...
```

The images passed as arguments with `data-bayeux/*jpg` should not contain
the first, almost black, reference image.
We do as if it was already deducted from all images.

## Web application

To build and run the Web application for this photometric stereo algorithm,
perform the following steps:

```sh
# move into the wasm directory
cd cal-qp-ps-wasm/

# build the rust code into a wasm module
wasm-pack build --target web -- --features console_error_panic_hook

# move into the web app directory
cd ../web-elm/

# build the frontend app
elm make src/Main.elm --optimize --output=static/Elm.js

# move into the statically served directory
cd static/

# turn the JS modules into simple JS files
# ps: --watch is rather convenient here
esbuild worker.mjs --bundle --preserve-symlinks --outfile=worker.js

# start a static server able to serve wasm files
# with the correct mime types such as http-server
http-server

# open http://localhost:8080 in your Web browser of choice
```
