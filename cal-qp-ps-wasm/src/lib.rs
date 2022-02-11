// SPDX-License-Identifier: MPL-2.0

use anyhow::anyhow;
use image::{DynamicImage, ImageError};
use nalgebra::{DMatrix, Scalar, Vector3, Vector4};
use serde::Deserialize;
use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use cal_qp_ps_lib as stenm;
use stenm::crop::{crop, Crop};
use stenm::interop::{IntoDMatrix, ToImage};
use stenm::pps;

#[macro_use]
mod utils; // define console_log! macro

#[wasm_bindgen(raw_module = "../worker.mjs")]
extern "C" {
    #[wasm_bindgen(js_name = "shouldStop")]
    async fn should_stop(step: &str, progress: Option<u32>) -> JsValue; // bool
}

/// Configuration (parameters) of the registration algorithm.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Config {
    pub max_iterations: usize,
    pub threshold: f32,
    pub verbosity: u32,
    pub z_mean: f32,
}

// This wrapper trick is because we cannot have async functions referencing &self.
// https://github.com/rustwasm/wasm-bindgen/issues/1858
#[wasm_bindgen]
pub struct Stenm(Rc<RefCell<StenmInner>>);

#[wasm_bindgen]
impl Stenm {
    pub fn init() -> Self {
        Stenm(Rc::new(RefCell::new(StenmInner::init())))
    }
    pub fn load(&mut self, id: String, img_file: &[u8]) -> Result<(), JsValue> {
        let inner = Rc::clone(&self.0);
        let result = (*inner).borrow_mut().load(id, img_file);
        result
    }
    pub fn push_light(&mut self, x: f32, y: f32, z: f32) {
        self.0.borrow_mut().push_light(x, y, z)
    }
    pub fn run(&mut self, params: JsValue) -> js_sys::Promise {
        let inner = Rc::clone(&self.0);
        wasm_bindgen_futures::future_to_promise(async_run_rc(inner, params))
    }
    pub fn image_ids(&self) -> Result<JsValue, JsValue> {
        self.0.borrow().image_ids()
    }
    pub fn normal_map(&self) -> Result<Box<[u8]>, JsValue> {
        self.0.borrow().normal_map()
    }
    pub fn register_and_save(&self) -> Result<Box<[u8]>, JsValue> {
        self.0.borrow().register_and_save()
    }
}

async fn async_run_rc(
    mutself: Rc<RefCell<StenmInner>>,
    params: JsValue,
) -> Result<JsValue, JsValue> {
    let mut inner = (*mutself).borrow_mut();
    let result = inner.run(params);
    result.await
}

struct StenmInner {
    image_ids: Vec<String>,
    dataset: Dataset,
    lights: Vec<(f32, f32, f32)>,
    normal_map: Vec<u8>,
}

enum Dataset {
    Empty,
    GrayImages(Vec<DMatrix<u8>>),
    GrayImagesU16(Vec<DMatrix<u16>>),
    RgbImages(Vec<DMatrix<(u8, u8, u8)>>),
    RgbImagesU16(Vec<DMatrix<(u16, u16, u16)>>),
}

#[wasm_bindgen]
#[derive(Deserialize)]
/// Type holding the algorithm parameters
pub struct Args {
    pub config: Config,
    pub crop: Option<Crop>,
}

/// 3D point, representing a 3D light direction
#[cfg_attr(feature = "wasm_bindgen", wasm_bindgen)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct Point3d {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl StenmInner {
    pub fn init() -> Self {
        utils::set_panic_hook();
        utils::WasmLogger::init().unwrap();
        utils::WasmLogger::setup(log::LevelFilter::Trace);
        Self {
            image_ids: Vec::new(),
            dataset: Dataset::Empty,
            // motion_vec: None,
            lights: Vec::new(),
            normal_map: Vec::<u8>::new(),
        }
    }

    pub fn push_light(&mut self, x: f32, y: f32, z: f32) {
        self.lights.push((x, y, z));
    }

    // Load and decode the images to be registered.
    pub fn load(&mut self, id: String, img_file: &[u8]) -> Result<(), JsValue> {
        console_log!("Loading an image");
        let reader = image::io::Reader::new(Cursor::new(img_file))
            .with_guessed_format()
            .expect("Cursor io never fails");
        // let image = reader.decode().expect("Error decoding the image");
        let dyn_img = reader.decode().map_err(utils::report_error)?;
        let reported_err = |str_msg: &str| Err(utils::report_error(anyhow!(str_msg.to_string())));

        match (&dyn_img, &mut self.dataset) {
            // Loading the first image (empty dataset)
            (DynamicImage::ImageLuma8(_), Dataset::Empty) => {
                log::info!("Images are of type Gray u8");
                self.dataset = Dataset::GrayImages(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageLuma8(_), Dataset::GrayImages(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageLuma16(_), Dataset::Empty) => {
                log::info!("Images are of type Gray u16");
                self.dataset = Dataset::GrayImagesU16(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageLuma16(_), Dataset::GrayImagesU16(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageRgb8(_), Dataset::Empty) => {
                log::info!("Images are of type RGB (u8, u8, u8)");
                self.dataset = Dataset::RgbImages(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageRgb8(_), Dataset::RgbImages(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            // Loading the first image (empty dataset)
            (DynamicImage::ImageRgb16(_), Dataset::Empty) => {
                log::info!("Images are of type RGB (u16, u16, u16)");
                self.dataset = Dataset::RgbImagesU16(vec![dyn_img.into_dmatrix()]);
                self.image_ids = vec![id];
            }
            // Loading of subsequent images
            (DynamicImage::ImageRgb16(_), Dataset::RgbImagesU16(imgs)) => {
                imgs.push(dyn_img.into_dmatrix());
                self.image_ids.push(id);
            }
            (DynamicImage::ImageBgr8(_), _) => return reported_err("BGR order not supported"),
            (DynamicImage::ImageBgra8(_), _) => return reported_err("Alpha channel not supported"),
            (DynamicImage::ImageLumaA8(_), _) => {
                return reported_err("Alpha channel not supported")
            }
            (DynamicImage::ImageLumaA16(_), _) => {
                return reported_err("Alpha channel not supported")
            }
            (DynamicImage::ImageRgba8(_), _) => return reported_err("Alpha channel not supported"),
            (DynamicImage::ImageRgba16(_), _) => {
                return reported_err("Alpha channel not supported")
            }
            _ => return reported_err("Images are not all of the same type"),
        }

        Ok(())
    }

    /// Run the main algorithm.
    ///                                                Vec<f32>
    async fn run(&mut self, params: JsValue) -> Result<JsValue, JsValue> {
        // self.motion_vec = None;
        self.normal_map.clear();
        let args: Args = params.into_serde().unwrap();
        utils::WasmLogger::setup(utils::verbosity_filter(args.config.verbosity));

        log::info!("No conf yet");

        let conf = pps::Config {
            max_iterations: args.config.max_iterations,
            threshold: args.config.threshold,
            z_mean: args.config.z_mean,
            lights: self.lights.clone(),
        };

        log::info!("Conf : {}", conf.z_mean);

        let (depths, normals, _) = match &self.dataset {
            Dataset::Empty => todo!(),
            Dataset::GrayImages(_) => {
                todo!()
            }
            Dataset::GrayImagesU16(_) => {
                todo!()
            }
            Dataset::RgbImages(imgs) => {
                // let cropped_imgs: &Vec<DMatrix<(u8, u8, u8)>> = match args.crop {
                //     None => imgs,
                //     Some(frame) => imgs
                //         .iter()
                //         .map(|im| {
                //             im.slice(
                //                 (frame.top, frame.left),
                //                 (frame.bottom - frame.top, frame.right - frame.left),
                //             )
                //             .into_owned()
                //         })
                //         .collect(),
                // };
                let raw_images: Vec<DMatrix<f32>> = match args.crop {
                    None => imgs.iter().map(|im| f32_image_matrix(im)).collect(),
                    Some(frame) => {
                        let cropped: Result<Vec<DMatrix<f32>>, _> = imgs
                            .iter()
                            .map(|im| {
                                log::info!("1st px : {:?}", im[1]);
                                im
                            })
                            .map(|im| f32_image_matrix(im))
                            .map(|im| {
                                log::info!("1st float : {:?}", im[1]);
                                let mean = im.mean();
                                log::info!("Moy :{}", mean);
                                im
                            })
                            .map(|im| crop(frame, &im))
                            .collect();
                        match cropped {
                            Err(e) => {
                                log::info!("Problem while cropping : {}", e);
                                imgs.iter().map(|im| f32_image_matrix(im)).collect()
                            }
                            Ok(cr) => {
                                log::info!("No problem while cropping");
                                cr
                            }
                        }
                    }
                };
                log::info!("RGB");
                let results = pps::photometric_stereo(conf, &raw_images);
                log::info!("Planar ok");
                results
            }
            Dataset::RgbImagesU16(_) => {
                todo!()
            }
        };

        log::info!("Computations ok");
        let (z_scale, rgb_normals_alpha_depth) = pps::join_normals_and_depths(&normals, &depths);

        // self.normal_map = save_normals(&normals).map_err(utils::report_error)?;
        self.normal_map = save_rgba(&rgb_normals_alpha_depth).map_err(utils::report_error)?;
        // self.normal_map = save_depths(&depths).map_err(utils::report_error)?;

        log::info!("Encode PNG OK");

        // TODO: what is this used for?
        let false_vec: Vec<f32> = Vec::new();
        JsValue::from_serde(&false_vec).map_err(utils::report_error)
    }

    // Return the ids of loaded images: [string]
    pub fn image_ids(&self) -> Result<JsValue, JsValue> {
        JsValue::from_serde(&self.image_ids).map_err(utils::report_error)
    }

    pub fn normal_map(&self) -> Result<Box<[u8]>, JsValue> {
        Ok(self.normal_map.clone().into_boxed_slice())
    }

    pub fn register_and_save(&self) -> Result<Box<[u8]>, JsValue> {
        log::info!("Downloading normal map as PNG");
        Ok(self.normal_map.clone().into_boxed_slice())
    }
}

async fn should_stop_bool(step: &str, progress: Option<u32>) -> bool {
    let js_bool = should_stop(step, progress).await;
    js_bool.as_bool().unwrap()
}

trait IntoF32Gray {
    fn into_gray_f32(self) -> f32;
}

impl IntoF32Gray for (u8, u8, u8) {
    fn into_gray_f32(self) -> f32 {
        (0.2989 * self.0 as f32 + 0.5870 * self.1 as f32 + 0.1140 * self.2 as f32).min(255.0)
            / 255.0
    }
}
impl IntoF32Gray for (u16, u16, u16) {
    fn into_gray_f32(self) -> f32 {
        (0.2989 * self.0 as f32 + 0.5870 * self.1 as f32 + 0.1140 * self.2 as f32)
            .min(256.0 * 256.0 - 1.0)
            / (256.0 * 256.0 - 1.0)
    }
}

fn f32_image_matrix<T: IntoF32Gray + Clone + Scalar>(mat: &DMatrix<T>) -> DMatrix<f32> {
    let (d1, d2) = mat.shape();
    DMatrix::from_iterator(d1, d2, mat.iter().map(|pix| pix.clone().into_gray_f32()))
}

type Coords = Vector3<f32>;

fn save_normals(img: &DMatrix<Coords>) -> Result<Vec<u8>, ImageError> {
    let img_u8 = img.map(|n| {
        (
            ((n.x + 1.0) / 2.0 * 255.0) as u8,
            ((n.y + 1.0) / 2.0 * 255.0) as u8,
            ((n.z + 1.0) / 2.0 * 255.0) as u8,
        )
    });
    encode(&img_u8)
}

fn save_depths(depths: &DMatrix<f32>) -> Result<Vec<u8>, ImageError> {
    // Prepare for the visualization of depths as u8.
    let depth_min = depths.min();
    let depth_max = depths.max();
    log::warn!("depths within [ {},  {} ]", depth_min, depth_max);
    log::warn!("mean depth: {}", depths.mean());
    let scale: f32 = depth_max - depth_min;
    let depth_to_gray = |z| ((z - depth_min) / scale * 255.0) as u8;
    let img_u8 = depths.map(depth_to_gray);
    encode(&img_u8)
}

fn save_rgba(img: &DMatrix<Vector4<u8>>) -> Result<Vec<u8>, ImageError> {
    log::warn!("Mean of rgba normal&depth: {}", img.map(|v| v.max()).max());
    let img_u8 = img.map(|n| (n.x, n.y, n.z, n.w));
    encode(&img_u8)
}

fn encode<Im: ToImage>(mat: &Im) -> Result<Vec<u8>, ImageError> {
    let img = mat.to_image();
    let mut buffer: Vec<u8> = Vec::new();
    img.write_to(&mut buffer, image::ImageOutputFormat::Png)?;
    Ok(buffer)
}
