// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Planar photometric stereo

use crate::interop::ToImage;
use crate::normal_integration;
use nalgebra::base::allocator::Allocator;
use nalgebra::base::default_allocator::DefaultAllocator;
use nalgebra::base::dimension::{Dim, Dynamic, U3};
use nalgebra::{DMatrix, DVector, MatrixMN, RealField, Vector3, Vector4};
use std::path::Path;

type Mat3D = DMatrix<(f32, f32, f32)>;
// type RgbMat = DMatrix<(u8, u8, u8)>;

/// Configuration (parameters) of the photometric stereo algorithm.
#[derive(Debug, Clone)]
pub struct Config {
    pub max_iterations: usize,
    pub threshold: f32,
    pub z_mean: f32,
    pub lights: Vec<(f32, f32, f32)>,
    // pub intrinsics: camera::Intrinsics,
}

/// Compute the depth, normals and albedo from a sequence of images
/// with different lighting conditions.
/// Returns (depths, normals, albedo)
pub fn photometric_stereo(
    config: Config,
    raw_images: &[DMatrix<f32>], // f32 in [0,1]
) -> (DMatrix<f32>, DMatrix<Vector3<f32>>, DMatrix<f32>) {
    // eprintln!("lights: {:#?}", config.lights);
    // eprintln!("intrinsics: {:#?}", config.intrinsics);

    // Substract ambient image and convert to f32.
    // let imgs: Vec<DMatrix<f32>> = substract_ambiant(&raw_images[1..], &raw_images[0]);
    let imgs = raw_images.clone();

    // % Light fall-off compensation (by 1/(source-scene distance)^2)
    // TODO: actually remove light_attenuation and point_to_directions steps for now.
    //       We read lights directions directly from the --lights CLI argument.
    // let imgs = light_attenuation(&config, imgs);
    //
    // // % Equivalent directional lighting
    // let lights_directions = point_to_directions(&config, imgs[0].shape());
    let lights_directions: Vec<Vector3<f32>> = config
        .lights
        .iter()
        .map(|(lx, ly, lz)| nalgebra::Vector3::new(*lx, *ly, *lz).normalize())
        .collect();

    // % Iterative refinement of intensities, normals and albedo
    // [N,rho,phi] = semicalibrated_PS(I,S,nrows,ncols,nimgs,maxit,tol);
    let imgs_matrix = DMatrix::from_iterator(
        imgs[0].len(),
        imgs.len(),
        imgs.iter().map(|im| im.iter().cloned()).flatten(),
    )
    .transpose();
    let lights_directions_matrix: MatrixMN<f32, U3, Dynamic> =
        MatrixMN::from_columns(lights_directions.as_slice());
    let lights_directions_matrix = lights_directions_matrix.transpose();
    let obs = Obs {
        image_shape: imgs[0].shape(),
        images: &imgs_matrix, // nb_imgs x nb_pixels matrix
        lights_directions: &lights_directions_matrix, // S : nb_imgs x 3
    };
    let (normals, albedo, lights_intensities) = semicalibrated_ps(&config, &obs);

    // Reshape normals.
    // TODO: figure out the problem with normal orientations.
    let normals: DMatrix<Vector3<f32>> = DMatrix::from_iterator(
        obs.image_shape.0,
        obs.image_shape.1,
        normals.column_iter().map(|n| Vector3::new(-n.y, -n.x, n.z)),
    );
    eprintln!("min n.z: {}", normals.map(|n| n.z).min());
    eprintln!("max n.z: {}", normals.map(|n| n.z).max());
    eprintln!("mean n.z: {}", normals.map(|n| n.z).mean());

    let depths = normal_integration::normal_integration(&normals.map(|n| -n));
    return (depths, normals, albedo);

    // % Perspective normal integration into a depth map
    // z = perspective_integration(N,K,z_mean,nrows,ncols);

    // % Postprocessing to "flatten" the depth
    // [XYZ,N] = postprocess(z,K,z_mean,nrows,ncols);

    todo!()
}

/// Returns (scale, rgb_normals_alpha_depth).
pub fn join_normals_and_depths(
    normals: &DMatrix<Vector3<f32>>,
    depths: &DMatrix<f32>,
) -> (f32, DMatrix<Vector4<u8>>) {
    // Prepare for the visualization of depths as u8.
    let depth_min = depths.min();
    let depth_max = depths.max();
    // log::warn!("depths within [ {},  {} ]", depth_min, depth_max);
    let scale: f32 = depth_max - depth_min;
    let depth_to_gray = |z| ((z - depth_min) / scale * 255.0) as u8;

    // Prepare visualization of normals as RGB u8.
    let normals_to_rgb = |n: &Vector3<f32>| {
        if n.x.abs() + n.y.abs() + n.z.abs() == 0.0 {
            Vector3::new(0, 0, 0)
        } else {
            n.map(|x| (0.5 * (x + 1.0) * 255.0).min(255.0) as u8)
        }
    };

    // Concatenate normals and depths into a single RGBA buffer.
    let rgba_matrix = DMatrix::from_iterator(
        depths.nrows(),
        depths.ncols(),
        normals.iter().zip(depths.into_iter()).map(|(n, z)| {
            let rgb = normals_to_rgb(n);
            Vector4::new(rgb.x, rgb.y, rgb.z, depth_to_gray(z))
        }),
    );

    (scale, rgba_matrix)
}

fn substract_ambiant(raw_imgs: &[DMatrix<f32>], ambiant: &DMatrix<f32>) -> Vec<DMatrix<f32>> {
    raw_imgs
        .iter()
        .map(|im| im.zip_map(ambiant, |x, xa| (x - xa).max(0.0)))
        .collect()
}

// // I = I./attenuation_map_fcn(S,K,z_mean,nrows,ncols);
// // I = I./max(I(:));
// fn light_attenuation(config: &Config, imgs: Vec<DMatrix<f32>>) -> Vec<DMatrix<f32>> {
//     let mut max_intensity = 0.0;
//
//     // Attenuate with the distance for every pixel in the image
//     let mut attenuated_imgs: Vec<DMatrix<f32>> = imgs
//         .iter()
//         .zip(&config.lights)
//         .map(|(im, (lx, ly, lz))| {
//             let light = nalgebra::Point3::new(*lx, *ly, *lz);
//             im.map_with_location(|y, x, v| {
//                 let point2d = nalgebra::Point2::new(x as f32, y as f32);
//                 let depth = config.z_mean;
//                 let p = config.intrinsics.back_project(point2d, depth);
//                 let d2 = (p - light).norm_squared();
//                 let new_v = v / d2;
//                 max_intensity = max_intensity.max(new_v);
//                 new_v
//             })
//         })
//         .collect();
//
//     // // TEMP: write attenuated images to disk.
//     // for (i, im) in attenuated_imgs.iter().enumerate() {
//     //     save_matrix(im, format!("temp/{:2.0}.png", i));
//     // }
//
//     // Re-normalize all images with color values in [0..1]
//     let coef = 1.0 / max_intensity;
//     attenuated_imgs.iter_mut().for_each(|im| *im *= coef);
//     attenuated_imgs
// }

fn save_matrix<P: AsRef<Path>>(img: &DMatrix<f32>, path: P) {
    let im_max = img.max();
    img.map(|x| (x / im_max * 255.0) as u8)
        .to_image()
        .save(path)
        .unwrap();
}

// /// Retrieve the principal direction for each light source.
// /// This is the direction from the light source
// /// to the point at the center of the captured subject (center of image).
// fn point_to_directions(config: &Config, img_shape: (usize, usize)) -> Vec<Vector3<f32>> {
//     let (nrows, ncols) = img_shape;
//     let img_center = nalgebra::Point2::new(0.5 * ncols as f32, 0.5 * nrows as f32);
//     let subject_center = config.intrinsics.back_project(img_center, config.z_mean);
//     config
//         .lights
//         .iter()
//         .map(|(lx, ly, lz)| {
//             let light = nalgebra::Point3::new(*lx, *ly, *lz);
//             (light - subject_center).normalize()
//         })
//         .collect()
// }

// let (normals, albedo, lights_intensities) = semicalibrated_ps(&config, &obs);
// Careful, albedo is not scaled to be in [0,1].
fn semicalibrated_ps(
    config: &Config,
    obs: &Obs,
) -> (MatrixMN<f32, U3, Dynamic>, DMatrix<f32>, DVector<f32>) {
    // Initialize the normals.
    let (nrows, ncols) = obs.image_shape;
    let (nimgs, npixels) = obs.images.shape();
    let normals = vec![Vector3::new(0.0, 0.0, -1.0); npixels];
    let mut state = State {
        nb_iter: 1,
        residual: std::f32::MAX,
        normals: MatrixMN::from_columns(&normals),
        albedo: DMatrix::repeat(nrows, ncols, 0.0), // is re-initialized at each iteration
        lights_intensities: DVector::repeat(nimgs, 1.0),
    };
    let step_config = StepConfig {
        max_iterations: config.max_iterations,
        threshold: config.threshold,
    };

    while state.step(&step_config, obs) == Continue::Forward {
        state.nb_iter += 1;
    }

    // Careful, albedo is not scaled to be in [0,1]
    (state.normals, state.albedo, state.lights_intensities)
}

/// Configuration parameters for the core loop of the algorithm.
struct StepConfig {
    max_iterations: usize,
    threshold: f32,
}

/// "Observations" contains the data provided outside the core of the algorithm.
/// These are immutable references since we are not supposed to mutate them.
struct Obs<'a> {
    image_shape: (usize, usize),
    images: &'a DMatrix<f32>, // nb_imgs x nb_pixels matrix
    lights_directions: &'a MatrixMN<f32, Dynamic, U3>, // S : nb_imgs x 3
}

/// Simple enum type to indicate if we should continue to loop.
/// This is to avoid the ambiguity of booleans.
#[derive(PartialEq)]
enum Continue {
    Forward,
    Stop,
}

/// State variables of the loop.
struct State {
    nb_iter: usize,
    normals: MatrixMN<f32, U3, Dynamic>, // N : 3 x n_pixels
    albedo: DMatrix<f32>,                // rho : nrows x ncols
    lights_intensities: DVector<f32>,    // phi : n_img x 1
    residual: f32,
}

impl State {
    /// Core iteration step of the algorithm.
    fn step(&mut self, config: &StepConfig, obs: &Obs) -> Continue {
        // scaled_lights are the lights directions vectors
        // scaled by their respective intensities.
        let mut scaled_lights = obs.lights_directions.clone();
        for mut col in scaled_lights.column_iter_mut() {
            col.component_mul_assign(&self.lights_intensities);
        }

        // scaled_normals M in matlab code:
        // the product of the albedo by the surface normals.
        // scaled_normals: MatrixMN<f32, U3, Dynamic>,        // M : 3 x n_pixels
        let scaled_normals = scaled_lights.pseudo_inverse(1e-9).unwrap() * obs.images;

        // Update normals and albedo from scaled normals.
        for ((scaled_col, mut normal_col), albedo) in scaled_normals
            .column_iter()
            .zip(self.normals.column_iter_mut())
            .zip(self.albedo.iter_mut())
        {
            *albedo = scaled_col.norm();
            if *albedo > 1e-9 {
                normal_col.x = scaled_col.x / *albedo;
                normal_col.y = scaled_col.y / *albedo;
                normal_col.z = scaled_col.z / *albedo;
            } else {
                normal_col.x = 0.0;
                normal_col.y = 0.0;
                normal_col.z = -1.0;
            }
        }

        // Intensities estimation.
        let shading = obs.lights_directions * &self.normals;
        let mut rho_times_shading = shading;
        for (mut col, albedo) in rho_times_shading.column_iter_mut().zip(self.albedo.iter()) {
            col *= *albedo;
        }

        let old_lights_intensities = self.lights_intensities.clone();
        self.lights_intensities = obs.images.component_mul(&rho_times_shading).column_sum();
        rho_times_shading.iter_mut().for_each(|x| *x *= *x);
        let rho_times_shading_sqr_norms = rho_times_shading.column_sum(); // compute a column vector with the sum of each row
        self.lights_intensities
            .component_div_assign(&rho_times_shading_sqr_norms);

        // Set the first light intensity (first image) to 1 as a reference constraint.
        // The albedo will compensate.
        eprintln!("{}", self.lights_intensities.transpose());
        self.lights_intensities /= self.lights_intensities[0];

        // Current energy does not change.
        let mut scaled_lights = obs.lights_directions.clone();
        for mut col in scaled_lights.column_iter_mut() {
            col.component_mul_assign(&self.lights_intensities);
        }
        self.residual = mse(&(scaled_lights * &scaled_normals - obs.images)).sqrt();

        // Energy variation on Phi.
        let relative_difference = norm(&(old_lights_intensities - &self.lights_intensities))
            / norm(&self.lights_intensities);
        eprintln!(
            "RMSE at iteration {}: {:.6}  -  Phi relative variation: {:.6}",
            self.nb_iter, self.residual, relative_difference
        );

        // Check convergence
        let mut continuation = Continue::Forward;
        if self.nb_iter >= config.max_iterations || self.residual < config.threshold {
            continuation = Continue::Stop;
        }
        continuation
    }
}

fn coordinates_column_major(shape: (usize, usize)) -> impl Iterator<Item = (usize, usize)> {
    let (nrows, ncols) = shape;
    (0..nrows)
        .map(move |r| (0..ncols).map(move |c| (r, c)))
        .flatten()
}

fn coordinates_from_mask(mask: &DMatrix<bool>) -> Vec<(usize, usize)> {
    let (height, width) = mask.shape();
    let coords = (0..width).map(|x| (0..height).map(move |y| (x, y)));
    extract_sparse(mask.iter().cloned(), coords.flatten()).collect()
}

fn extract_sparse<T, I: Iterator<Item = bool>>(
    sparse_pixels: I,
    mat: impl Iterator<Item = T>,
) -> impl Iterator<Item = T> {
    sparse_pixels
        .zip(mat)
        .filter_map(|(b, v)| if b { Some(v) } else { None })
}

/// Computes the sqrt of the sum of squared values.
/// This is the L2 norm of the vectorized version of the matrix.
fn norm<R: Dim, C: Dim>(matrix: &MatrixMN<f32, R, C>) -> f32
where
    DefaultAllocator: Allocator<f32, R, C>,
{
    norm_sqr(matrix).sqrt() as f32
}

fn norm_sqr<R: Dim, C: Dim>(matrix: &MatrixMN<f32, R, C>) -> f64
where
    DefaultAllocator: Allocator<f32, R, C>,
{
    matrix.iter().map(|&x| (x as f64).powi(2)).sum()
}

/// Mean squared error (MSE)
fn mse<R: Dim, C: Dim>(matrix: &MatrixMN<f32, R, C>) -> f32
where
    DefaultAllocator: Allocator<f32, R, C>,
{
    let npixels = matrix.len();
    (matrix.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / npixels as f64) as f32
}

/// Shrink values toward 0.
fn shrink<T: RealField>(alpha: T, x: T) -> T {
    let alpha = alpha.abs();
    if x.is_sign_positive() {
        (x - alpha).max(T::zero())
    } else {
        (x + alpha).min(T::zero())
    }
}
