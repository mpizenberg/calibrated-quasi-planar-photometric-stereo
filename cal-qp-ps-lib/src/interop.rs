// SPDX-License-Identifier: MPL-2.0

//! Interoperability conversions between the image and matrix types.

use image::{DynamicImage, ImageBuffer, Luma, Primitive, Rgb};
use nalgebra::{DMatrix, Scalar};

// Convert a DMatrix into an Image ---------------------------------------------
// -----------------------------------------------------------------------------

/// Convert a matrix into a gray level image.
/// Inverse operation of `matrix_from_image`.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn image_from_matrix<T: Scalar + Primitive>(mat: &DMatrix<T>) -> ImageBuffer<Luma<T>, Vec<T>> {
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        *pixel = Luma([mat[(y as usize, x as usize)]]);
    }
    img_buf
}

/// Convert a `(T,T,T)` RGB matrix into an RGB image.
/// Inverse operation of matrix_from_rgb_image.
///
/// This performs a transposition to accomodate for the
/// column major matrix into the row major image.
#[allow(clippy::cast_possible_truncation)]
pub fn rgb_from_matrix<T: Scalar + Primitive>(
    mat: &DMatrix<(T, T, T)>,
) -> ImageBuffer<Rgb<T>, Vec<T>> {
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (nb_rows, nb_cols) = mat.shape();
    let mut img_buf = ImageBuffer::new(nb_cols as u32, nb_rows as u32);
    for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
        let (r, g, b) = mat[(y as usize, x as usize)];
        *pixel = Rgb([r, g, b]);
    }
    img_buf
}

pub trait ToImage {
    fn to_image(&self) -> DynamicImage;
}

impl ToImage for DMatrix<u8> {
    fn to_image(&self) -> DynamicImage {
        DynamicImage::ImageLuma8(image_from_matrix(self))
    }
}

impl ToImage for DMatrix<u16> {
    fn to_image(&self) -> DynamicImage {
        DynamicImage::ImageLuma16(image_from_matrix(self))
    }
}

impl ToImage for DMatrix<(u8, u8, u8)> {
    fn to_image(&self) -> DynamicImage {
        DynamicImage::ImageRgb8(rgb_from_matrix(self))
    }
}

impl ToImage for DMatrix<(u16, u16, u16)> {
    fn to_image(&self) -> DynamicImage {
        DynamicImage::ImageRgb16(rgb_from_matrix(self))
    }
}

// Convert an Image into a DMatrix ---------------------------------------------
// -----------------------------------------------------------------------------

/// Convert a gray image into a matrix.
/// Inverse operation of `image_from_matrix`.
pub fn matrix_from_image<T: Scalar + Primitive>(img: ImageBuffer<Luma<T>, Vec<T>>) -> DMatrix<T> {
    let (width, height) = img.dimensions();
    DMatrix::from_row_slice(height as usize, width as usize, &img.into_raw())
}

/// Convert an RGB image into a `(T, T, T)` RGB matrix.
/// Inverse operation of `rgb_from_matrix`.
pub fn matrix_from_rgb_image<T: Scalar + Primitive>(
    img: ImageBuffer<Rgb<T>, Vec<T>>,
) -> DMatrix<(T, T, T)> {
    // TODO: improve the suboptimal allocation in addition to transposition.
    let (width, height) = img.dimensions();
    DMatrix::from_iterator(
        width as usize,
        height as usize,
        img.as_raw().chunks_exact(3).map(|s| (s[0], s[1], s[2])),
    )
    .transpose()
}

pub trait IntoDMatrix<P, T: Scalar> {
    fn into_dmatrix(self) -> DMatrix<T>;
}

impl IntoDMatrix<Luma<u8>, u8> for DynamicImage {
    fn into_dmatrix(self) -> DMatrix<u8> {
        assert!(self.as_luma8().is_some(), "This isn't a Luma<u8> image.");
        matrix_from_image(self.into_luma8())
    }
}

impl IntoDMatrix<Luma<u16>, u16> for DynamicImage {
    fn into_dmatrix(self) -> DMatrix<u16> {
        assert!(self.as_luma16().is_some(), "This isn't a Luma<u16> image.");
        matrix_from_image(self.into_luma16())
    }
}

impl IntoDMatrix<Rgb<u8>, (u8, u8, u8)> for DynamicImage {
    fn into_dmatrix(self) -> DMatrix<(u8, u8, u8)> {
        assert!(self.as_rgb8().is_some(), "This isn't a Rgb<u8> image.");
        matrix_from_rgb_image(self.into_rgb8())
    }
}

impl IntoDMatrix<Rgb<u16>, (u16, u16, u16)> for DynamicImage {
    fn into_dmatrix(self) -> DMatrix<(u16, u16, u16)> {
        assert!(self.as_rgb16().is_some(), "This isn't a Rgb<u16> image.");
        matrix_from_rgb_image(self.into_rgb16())
    }
}
