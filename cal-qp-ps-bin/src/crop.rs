// SPDX-License-Identifier: MPL-2.0

use nalgebra::{DMatrix, Scalar};
use std::convert::TryFrom;
use thiserror::Error;

#[cfg(feature = "wasm-bindgen")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "serde")]
use serde::Deserialize;

#[cfg_attr(feature = "wasm-bindgen", wasm_bindgen)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct Crop {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

#[derive(Error, Debug)]
pub enum CropError {
    #[error("Invalid crop frame coordinates: {0}")]
    InvalidFrame(String),
    #[error("Not enough arguments: expected 4 but got only {0}")]
    NotEnoughArgs(usize),
    #[error("Too many arguments: expected 4 but got more")]
    TooManyArgs,
    #[error("Error parsing crop frame coordinates")]
    Parse(#[from] std::num::ParseIntError),
}

impl TryFrom<Vec<&str>> for Crop {
    type Error = CropError;
    fn try_from(vs: Vec<&str>) -> Result<Self, Self::Error> {
        let mut vs = vs.iter();
        match (vs.next(), vs.next(), vs.next(), vs.next(), vs.next()) {
            (None, _, _, _, _) => Err(CropError::NotEnoughArgs(0)),
            (_, None, _, _, _) => Err(CropError::NotEnoughArgs(1)),
            (_, _, None, _, _) => Err(CropError::NotEnoughArgs(2)),
            (_, _, _, None, _) => Err(CropError::NotEnoughArgs(3)),
            (_, _, _, _, Some(_)) => Err(CropError::TooManyArgs),
            (Some(left), Some(top), Some(right), Some(bottom), None) => Ok(Crop {
                left: left.parse()?,
                top: top.parse()?,
                right: right.parse()?,
                bottom: bottom.parse()?,
            }),
        }
    }
}

pub fn crop<T: Scalar>(frame: Crop, img: &DMatrix<T>) -> Result<DMatrix<T>, CropError> {
    let Crop {
        left,
        top,
        right,
        bottom,
    } = frame;
    let (height, width) = img.shape();

    // Check that the frame coordinates make sense.
    if left >= width {
        return Err(CropError::InvalidFrame(format!(
            "left >= width ({} >= {})",
            left, width
        )));
    }
    if right > width {
        return Err(CropError::InvalidFrame(format!(
            "right > width ({} > {})",
            right, width
        )));
    }
    if top >= height {
        return Err(CropError::InvalidFrame(format!(
            "top >= height ({} >= {})",
            top, height
        )));
    }
    if bottom > height {
        return Err(CropError::InvalidFrame(format!(
            "bottom > height ({} > {})",
            bottom, height
        )));
    }
    if left >= right {
        return Err(CropError::InvalidFrame(format!(
            "left >= right ({} >= {})",
            left, right
        )));
    }
    if top >= bottom {
        return Err(CropError::InvalidFrame(format!(
            "top >= bottom ({} >= {})",
            top, bottom
        )));
    }
    // Extract the cropped slice.
    let nrows = bottom - top;
    let ncols = right - left;
    Ok(img.slice((top, left), (nrows, ncols)).into_owned())
}
