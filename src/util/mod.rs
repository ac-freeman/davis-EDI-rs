use std::arch::x86_64::{_mm_loadu_pd, _mm_storeu_pd};
use nalgebra::{Dyn, OMatrix};
use opencv::core::{Mat, MatExprTraitConst, MatTrait, CV_64F};
use std::arch::x86_64::*;
use std::arch::x86_64::{_mm256_loadu_pd, _mm256_storeu_pd};



pub(crate) mod event_adder;
pub mod reconstructor;
mod threaded_decoder;

fn omatrix_to_mat(omatrix: &OMatrix<f64, Dyn, Dyn>) -> Mat {
    let rows = omatrix.nrows() as i32;
    let cols = omatrix.ncols() as i32;
    let mut mat = Mat::zeros(rows, cols, CV_64F).unwrap().to_mat().unwrap();

    unsafe {
        let mat_ptr = mat.ptr_mut(0).unwrap() as *mut f64;
        let omatrix_ptr = omatrix.transpose().as_slice().as_ptr();

        let len = (rows * cols) as isize;
        let simd_width = 4; // Number of f64 values processed per SIMD operation

        let mut i = 0;
        while i <= len - simd_width {
            let data = _mm256_loadu_pd(omatrix_ptr.offset(i));
            _mm256_storeu_pd(mat_ptr.offset(i), data);
            i += simd_width;
        }

        // Handle remaining elements
        while i < len {
            *mat_ptr.offset(i) = *omatrix_ptr.offset(i);
            i += 1;
        }
    }

    mat
}