use nalgebra::{Dyn, OMatrix};
use opencv::core::{Mat, MatExprTraitConst, MatTrait, CV_64F};

pub(crate) mod event_adder;
pub mod reconstructor;
mod threaded_decoder;

fn omatrix_to_mat(omatrix: &OMatrix<f64, Dyn, Dyn>) -> Mat {
    let rows = omatrix.nrows() as i32;
    let cols = omatrix.ncols() as i32;
    let mut mat = Mat::zeros(rows, cols, CV_64F).unwrap().to_mat().unwrap();

    for i in 0..rows {
        for j in 0..cols {
            *mat.at_2d_mut::<f64>(i, j).unwrap() = omatrix[(i as usize, j as usize)];
        }
    }

    mat
}