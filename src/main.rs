use std::error::Error;
use crate::reconstructor::{ReconstructionError, Reconstructor, show_display};
use clap::Parser;
use opencv::core::{CV_8U, Mat, MatTraitConst};
use serde;
use serde::Deserialize;

mod reconstructor;
mod event_adder;
mod event_adder_new;

#[derive(Parser, Debug, Deserialize, Default)]
pub struct Args {
    /// Filename for args (optional; must be in .toml format)
    #[clap(short, long, default_value = "")]
    pub(crate) args_filename: String,

    /// Directory containing the input aedat4 file
    #[clap(short, long, default_value = "")]
    pub(crate) base_path: String,

    /// Name of the input aedat4 file
    #[clap(short, long, default_value = "")]
    pub(crate) events_filename: String,

    /// Starting value for c (contrast threshold)
    #[clap(long, default_value_t = 0.3)]
    pub(crate) start_c: f64,

    /// Optimize c? (0=no, 1=yes)
    /// If no, then the system will only use the start_c value
    #[clap(long, default_value_t = 1)]
    pub(crate) optimize_c: i32,

    /// Show live view display? (0=no, 1=yes)
    #[clap(short, long, default_value_t = 1)]
    pub(crate) show_display: i32,

    /// Output frames per second. Assume that the input file has microsecond subdivision,
    /// i.e., 1000000 ticks per second. Then each output frame will constitute 1000000/[FPS] ticks
    #[clap(short, long, default_value_t = 100.0)]
    pub(crate) output_fps: f64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = Args::parse();
    if !args.args_filename.is_empty() {
        let content = std::fs::read_to_string(args.args_filename)?;
        args = toml::from_str(&content).unwrap();
    }
    let mut reconstructor = Reconstructor::new(
        args.base_path,
        args.events_filename,
        args.start_c,
        args.optimize_c != 0,
        args.show_display != 0,
        args.output_fps,
    );
    loop {
        match reconstructor.next() {
            None => {}
            Some(image) => {
                let image = match image {
                    Ok(a) => {a}
                    Err(_) => {panic!("No image")}
                };
                let mut mat_8u = Mat::default();
                image.convert_to(&mut mat_8u, CV_8U, 1.0, 0.0).unwrap();

                // Iterate through images by pressing a key on keyboard. To iterate automatically,
                // change `wait` to 1
                show_display("RETURNED", &mat_8u, 1, &reconstructor);
            }
        }
    }
    Ok(())
}
