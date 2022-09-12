use crate::reconstructor::{show_display, Reconstructor};
use clap::Parser;
use std::error::Error;
use std::time::Instant;

use serde::Deserialize;

mod event_adder;
mod reconstructor;

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

    /// Show live view display for the blurry input APS images? (0=no, 1=yes)
    #[clap(long, default_value_t = 0)]
    pub(crate) show_blurred_display: i32,

    /// Output frames per second. Assume that the input file has microsecond subdivision,
    /// i.e., 1000000 ticks per second. Then each output frame will constitute 1000000/[FPS] ticks
    #[clap(short, long, default_value_t = 100.0)]
    pub(crate) output_fps: f64,

    /// Use mEDI method? (0=no, 1=yes)
    /// Use mEDI instead of EDI method
    #[clap(long, default_value_t = 1)]
    pub(crate) m_edi: i32,
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
        args.show_blurred_display != 0,
        args.output_fps,
        args.m_edi != 0,
    );
    let mut last_time = Instant::now();
    let first_time = last_time;
    let mut frame_count = 0;
    loop {
        match reconstructor.next() {
            None => {
                println!("\nFinished!");
                break;
            }
            Some(image) => {
                frame_count += 1;
                let image = match image {
                    Ok(a) => a,
                    Err(_) => {
                        panic!("No image")
                    }
                };
                // let mut mat_8u = Mat::default();
                // image.convert_to(&mut mat_8u, CV_8U, 255.0, 0.0).unwrap();

                // Don't refresh the window more than 60 Hz
                if (Instant::now() - last_time).as_millis() > args.output_fps as u128 / 60 {
                    last_time = Instant::now();
                    // Iterate through images by pressing a key on keyboard. To iterate automatically,
                    // change `wait` to 1
                    show_display("RETURNED", &image, 1, &reconstructor);
                }
            }
        }
    }
    println!("Reconstructed {} frames in {} seconds, at an average {} FPS",
             frame_count,
             (Instant::now() - first_time).as_secs(),
             frame_count as f32 / (Instant::now() - first_time).as_secs_f32());
    Ok(())
}
