pub mod util;

use clap::Parser;
use serde::Deserialize;

#[derive(Parser, Debug, Deserialize, Default)]
pub struct Args {
    /// Filename for args (optional; must be in .toml format)
    #[clap(short, long, default_value = "")]
    pub args_filename: String,

    /// Input mode. Valid options are "file", "socket", and "tcp"
    #[clap(short, long, default_value = "file")]
    pub mode: String,

    /// Directory containing the input aedat4 file
    #[clap(short, long, default_value = "")]
    pub base_path: String,

    /// Name of the input aedat4 file
    #[clap(long, default_value = "")]
    pub events_filename_0: String,

    /// Name of the input aedat4 file
    #[clap(long, default_value = "")]
    pub events_filename_1: String,

    /// Starting value for c (contrast threshold)
    #[clap(long, default_value_t = 0.3)]
    pub start_c: f64,

    /// Deblur only? (0=no, 1=yes)
    /// If yes, then the system will only deblur the APS images, and NOT generate the intermediate
    /// image frames. This is useful for transcoding to another event representation
    /// (https://github.com/ac-freeman/adder-codec-rs)
    #[clap(long, default_value_t = 1)]
    pub deblur_only: i32,

    #[clap(long, default_value_t = 1)]
    pub events_only: i32,

    /// The target maximum latency (in milliseconds) between an APS frame packet being decoded from the camera, and
    /// deblurring it.
    #[clap(short, long, default_value_t = 200.0)]
    pub target_latency: f64,

    /// Optimize c? (0=no, 1=yes)
    /// If no, then the system will only use the start_c value
    #[clap(long, default_value_t = 1)]
    pub optimize_c: i32,

    /// Enable the optimization controller? (0=no, 1=yes)
    /// If no, then the system will maintain a constant reconstruction frame rate, and may fall
    /// behind real time. If yes, then the controller will adjust the reconstruction rate and
    /// c optimization frequency to try to maintain real-time performance.
    #[clap(long, default_value_t = 1)]
    pub optimize_controller: i32,

    /// Show live view display? (0=no, 1=yes)
    #[clap(short, long, default_value_t = 1)]
    pub show_display: i32,

    /// Show live view display for the blurry input APS images? (0=no, 1=yes)
    #[clap(long, default_value_t = 0)]
    pub show_blurred_display: i32,

    /// Output frames per second. Assume that the input file has microsecond subdivision,
    /// i.e., 1000000 ticks per second. Then each output frame will constitute 1000000/[FPS] ticks
    #[clap(short, long, default_value_t = 100.0)]
    pub output_fps: f64,

    /// Write out framed video reconstruction? (0=no, 1=yes)
    #[clap(long, default_value_t = 0)]
    pub write_video: i32,
}