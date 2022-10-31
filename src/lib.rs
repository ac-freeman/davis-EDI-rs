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

    /// Deblur only?
    /// If yes, then the system will only deblur the APS images, and NOT generate the intermediate
    /// image frames. This is useful for transcoding to another event representation
    /// (https://github.com/ac-freeman/adder-codec-rs)
    #[clap(long, action)]
    pub deblur_only: bool,

    #[clap(long, action)]
    pub events_only: bool,

    /// If true (value = 1), then the program simulates the latency of the event packets. This is
    /// useful when the source is a pre-recorded file, so the packets get ingested at the same rate
    /// as though they were being produced by a live camera.
    #[clap(long, action)]
    pub simulate_packet_latency: bool,

    /// The target maximum latency (in milliseconds) between an APS frame packet being decoded from
    /// the camera, and deblurring it.
    #[clap(short, long, default_value_t = 200.0)]
    pub target_latency: f64,

    /// Optimize c?
    /// If no, then the system will only use the start_c value
    #[clap(long, action)]
    pub optimize_c: bool,

    /// Enable the optimization controller?
    /// If no, then the system will maintain a constant reconstruction frame rate, and may fall
    /// behind real time. If yes, then the controller will adjust the reconstruction rate and
    /// c optimization frequency to try to maintain real-time performance.
    #[clap(long, action)]
    pub optimize_controller: bool,

    /// Show live view display?
    #[clap(short, long, action)]
    pub show_display: bool,

    /// Show live view display for the blurry input APS images?
    #[clap(long, action)]
    pub show_blurred_display: bool,

    /// Output frames per second. Assume that the input file has microsecond subdivision,
    /// i.e., 1000000 ticks per second. Then each output frame will constitute 1000000/[FPS] ticks
    #[clap(short, long, default_value_t = 100.0)]
    pub output_fps: f64,

    /// Write out framed video reconstruction?
    #[clap(long, action)]
    pub write_video: bool,
}