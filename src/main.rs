// use crate::reconstructor::{show_display, Reconstructor, Reconstructors, ReconstructionError};
use aedat::base::ioheader_generated::Compression;
use clap::Parser;
use std::error::Error;
use std::time::Instant;
use cv_convert::IntoCv;
use nalgebra::{DMatrix, Dynamic, OMatrix, U2, U3};
use opencv::core::{CV_8U, Mat, MatTraitConst, MatTraitConstManual, Size};
use opencv::hub_prelude::VideoWriterTrait;
use opencv::videoio::VideoWriter;

use crate::reconstructor::show_display;
use crate::reconstructor::Reconstructor;
use serde::Deserialize;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::process::Command;

mod event_adder;
mod reconstructor;
mod threaded_decoder;

#[derive(Parser, Debug, Deserialize, Default)]
pub struct Args {
    /// Filename for args (optional; must be in .toml format)
    #[clap(short, long, default_value = "")]
    pub(crate) args_filename: String,

    /// Input mode. Valid options are "file", "socket", and "tcp"
    #[clap(short, long, default_value = "file")]
    pub(crate) mode: String,

    /// Directory containing the input aedat4 file
    #[clap(short, long, default_value = "")]
    pub(crate) base_path: String,

    /// Name of the input aedat4 file
    #[clap(long, default_value = "")]
    pub(crate) events_filename_0: String,

    /// Name of the input aedat4 file
    #[clap(long, default_value = "")]
    pub(crate) events_filename_1: String,

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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Args = Args::parse();
    if !args.args_filename.is_empty() {
        let content = std::fs::read_to_string(args.args_filename)?;
        args = toml::from_str(&content).unwrap();
    }

    let mut reconstructor = Reconstructor::new(
        args.base_path,
        args.events_filename_0,
        args.events_filename_1,
        args.mode,
        args.start_c,
        args.optimize_c != 0,
        args.show_display != 0,
        args.show_blurred_display != 0,
        args.output_fps,
        Compression::None,
        346,
        260,
    )
    .await;
    let mut last_time = Instant::now();
    let first_time = last_time;
    let mut frame_count = 0;
    let mut video_writer = BufWriter::new(File::create("./tmp.gray8").await.unwrap());
    let mut image_8u = Mat::default();
    loop {
        match reconstructor.next().await {
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


                image.clone().convert_to(&mut image_8u, CV_8U, 255.0, 0.0).unwrap();

                // Don't refresh the window more than 60 Hz
                if (Instant::now() - last_time).as_millis() > args.output_fps as u128 / 60 {
                    last_time = Instant::now();
                    // Iterate through images by pressing a key on keyboard. To iterate automatically,
                    // change `wait` to 1. Break out of loop if user presses a key on keyboard
                    let k =  show_display("RETURNED", &image, 1, &reconstructor);
                    if k != -1
                    {
                        println!("\nExiting by keystroke k={}",k);
                        break
                    };

                }

                unsafe {
                    for idx in 0..(reconstructor.height * reconstructor.width) as i32{

                        let val: *const u8 = image_8u.at_unchecked(idx).unwrap() as *const u8;
                        video_writer.write(
                            std::slice::from_raw_parts(val, 1)
                        ).await.unwrap();
                    }
                }
            }
        }
    }
    println!(
        "Reconstructed {} frames in {} seconds, at an average {} FPS",
        frame_count,
        (Instant::now() - first_time).as_secs(),
        frame_count as f32 / (Instant::now() - first_time).as_secs_f32()
    );
    video_writer.flush().await.unwrap();
    drop(video_writer);

    // ffmpeg -f rawvideo -pix_fmt gray -s:v 346x260 -r 60 -i ./tmp.gray8 -crf 0 -c:v libx264 ./output_file.mp4
    println!("Writing reconstruction as .mp4 with ffmpeg");
    Command::new("ffmpeg")
        .args(&["-f", "rawvideo", "-pix_fmt", "gray", "-s:v", "346x260", "-r", "30", "-i", "./tmp.gray8", "-crf", "0", "-c:v", "libx264", "-y", "./output_file.mp4"])
        .output()
        .await.expect("failed to execute process");
    Ok(())
}
