use aedat::base::ioheader_generated::Compression;
use clap::Parser;
use davis_edi_rs::util::reconstructor::{show_display, Reconstructor};
use davis_edi_rs::Args;
use opencv::core::{Mat, MatTraitConst, CV_8U};
use std::error::Error;
use std::time::Instant;
use opencv::videoio::VideoWriter;
use opencv::prelude::VideoWriterTrait;

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
        args.optimize_c,
        args.optimize_controller,
        args.show_display,
        args.show_blurred_display,
        args.output_fps,
        Compression::None,
        346,
        260,
        args.deblur_only,
        args.events_only,
        args.target_latency,
        args.simulate_packet_latency,
    )
    .await;
    let mut last_time = Instant::now();
    let first_time = last_time;
    let mut frame_count = 0;
    let mut image_8u = Mat::default();
    let write_video = args.write_video;

    // /mnt/tmp is a mounted ramdisk, eg.:
    // sudo mount -t tmpfs -o rw,size=20G tmpfs /mnt/tmp
    let mut cv_video_writer = VideoWriter::new(
        "/mnt/tmp/tmp.avi",
        opencv::videoio::VideoWriter::fourcc('M' as i8, 'J' as i8, 'P' as i8, 'G' as i8).unwrap(),
        30.0,
        opencv::core::Size::new(reconstructor.width as i32, reconstructor.height as i32),
        false,
    )?;
    loop {
        match reconstructor.next(false).await {
            None => {
                println!("\nFinished!");
                break;
            }
            Some(image_res) => {
                frame_count += 1;
                let image = match image_res {
                    Ok((a, _packet_ts, _)) => a,
                    Err(_) => {
                        panic!("No image")
                    }
                };

                if write_video {
                    image
                        .clone()
                        .convert_to(&mut image_8u, CV_8U, 255.0, 0.0)
                        .unwrap();
                    cv_video_writer.write(&image_8u)?;
                }

                // Don't refresh the window more than 60 Hz
                if (Instant::now() - last_time).as_millis() > args.output_fps as u128 / 60 {
                    last_time = Instant::now();
                    // Iterate through images by pressing a key on keyboard. To iterate automatically,
                    // change `wait` to 1. Break out of loop if user presses a key on keyboard
                    let k = show_display("RETURNED", &image, 1, &reconstructor);
                    if k != -1 {
                        println!("\nExiting by keystroke k={}", k);
                        break;
                    };
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
    cv_video_writer.release()?;
    drop(cv_video_writer);

    Ok(())
}
