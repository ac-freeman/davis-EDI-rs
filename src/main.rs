use aedat::base::ioheader_generated::Compression;
use clap::Parser;
use davis_edi_rs::util::reconstructor::{show_display, Reconstructor};
use davis_edi_rs::Args;
use opencv::core::{Mat, MatTraitConst, MatTraitConstManual, CV_8U};
use std::error::Error;
use std::time::Instant;
use tokio::fs::File;
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::process::Command;

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
        args.optimize_controller != 0,
        args.show_display != 0,
        args.show_blurred_display != 0,
        args.output_fps,
        Compression::None,
        346,
        260,
        args.deblur_only != 0,
        args.target_latency,
    )
    .await;
    let mut last_time = Instant::now();
    let first_time = last_time;
    let mut frame_count = 0;
    let mut video_writer = BufWriter::new(File::create("./tmp.gray8").await.unwrap());
    let mut image_8u = Mat::default();
    let write_video = args.write_video != 0;
    loop {
        match reconstructor.next(false).await {
            None => {
                println!("\nFinished!");
                break;
            }
            Some(image_res) => {
                frame_count += 1;
                let image = match image_res {
                    Ok((a, _)) => a,
                    Err(_) => {
                        panic!("No image")
                    }
                };

                image
                    .clone()
                    .convert_to(&mut image_8u, CV_8U, 255.0, 0.0)
                    .unwrap();

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

                if write_video {
                    unsafe {
                        for idx in 0..(reconstructor.height * reconstructor.width) as i32 {
                            let val: *const u8 = image_8u.at_unchecked(idx).unwrap() as *const u8;
                            video_writer
                                .write(std::slice::from_raw_parts(val, 1))
                                .await
                                .unwrap();
                        }
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

    if write_video {
        // ffmpeg -f rawvideo -pix_fmt gray -s:v 346x260 -r 60 -i ./tmp.gray8 -crf 0 -c:v libx264 ./output_file.mp4
        println!("Writing reconstruction as .mp4 with ffmpeg");
        Command::new("ffmpeg")
            .args(&[
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s:v",
                "346x260",
                "-r",
                "30",
                "-i",
                "./tmp.gray8",
                "-crf",
                "0",
                "-c:v",
                "libx264",
                "-y",
                "./output_file.mp4",
            ])
            .output()
            .await
            .expect("failed to execute process");
    }
    Ok(())
}
