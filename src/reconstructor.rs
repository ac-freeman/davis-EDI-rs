use crate::event_adder::{BlurInfo, deblur_image, EventAdder};
use aedat::base::{Decoder, Packet, ParseError, Stream, StreamContent};
use aedat::base::ioheader_generated::Compression;

use opencv::core::{
    Mat, MatTrait, MatTraitConst, Size, CV_8S,
    NORM_MINMAX,
};
use opencv::highgui;
use opencv::imgproc::resize;
use std::collections::VecDeque;
use std::{io, mem};
use std::fs::File;
use std::io::{BufRead, Cursor, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::thread::sleep;
use std::time::{Duration, Instant};
use simple_error::SimpleError;
use crossbeam_utils::thread;
use nalgebra::DMatrix;
use cv_convert::TryFromCv;
use byteorder::{LittleEndian, ReadBytesExt};

#[derive(Default)]
pub struct BlurredInput {
    pub image: Mat,
    pub exposure_begin_t: i64,
    pub exposure_end_t: i64,
}
unsafe impl Sync for Reconstructor  {}
unsafe impl Send for Reconstructor  {}

pub struct Reconstructor {
    show_display: bool,
    show_blurred_display: bool,
    aedat_decoder_0: aedat::base::Decoder,
    aedat_decoder_1: Option<aedat::base::Decoder>,
    height: usize,
    width: usize,
    packet_queue: VecDeque<Packet>,
    event_adder: EventAdder,
    latent_image_queue: VecDeque<Mat>,
}

impl Reconstructor {
    pub fn new(
        directory: String,
        aedat_filename: String,
        mode: String,
        start_c: f64,
        optimize_c: bool,
        display: bool,
        blurred_display: bool,
        output_fps: f64,
        compression: Compression,
        mut width: u16,
        mut height: u16
    ) -> Reconstructor
    {
        let mut decoder_0= match mode.as_str() {
            "file" => {
                aedat::base::Decoder::new_from_file(Path::new(&(directory.clone() + "/" + &aedat_filename))).unwrap()
            }
            "socket" => {
                aedat::base::Decoder::new_from_unix_stream(
                    Path::new(&(directory.clone() + "/" + &aedat_filename)),
                    StreamContent::Events,
                    compression,
                    width,
                    height
                ).unwrap()
            }
            "tcp" => {
                aedat::base::Decoder::new_from_tcp_stream(
                    &(directory.clone() + "/" + &aedat_filename),
                    StreamContent::Events,
                    compression,
                    width,
                    height
                ).unwrap()
            }
            _ => panic!("")
        };

        let mut decoder_1= match mode.as_str() {
            "file" => {
                (height, width) = split_camera_info(&decoder_0.id_to_stream[&0]);
                None
            }
            "socket" => {
                Some(aedat::base::Decoder::new_from_unix_stream(
                    Path::new(&(directory + "/" + &aedat_filename)),
                    StreamContent::Frame,
                    compression,
                    width,
                    height
                ).unwrap())
            }
            "tcp" => {
                Some(aedat::base::Decoder::new_from_tcp_stream(
                    &(directory + "/" + &aedat_filename),
                    StreamContent::Frame,
                    compression,
                    width,
                    height
                ).unwrap())
            }
            _ => panic!("")
        };


        let mut event_counter = Mat::default();

        // Signed integers, to allow for negative polarities dominating the interval
        unsafe {
            event_counter
                .create_rows_cols(height as i32, width as i32, CV_8S)
                .unwrap();
        }

        let packet_queue: VecDeque<Packet> = VecDeque::new();
        let output_frame_length = (1000000.0 / output_fps) as i64;

        // Get the first frame and ignore events before it
        loop {
            if let Ok(p) = decoder_0.next().unwrap() {
                if matches!(decoder_0.id_to_stream.get(&p.stream_id).unwrap().content, aedat::base::StreamContent::Frame) {
                    match aedat::frame_generated::size_prefixed_root_as_frame(&p.buffer)
                    {
                        Ok(result) => result,
                        Err(_) => {
                            panic!("the packet does not have a size prefix");
                        }
                    };
                    break;
                }
            }
        }

        let mut r = Reconstructor {
            show_display: false,
            show_blurred_display: false,
            aedat_decoder_0: decoder_0,
            aedat_decoder_1: decoder_1,
            height: 0,
            width: 0,
            packet_queue: Default::default(),
            event_adder: EventAdder::new(
                height as usize,
                width as usize,
                output_frame_length,
                start_c,
                optimize_c,
            ),
            latent_image_queue: Default::default()
        };
        let blur_info = fill_packet_queue_to_frame(
            &mut r.aedat_decoder_0,
            &mut r.packet_queue,
            r.height as i32,
            r.width as i32,
        ).unwrap();
        r.event_adder.blur_info = Some(blur_info);

        r
    }


    /// Generates reconstructed images from the next packet of events
    fn get_more_images(&mut self) -> Result<(), SimpleError>{

        while let Some(p) = self.packet_queue.pop_front() {
            match self.aedat_decoder_0.id_to_stream.get(&p.stream_id).unwrap().content {
                    aedat::base::StreamContent::Frame => {}
                    aedat::base::StreamContent::Events => {
                        self.event_adder.sort_events(p);
                    }
                    _ => {
                        println!("debug 2")
                    }
            }
        }

        match thread::scope(|s| {
            let join_handle = s.spawn(|_| {
                if self.show_blurred_display {
                    let tmp_blurred_mat = Mat::try_from_cv(&self.event_adder.blur_info.as_ref().unwrap().blurred_image).unwrap();
                    _show_display_force("blurred input", &tmp_blurred_mat, 1, false);
                }
                deblur_image(&self.event_adder)
            });

            let next_blur_info = match fill_packet_queue_to_frame(
                &mut self.aedat_decoder_0,
                &mut self.packet_queue,
                self.height as i32,
                self.width as i32,
            ) {
                Ok(blur_info) => { Some(blur_info) },
                Err(_) => None
            };

            (join_handle.join().unwrap(), next_blur_info)
        }) {
            Ok((None, _)) => {
                panic!("No images returned from deblur call")
            }
            Ok((Some(deblur_return), Some(next_blur_info))) => {
                self.event_adder.latent_image = deblur_return.ret_vec.last().unwrap().clone();
                self.event_adder.last_interval_start_timestamp = deblur_return.last_interval_start_timestamp;
                self.latent_image_queue.append(&mut VecDeque::from(deblur_return.ret_vec));
                self.event_adder.reset_event_queues();
                self.event_adder.next_blur_info = Some(next_blur_info);
                self.event_adder.current_c = deblur_return.found_c;
            }
            _ => {
                return Err(SimpleError::new("End of aedat file"))
            }
        };

        Ok(())
    }
}

/// Read packets until the next APS frame is reached (inclusive)
fn fill_packet_queue_to_frame(
    aedat_decoder: &mut aedat::base::Decoder,
    packet_queue: &mut VecDeque<Packet>,
    height: i32,
    width: i32) -> Result<BlurInfo, SimpleError> {
    loop {
        match aedat_decoder.next() {
            Some(Ok(p)) => {
                if matches!(aedat_decoder.id_to_stream.get(&p.stream_id).unwrap().content, aedat::base::StreamContent::Frame) {
                    let frame =
                        match aedat::frame_generated::size_prefixed_root_as_frame(&p.buffer) {
                            Ok(result) => result,
                            Err(_) => {
                                panic!("the packet does not have a size prefix");
                            }
                        };

                    let frame_px = frame.pixels().unwrap();
                    let mut image = DMatrix::<f64>::zeros(height as usize, width as usize);
                    for (row_idx, mut im_row) in image.row_iter_mut().enumerate() {
                        for (col_idx, im_px) in im_row.iter_mut().enumerate() {
                            *im_px = frame_px[row_idx * width as usize + col_idx] as f64 / 255.0;
                        }
                    }

                    // TODO: TMP
                    let tmp_blurred_mat = Mat::try_from_cv(&image).unwrap();
                    _show_display_force("blurred input", &tmp_blurred_mat, 1, false);

                    let blur_info = BlurInfo::new(
                        image,
                        frame.exposure_begin_t(),
                        frame.exposure_end_t(),
                    );

                    // return Ok(blur_info);
                } else if matches!(aedat_decoder.id_to_stream.get(&p.stream_id).unwrap().content, aedat::base::StreamContent::Events) {
                    packet_queue.push_back(p);
                }
            }
            Some(Err(e)) => panic!("{}", e),
            None => return Err(SimpleError::new("End of aedat file"))
        }
    }
}

#[derive(Debug)]
pub struct ReconstructionError {
    message: String,
}

impl ReconstructionError {
    pub fn _new(message: &str) -> ReconstructionError {
        ReconstructionError {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ReconstructionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::convert::From<ParseError> for ReconstructionError {
    fn from(error: ParseError) -> Self {
        ReconstructionError {
            message: error.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatentImage {
    pub frame: Mat,
}

impl Iterator for Reconstructor {
    type Item = Result<Mat, ReconstructionError>;

    /// Get the next reconstructed image
    fn next(&mut self) -> Option<Self::Item> {
        return match self.latent_image_queue.pop_front() {
            // If we have a queue of images already, just return the next one
            Some(image) => Some(Ok(image)),

            // Else we need to rebuild the queue
            _ => {
                let now = Instant::now();

                if self.event_adder.next_blur_info.is_some() {
                    mem::swap(&mut self.event_adder.blur_info, &mut self.event_adder.next_blur_info);
                    self.event_adder.next_blur_info = None;
                }
                //
                //     self.fill_packet_queue_to_frame()


                // let join_handle: thread::JoinHandle<_> = thread::spawn(|| {
                match self.get_more_images() {
                    Ok(_) => {}
                    Err(_) => return None
                };
                // });
                let running_fps = self.latent_image_queue.len() as f64
                    / now.elapsed().as_millis() as f64 * 1000.0;
                print!(
                    "\r{} frames in  {}ms -- Current FPS: {:.2}, Current c: {:.5}",
                    self.latent_image_queue.len(),
                    now.elapsed().as_millis(),
                    running_fps,
                    self.event_adder.current_c
                );
                io::stdout().flush().unwrap();
                match self.latent_image_queue.pop_front() {
                    None => {
                        panic!("No images in the returned queue")
                    }
                    Some(image) => {
                        return Some(Ok(image));
                    }
                }
            }
        };
    }
}

fn split_camera_info(stream: &Stream) -> (u16, u16) {
    (stream.height, stream.width)
}

/// If [`MyArgs`]`.show_display`, shows the given [`Mat`] in an OpenCV window
pub fn show_display(window_name: &str, mat: &Mat, wait: i32, reconstructor: Reconstructor) {
    if reconstructor.show_display {
        let mut tmp = Mat::default();

        if mat.rows() != 540 {
            let factor = mat.rows() as f32 / 540.0;
            resize(
                mat,
                &mut tmp,
                Size {
                    width: (mat.cols() as f32 / factor) as i32,
                    height: 540,
                },
                0.0,
                0.0,
                0,
            )
            .unwrap();
            highgui::imshow(window_name, &tmp).unwrap();
        } else {
            highgui::imshow(window_name, mat).unwrap();
        }
        highgui::wait_key(wait).unwrap();
    }
}

/// TODO: Remove. Just for debugging.
pub fn _show_display_force(window_name: &str, mat: &Mat, wait: i32, normalize: bool) {
    let mut normed = mat.clone();
    let mut tmp = Mat::default();
    if normalize {
        opencv::core::normalize(
            &mat,
            &mut normed,
            0.0,
            1.0,
            NORM_MINMAX,
            -1,
            &opencv::core::no_array(),
        ).unwrap();
    }

    if mat.rows() != 540 {
        let factor = mat.rows() as f32 / 540.0;
        resize(
            &normed,
            &mut tmp,
            Size {
                width: (mat.cols() as f32 / factor) as i32,
                height: 540,
            },
            0.0,
            0.0,
            0,
        )
        .unwrap();
        highgui::imshow(window_name, &tmp).unwrap();
    } else {
        highgui::imshow(window_name, mat).unwrap();
    }
    highgui::wait_key(wait).unwrap();
}
