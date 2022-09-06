use std::collections::VecDeque;
use std::path::Path;
use aedat::base::{Packet, ParseError, Stream};
use opencv::core::{CV_64F, CV_8S, CV_8U, Mat, MatExprTraitConst, MatTrait, MatTraitConst, MatTraitManual, NORM_MINMAX, Size};
use opencv::highgui;
use opencv::imgproc::resize;
use crate::event_adder_new::{BlurInfo, EventAdderNew};

// TODO: find this through optimization
// pub const C_THRES: f64 = 0.2;

#[derive(Default)]
pub struct BlurredInput {
    pub image: Mat,
    pub exposure_begin_t: i64,
    pub exposure_end_t: i64,
}

pub struct Reconstructor {
    show_display: bool,
    current_blurred_image: BlurredInput,
    aedat_decoder: aedat::base::Decoder,
    height: usize,
    width: usize,
    packet_queue: VecDeque<Packet>,
    t_shift: i64,
    output_frame_length: i64,
    event_adder: EventAdderNew,
    latent_image_queue: VecDeque<Mat>,
}

impl Reconstructor {
    pub fn new(directory: String, aedat_filename: String, start_c: f64, optimize_c: bool, display: bool, output_fps: f64) -> Reconstructor {
        let mut aedat_decoder = aedat::base::Decoder::new(Path::new(&(directory.to_owned() + "/" + &aedat_filename))).unwrap();
        let (height, width) = split_camera_info(&aedat_decoder.id_to_stream[&0]);



        let mut event_counter = Mat::default();

        // Signed integers, to allow for negative polarities dominating the interval
        unsafe { event_counter.create_rows_cols(height as i32, width as i32, CV_8S).unwrap(); }

        let packet_queue = VecDeque::new();
        let mut t_shift;
        let output_frame_length = (1000000.0 / output_fps) as i64;


        // Get the first frame and ignore events before it
        loop {
            match aedat_decoder.next().unwrap() {
                Ok(p) => {
                    if p.stream_id == aedat::base::StreamContent::Frame as u32 {
                        let frame = match aedat::frame_generated::size_prefixed_root_as_frame(&p.buffer) {
                            Ok(result) => result,
                            Err(_) => {
                                panic!("the packet does not have a size prefix");
                            }
                        };
                        t_shift = frame.exposure_end_t();
                        // output_frame_length = (frame.exposure_end_t() - frame.exposure_begin_t()) / frame_exp_divisor;
                        break;
                    }
                }
                _ => {}
            }
        }



        let r = Reconstructor {
            show_display: display,
            current_blurred_image: Default::default(),
            aedat_decoder,
            height: height as usize,
            width: width as usize,
            packet_queue,
            t_shift,
            output_frame_length,
            event_adder: EventAdderNew::new(height as usize, width as usize, t_shift, output_frame_length, start_c, optimize_c),
            latent_image_queue: VecDeque::new(),
        };

        r
    }

    /// Read packets until the next APS frame is reached (inclusive)
    fn fill_packet_queue_to_frame (&mut self) {
        loop {
            match self.aedat_decoder.next().unwrap() {
                Ok(p) => {
                    if p.stream_id == aedat::base::StreamContent::Frame as u32 {
                        let frame = match aedat::frame_generated::size_prefixed_root_as_frame(&p.buffer) {
                            Ok(result) => result,
                            Err(_) => {
                                panic!("the packet does not have a size prefix");
                            }
                        };
                        let mut mat_8u = Mat::zeros(self.height as i32, self.width as i32, CV_8U).unwrap().to_mat().unwrap();
                        let bytes = mat_8u.data_bytes_mut().unwrap();
                        for (idx, px) in bytes.iter_mut().enumerate() {
                            *px = frame.pixels().unwrap()[idx];
                        }
                        let mut mat_64f = Mat::default();
                        mat_8u.convert_to(&mut mat_64f, CV_64F, 1.0/255.0, 0.0).unwrap();

                        let blur_info = BlurInfo::new(
                        mat_64f,
                        frame.exposure_begin_t(),
                        frame.exposure_end_t(),
                        self.t_shift,
                        self.output_frame_length,
                        self.height as i32,
                        self.width as i32,
                        self.event_adder.intervals_popped,
                        );
                        match self.event_adder.blur_info.init {
                            false => {
                                self.event_adder.blur_info = blur_info;
                            }
                            true => {
                                self.event_adder.next_blur_info = blur_info;
                            }
                        }

                        show_display_force("blurred input", &self.event_adder.blur_info.blurred_image, 1, false);
                        return
                    }
                    else if p.stream_id == aedat::base::StreamContent::Events as u32 {
                        self.packet_queue.push_back(p);
                    }
                }
                Err(e) => panic!("{}", e)
            }
        };
    }

    /// Generates reconstructed images from the next packet of events
    fn get_more_images(&mut self) {
        loop {
            // match self.aedat_decoder.next().unwrap() {
            match self.packet_queue.pop_front() {
                Some(p) => {
                    match p.stream_id {
                        a if a == aedat::base::StreamContent::Frame as u32 => { }
                        a if a == aedat::base::StreamContent::Events as u32 => {
                            self.event_adder.sort_events(p);
                        }
                        _ => {println!("debug 2")}
                    }
                }
                _ => {
                    let rett = self.event_adder.deblur_image();
                    self.fill_packet_queue_to_frame();
                }
            }
        }
    }
}



#[derive(Debug)]
pub struct ReconstructionError {
    message: String,
}

impl ReconstructionError {
    pub fn new(message: &str) -> ReconstructionError {
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
            Some(image) => {
                Some(Ok(image))
            },

            // Else we need to rebuild the queue
            _ => {
                self.get_more_images();
                match self.latent_image_queue.pop_front() {
                    None => { panic!("No images in the returned queue")}
                    Some(image) => {

                        // TODO: Split this off so that it can execute in its own thread.
                        // After reaching this point, immediately call it again in thread (maybe
                        // a few times?), so that it runs in the background. This will help hide
                        // the latency
                        self.fill_packet_queue_to_frame();
                        return Some(Ok(image))}
                }

            }

        };
    }
}

fn split_camera_info(stream: &Stream) -> (u16, u16) {
    (stream.height, stream.width)
}

/// If [`MyArgs`]`.show_display`, shows the given [`Mat`] in an OpenCV window
pub fn show_display(window_name: &str, mat: &Mat, wait: i32, reconstructor: &Reconstructor) {
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
pub fn show_display_force(window_name: &str, mat: &Mat, wait: i32, normalize: bool) {
    let mut normed = mat.clone();
    let mut tmp = Mat::default();
    if normalize {
        opencv::core::normalize(&mat, &mut normed, 0.0, 1.0, NORM_MINMAX, -1, &opencv::core::no_array());
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