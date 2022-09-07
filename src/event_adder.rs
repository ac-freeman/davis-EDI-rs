use std::mem;
use std::ops::Mul;

use aedat::base::Packet;
use aedat::events_generated::Event;
use opencv::core::{
    div_mat_f64, div_mat_matexpr, exp, sub_mat_scalar, sub_scalar_mat, ElemMul, Mat,
    MatExprTraitConst, MatTrait, MatTraitConst, Scalar, CV_64F,
};

use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

pub struct DeblurReturn {
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) ret_vec: Vec<Mat>
}

pub struct EventAdder {
    /// The time span of each reconstructed frame
    interval_t: i64,

    /// Events occurring before the current blurred image
    event_before_queue: Vec<Event>,

    /// Events occurring during the current blurred image
    event_during_queue: Vec<Event>,

    /// Events occurring after the current blurred image
    event_after_queue: Vec<Event>,
    height: i32,
    width: i32,
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) latent_image: Mat,
    pub(crate) blur_info: BlurInfo,
    pub(crate) next_blur_info: BlurInfo,
    current_c: f64,
    optimize_c: bool,
}

unsafe impl Send for EventAdder {}
unsafe impl Sync for EventAdder {}

impl EventAdder {
    pub fn new(
        height: usize,
        width: usize,
        output_frame_length: i64,
        start_c: f64,
        optimize_c: bool,
    ) -> EventAdder {
        EventAdder {
            interval_t: output_frame_length,
            event_before_queue: Vec::new(),
            event_during_queue: Vec::new(),
            event_after_queue: Vec::new(),
            height: height as i32,
            width: width as i32,
            last_interval_start_timestamp: 0,
            latent_image: Mat::zeros(height as i32, width as i32, CV_64F)
                .unwrap()
                .to_mat()
                .unwrap(),
            blur_info: Default::default(),
            next_blur_info: Default::default(),
            current_c: start_c,
            optimize_c,
        }
    }

    pub fn sort_events(&mut self, packet: Packet) {
        let event_packet =
            match aedat::events_generated::size_prefixed_root_as_event_packet(&packet.buffer) {
                Ok(result) => result,
                Err(_) => {
                    panic!("the packet does not have a size prefix");
                }
            };

        let event_arr = match event_packet.elements() {
            None => return,
            Some(events) => events,
        };

        for event in event_arr {
            match event.t() {
                a if a < self.blur_info.exposure_begin_t => {
                    self.event_before_queue.push(*event);
                }
                a if a > self.blur_info.exposure_end_t => {
                    self.event_after_queue.push(*event);
                }
                _ => {
                    self.event_during_queue.push(*event);
                }
            }
        }
    }

    pub fn reset_event_queues(&mut self) {
        mem::swap(&mut self.event_before_queue, &mut self.event_after_queue);
        self.event_after_queue.clear();
        self.event_during_queue.clear();
    }

    fn get_intermediate_image(&self, c: f64, timestamp_start: i64) -> Mat {
        if self.event_before_queue.is_empty() {
            panic!("Empty before queue");
        }

        // let mut latent_log = self.make_log(&self.latent_image);
        // show_display_force("latent_log_0", &latent_log, 1, true);

        // TODO: Need to avoid having to traverse the whole queue each time?
        let start_index = 0;
        let mut end_index = 0;
        loop {
            // if self.event_before_queue[end_index+1].t() > timestamp_start {
            //     start_index = end_index;
            // }
            if end_index + 1 == self.event_before_queue.len()
                || self.event_before_queue[end_index + 1].t() > timestamp_start + self.interval_t {
                break;
            }
            end_index += 1;
        }

        let mut event_counter = Mat::zeros(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        for event in &self.event_before_queue[start_index..end_index] {
            *mat_at_mut::<f64>(&mut event_counter, event) += event_polarity_float(event);
        }

        let mut tmp_mat = self.latent_image.clone();
        // L^tilde(t) = L^tilde(f) + cE(t)
        // Take the exp of L^tilde(t) to get L(t), the final latent image
        event_counter = event_counter
            .clone()
            .mul(c)
            .into_result()
            .unwrap()
            .to_mat()
            .unwrap();
        exp(&event_counter, &mut tmp_mat).unwrap();
        event_counter = self
            .latent_image
            .clone()
            .elem_mul(&tmp_mat)
            .into_result()
            .unwrap()
            .to_mat()
            .unwrap();

        // show_display_force("latent_log", &latent_log, 1, true);
        // show_display_force("latent", &event_counter, 1, false);
        event_counter
    }

    fn get_latent_and_edge(&self, c: f64, timestamp_start: i64) -> Mat {
        if self.event_during_queue.is_empty() {
            panic!("No during queue")
        }
        // TODO: Need to avoid having to traverse the whole queue each time?
        let mut start_index = 0;
        loop {
            if start_index + 1 == self.event_during_queue.len()
                || self.event_during_queue[start_index + 1].t() > timestamp_start {
                break;
            }
            start_index += 1;
        }

        let mut latent_image = Mat::zeros(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        let mut edge_image = Mat::zeros(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();

        let mut event_counter = Mat::zeros(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        let mut timestamps = Mat::ones(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        timestamps = timestamps
            .mul(timestamp_start as f64)
            .into_result()
            .unwrap()
            .to_mat()
            .unwrap();

        // Events occurring AFTER this timestamp
        for event in &self.event_during_queue[start_index..] {
            *mat_at_mut::<f64>(&mut latent_image, event) +=
                (c * *mat_at::<f64>(&event_counter, event)).exp()
                    * (event.t() as f64 - *mat_at::<f64>(&timestamps, event));

            *mat_at_mut::<f64>(&mut event_counter, event) += event_polarity_float(event);
            *mat_at_mut::<f64>(&mut edge_image, event) += event_polarity_float(event)
                * c
                * (-(event.t() as f64 - *mat_at::<f64>(&timestamps, event))).exp();
            *mat_at_mut::<f64>(&mut timestamps, event) = event.t() as f64;
        }
        let mut event_counter_exp = Mat::default();
        exp(
            &event_counter
                .mul(c)
                .into_result()
                .unwrap()
                .to_mat()
                .unwrap(),
            &mut event_counter_exp,
        )
        .unwrap();
        latent_image = (latent_image
            + (event_counter_exp
                .elem_mul(
                    sub_scalar_mat(
                        Scalar::from(self.event_during_queue.last().unwrap().t() as f64),
                        &timestamps,
                    )
                    .unwrap(),
                )
                .into_result()
                .unwrap()))
        .into_result()
        .unwrap()
        .to_mat()
        .unwrap();

        // Events occurring BEFORE this timestamp
        timestamps = Mat::ones(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        timestamps = timestamps
            .mul(timestamp_start as f64)
            .into_result()
            .unwrap()
            .to_mat()
            .unwrap();
        event_counter = Mat::zeros(self.height, self.width, CV_64F)
            .unwrap()
            .to_mat()
            .unwrap();
        for event in &self.event_during_queue[..start_index] {
            *mat_at_mut::<f64>(&mut latent_image, event) +=
                (c * *mat_at::<f64>(&event_counter, event)).exp()
                    * (*mat_at::<f64>(&timestamps, event) - event.t() as f64);

            *mat_at_mut::<f64>(&mut event_counter, event) -= event_polarity_float(event);
            *mat_at_mut::<f64>(&mut edge_image, event) -= event_polarity_float(event)
                * c
                * (-(*mat_at::<f64>(&timestamps, event) - event.t() as f64)).exp();
            *mat_at_mut::<f64>(&mut timestamps, event) = event.t() as f64;
        }
        event_counter_exp = Mat::default();
        exp(
            &event_counter
                .mul(c)
                .into_result()
                .unwrap()
                .to_mat()
                .unwrap(),
            &mut event_counter_exp,
        )
        .unwrap();
        latent_image = (latent_image
            + (event_counter_exp
                .elem_mul(
                    sub_mat_scalar(
                        &timestamps,
                        Scalar::from(self.event_during_queue[0].t() as f64),
                    )
                    .unwrap(),
                )
                .into_result()
                .unwrap()))
        .into_result()
        .unwrap()
        .to_mat()
        .unwrap();

        latent_image = div_mat_matexpr(
            &self.blur_info.blurred_image,
            &div_mat_f64(
                &latent_image,
                self.event_during_queue.last().unwrap().t() as f64
                    - self.event_during_queue[0].t() as f64,
            )
            .unwrap(),
        )
        .unwrap()
        .to_mat()
        .unwrap();

        // show_display_force("latent", &latent_image, 1, false);
        latent_image
    }

    // fn make_log(&self, mat: &Mat) -> Mat {
    //     let mut log_mat = Mat::zeros(self.height as i32, self.width as i32, CV_64F).unwrap().to_mat().unwrap();
    //     for i in 0..self.height as i32 {
    //         for j in 0..self.width as i32 {
    //             let px = mat.at_2d::<f64>(i, j).unwrap();
    //             let log_px = log_mat.at_2d_mut::<f64>(i, j).unwrap();
    //             *log_px = match px.ln() {
    //                 a if a == f64::NEG_INFINITY => {
    //                     0.0
    //                 }
    //                 a if a == f64::INFINITY => {
    //                     panic!("Positive infinity value")
    //                 }
    //                 a if a == f64::
    //                 a => { a }
    //             };
    //         }
    //     }
    //     log_mat
    // }
}

pub fn deblur_image(event_adder: &EventAdder) -> Option<DeblurReturn> {
    if !event_adder.blur_info.init {
        return None;
    }

    // The beginning time for interval 0. Probably before the blurred image exposure beginning time
    let interval_beginning_start =
        ((event_adder.blur_info.exposure_begin_t) / event_adder.interval_t) * event_adder.interval_t;
    let interval_end_start =
        ((event_adder.blur_info.exposure_end_t) / event_adder.interval_t) * event_adder.interval_t;
    let mut ret_vec = Vec::with_capacity(
        ((interval_end_start - interval_beginning_start) / event_adder.interval_t) as usize * 2,
    );

    ////////////////////////
    // First, do the queue'd up events preceding this image. These intermediate images
    // are based on the most recent deblurred latent image
    if event_adder.last_interval_start_timestamp > 0 {
        let mut intermediate_interval_start_timestamps = vec![(
            event_adder.last_interval_start_timestamp + event_adder.interval_t,
            Mat::default(),
        )];
        let mut current_ts = intermediate_interval_start_timestamps[0].0 + event_adder.interval_t;
        loop {
            if current_ts < interval_beginning_start {
                intermediate_interval_start_timestamps.push((current_ts, Mat::default()));
                current_ts += event_adder.interval_t;
            } else {
                break;
            }
        }

        intermediate_interval_start_timestamps
            .par_iter_mut()
            .for_each(|(timestamp_start, mat)| {
                // let c = optimize_c()
                *mat = event_adder.get_intermediate_image(event_adder.current_c, *timestamp_start);
            });

        for elem in intermediate_interval_start_timestamps {
            ret_vec.push(elem.1)
        }
    }

    ////////////////////////

    // Make a vec of these timestamps so we can (eventually) iterate them concurrently
    let mut interval_start_timestamps = vec![(interval_beginning_start, Mat::default())];
    let mut current_ts = interval_beginning_start + event_adder.interval_t;
    loop {
        if current_ts <= interval_end_start {
            interval_start_timestamps.push((current_ts, Mat::default()));
            current_ts += event_adder.interval_t;
        } else {
            break;
        }
    }

    interval_start_timestamps
        .par_iter_mut()
        .for_each(|(timestamp_start, mat)| {
            // let c = optimize_c()
            *mat = event_adder.get_latent_and_edge(event_adder.current_c, *timestamp_start);
        });

    // let mut ret_vec = Vec::with_capacity(interval_start_timestamps.len());
    // self.last_interval_start_timestamp = interval_start_timestamps.last().unwrap().0;
    let last_interval_start_timestamp = interval_start_timestamps.last().unwrap().0;
    for elem in interval_start_timestamps {
        ret_vec.push(elem.1)
    }

    // self.latent_image = ret_vec.last().unwrap().clone();

    Some(DeblurReturn {
        last_interval_start_timestamp,
        ret_vec
    })
}


fn event_polarity_float(event: &Event) -> f64 {
    match event.on() {
        true => 1.0,
        false => -1.0,
    }
}

use opencv::core::DataType;

fn mat_at_mut<'a, T: DataType>(mat: &'a mut Mat, event: &Event) -> &'a mut T {
    mat.at_2d_mut(event.y().into(), event.x().into())
        .expect("Mat error")
}

fn mat_at<'a, T: DataType>(mat: &'a Mat, event: &Event) -> &'a T {
    mat.at_2d(event.y().into(), event.x().into())
        .expect("Mat error")
}

#[derive(Default)]
pub struct BlurInfo {
    pub blurred_image: Mat,
    exposure_begin_t: i64,
    exposure_end_t: i64,
    pub init: bool, // TODO: not very rusty
}

impl BlurInfo {
    pub fn new(
        image: Mat,
        exposure_begin_t: i64,
        exposure_end_t: i64,
    ) -> BlurInfo {
        BlurInfo {
            blurred_image: image,
            exposure_begin_t,
            exposure_end_t,
            init: true,
        }
    }
}
