use aedat::base::Packet;
use aedat::events_generated::Event;
use cv_convert::TryFromCv;
use nalgebra::{DMatrix, Dynamic, OMatrix};
use opencv::core::{
    create_continuous, mean, no_array, normalize, sqrt, sum_elems, ElemMul, Mat, MatExprTraitConst,
    BORDER_DEFAULT, CV_64F, NORM_MINMAX,
};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::mem;
use std::ops::{AddAssign, DivAssign, MulAssign};
use std::time::Instant;

const FIB: [f64; 22] = [
    1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0, 377.0, 610.0, 987.0,
    1597.0, 2584.0, 4181.0, 6765.0, 10946.0, 17711.0,
];

pub struct DeblurReturn {
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) ret_vec: Vec<Mat>,
    pub(crate) found_c: f64,
}

pub struct EventAdder {
    /// The time span of each reconstructed frame
    pub interval_t: i64,

    interval_count: u32,

    /// Events occurring before the current blurred image
    pub(crate) event_before_queue: Vec<Event>,

    /// Events occurring during the current blurred image
    pub(crate) event_during_queue: Vec<Event>,

    /// Events occurring after the current blurred image
    pub(crate) event_after_queue: Vec<Event>,
    height: i32,
    width: i32,
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) latent_image: Mat,
    pub(crate) blur_info: Option<BlurInfo>,
    pub(crate) next_blur_info: Option<BlurInfo>,
    pub(crate) current_c: f64,
    pub(crate) optimize_c: bool,
    pub(crate) optimize_c_frequency: u32,
    pub(crate) deblur_only: bool,
    pub(crate) events_only: bool,
}

unsafe impl Send for EventAdder {}
unsafe impl Sync for EventAdder {}

impl EventAdder {
    pub fn new(
        height: u16,
        width: u16,
        output_frame_length: i64,
        start_c: f64,
        optimize_c: bool,
        optimize_c_frequency: u32,
        deblur_only: bool,
        events_only: bool,
    ) -> EventAdder {
        let mut continuous_mat = Mat::default();
        create_continuous(height as i32, width as i32, CV_64F, &mut continuous_mat).unwrap();
        EventAdder {
            interval_t: output_frame_length,
            interval_count: 0,
            event_before_queue: Vec::new(),
            event_during_queue: Vec::new(),
            event_after_queue: Vec::new(),
            height: height as i32,
            width: width as i32,
            last_interval_start_timestamp: 0,
            latent_image: continuous_mat,
            blur_info: None,
            next_blur_info: Default::default(),
            current_c: start_c,
            optimize_c,
            optimize_c_frequency,
            deblur_only,
            events_only,
        }
    }

    pub fn sort_events(&mut self, packet: Packet) {
        let blur_info = match &self.blur_info {
            None => {
                panic!("blur_info not initialized")
            }
            Some(a) => a,
        };
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
                // _ if self.events_only => {
                //     self.event_after_queue.push(*event);
                // }
                a if a < blur_info.exposure_begin_t => {
                    self.event_before_queue.push(*event);
                }
                a if a > blur_info.exposure_end_t => {
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
        // self.event_before_queue.clear();
    }

    fn get_intermediate_image(&self, c: f64, timestamp_start: i64) -> Mat {
        if self.event_before_queue.is_empty() {
            panic!("Empty before queue");
        }

        // TODO: Need to avoid having to traverse the whole queue each time?
        let start_index = 0;
        let mut end_index = 0;
        loop {
            if end_index + 1 == self.event_before_queue.len()
                || self.event_before_queue[end_index + 1].t() > timestamp_start + self.interval_t
            {
                break;
            }
            end_index += 1;
        }

        let mut event_counter = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        let (mut y, mut x);
        for event in &self.event_before_queue[start_index..end_index] {
            y = event.y() as usize;
            x = event.x() as usize;
            event_counter[(y, x)] += event_polarity_float(event);
        }

        // L^tilde(t) = L^tilde(f) + cE(t)
        // Take the exp of L^tilde(t) to get L(t), the final latent image
        event_counter.mul_assign(c);
        event_counter = event_counter.map(|x: f64| x.exp());
        let event_counter_mat = Mat::try_from_cv(event_counter).unwrap();

        self.latent_image
            .clone()
            .elem_mul(&event_counter_mat)
            .into_result()
            .unwrap()
            .to_mat()
            .unwrap()
    }

    // TODO: Vary the rate of optimizing c based on the reconstruction frame rate (vs the target fps)
    pub(crate) fn optimize_c(&self, timestamp_start: i64) -> f64 {
        // Fibonacci search
        let mut a: f64 = 0.1;
        let mut b: f64 = 0.5;
        let n_points = 15.0;
        let mut fib_index = 3;
        while FIB[fib_index] < n_points {
            fib_index += 1;
        }

        let mut x1 = a + FIB[fib_index - 2] / FIB[fib_index] * (b - a);
        let mut x2 = b - FIB[fib_index - 2] / FIB[fib_index] * (b - a);
        let mut fx1 = self.get_phi(x1, timestamp_start);
        let mut fx2 = self.get_phi(x2, timestamp_start);

        for k in 1..fib_index - 2 {
            if fx1 < fx2 {
                b = x2;
                x2 = x1;
                fx2 = fx1;
                x1 = a + FIB[fib_index - k - 1] / FIB[fib_index - k + 1] * (b - a);
                fx1 = self.get_phi(x1, timestamp_start);
            } else {
                a = x1;
                x1 = x2;
                fx1 = fx2;
                x2 = b - FIB[fib_index - k - 1] / FIB[fib_index - k + 1] * (b - a);
                fx2 = self.get_phi(x2, timestamp_start);
            }
        }
        if fx1 < fx2 {
            x1
        } else {
            x2
        }
    }

    fn get_phi(&self, c: f64, timestamp_start: i64) -> f64 {
        let (latent_image, mt_image) = self.get_latent_and_edge(c, timestamp_start);
        // _show_display_force("mt_image", &mt_image, 1, true);

        let (latent_grad, latent_edges) = self.get_gradient_and_edges(latent_image);
        // _show_display_force("grad", &latent_grad, 1, false);
        // _show_display_force("grad_edges", &latent_edges, 1, false);
        let (_mt_grad, mt_edges) = self.get_gradient_and_edges(mt_image);

        let phi_edge = sum_elems(
            &latent_edges
                .elem_mul(mt_edges)
                .into_result()
                .unwrap()
                .to_mat()
                .unwrap(),
        )
        .unwrap()
        .0[0];
        // dbg!(phi_edge);

        let phi_tv = sum_elems(&latent_grad).unwrap().0[0];
        // dbg!(phi_tv);

        // dbg!(phi);
        0.15 * phi_tv - phi_edge
    }

    fn get_gradient_and_edges(&self, image: Mat) -> (Mat, Mat) {
        let mut image_sobel_x = Mat::default();
        sobel(
            &image,
            &mut image_sobel_x,
            CV_64F,
            1,
            0,
            3,
            1.0,
            0.0,
            BORDER_DEFAULT,
        )
        .expect("Sobel error");

        let mut image_sobel_y = Mat::default();
        sobel(
            &image,
            &mut image_sobel_y,
            CV_64F,
            0,
            1,
            3,
            1.0,
            0.0,
            BORDER_DEFAULT,
        )
        .expect("Sobel error");
        let tmp = (image_sobel_x.clone().elem_mul(&image_sobel_x)
            + image_sobel_y.clone().elem_mul(&image_sobel_y))
        .into_result()
        .unwrap()
        .to_mat()
        .unwrap();

        let mut grad = Mat::default();
        sqrt(&tmp, &mut grad).unwrap();

        let mut grad_norm = Mat::default();
        normalize(
            &grad,
            &mut grad_norm,
            0.0,
            1.0,
            NORM_MINMAX,
            -1,
            &no_array(),
        )
        .expect("Norm error");

        let mut thresholded = Mat::default();
        let mut threshold_val = mean(&grad_norm, &no_array()).unwrap().0[0];
        threshold_val += (1.0 - threshold_val) / 3.0;
        threshold(
            &grad_norm,
            &mut thresholded,
            threshold_val,
            1.0,
            THRESH_BINARY,
        )
        .unwrap();

        (grad, thresholded)
    }

    fn get_latent_and_edge(&self, c: f64, timestamp_start: i64) -> (Mat, Mat) {
        let mut latent_image = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);
        let mut edge_image = latent_image.clone();
        if self.event_during_queue.is_empty() {
            return (
                Mat::try_from_cv(self.blur_info.as_ref().unwrap().blurred_image.clone_owned())
                    .unwrap(),
                Mat::try_from_cv(edge_image).unwrap(),
            );
        }

        // TODO: Need to avoid having to traverse the whole queue each time?
        let mut start_index = 0;
        loop {
            if start_index + 1 == self.event_during_queue.len()
                || self.event_during_queue[start_index + 1].t() > timestamp_start
            {
                break;
            }
            start_index += 1;
        }

        //
        let mut event_counter = latent_image.clone();
        let mut timestamps = latent_image.clone();
        timestamps.add_scalar_mut(timestamp_start as f64);

        let (mut y, mut x);
        // Events occurring AFTER this timestamp
        for event in &self.event_during_queue[start_index..] {
            y = event.y() as usize;
            x = event.x() as usize;
            latent_image[(y, x)] +=
                (c * event_counter[(y, x)]).exp() * (event.t() as f64 - timestamps[(y, x)]);

            event_counter[(y, x)] += event_polarity_float(event);

            if self.optimize_c {
                edge_image[(y, x)] += event_polarity_float(event)
                    // * c
                    * (-(event.t() as f64 - timestamps[(y, x)])/1000000.0).exp();
                // We assume a timescale of microseconds as in the original paper;
                // i.e., 1e6 microseconds per second
            }
            timestamps[(y, x)] = event.t() as f64;
        }

        event_counter.mul_assign(c);
        event_counter = event_counter.map(|x: f64| x.exp());

        timestamps.mul_assign(-1.0);
        timestamps.add_scalar_mut(self.event_during_queue.last().unwrap().t() as f64);
        event_counter.component_mul_assign(&timestamps);
        latent_image.add_assign(&event_counter);

        // Events occurring BEFORE this timestamp

        timestamps = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);
        timestamps.add_scalar_mut(timestamp_start as f64);
        event_counter = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        for event in &self.event_during_queue[..start_index] {
            y = event.y() as usize;
            x = event.x() as usize;
            latent_image[(y, x)] +=
                (c * event_counter[(y, x)]).exp() * (timestamps[(y, x)] - event.t() as f64);

            event_counter[(y, x)] -= event_polarity_float(event);

            if self.optimize_c {
                edge_image[(y, x)] -= event_polarity_float(event)
                    // * c
                    * (-(timestamps[(y, x)] - event.t() as f64)/1000000.0).exp();
            }

            timestamps[(y, x)] = event.t() as f64;
        }

        event_counter.mul_assign(c);
        event_counter = event_counter.map(|x: f64| x.exp());

        timestamps.add_scalar_mut(-self.event_during_queue[0].t() as f64);
        event_counter.component_mul_assign(&timestamps);
        latent_image.add_assign(&event_counter);

        latent_image.div_assign(
            self.event_during_queue.last().unwrap().t() as f64
                - self.event_during_queue[0].t() as f64,
        );
        let blurred_image = &self.blur_info.as_ref().unwrap().blurred_image;
        latent_image = blurred_image.component_div(&latent_image);

        // The last gathered latent image might get completely black pixels if there are some
        // negative polarity events right near the end of the exposure time. This looks unreasonably
        // bad, so I'm fixing it manually here. It's likely due to some DVS pixels firing slightly
        // sooner than others for the same kind of intensity change.
        for (latent_px, blurred_px) in latent_image.iter_mut().zip(blurred_image.iter()) {
            if *latent_px > 1.1 {
                *latent_px = 1.1;
            } else if *latent_px <= 0.0 {
                if *blurred_px == 1.0 {
                    *latent_px = 1.0;
                } else {
                    *latent_px = 0.0;
                }
            }
        }

        // show_display_force("latent", &latent_image, 1, false);
        (
            Mat::try_from_cv(latent_image).unwrap(),
            Mat::try_from_cv(edge_image).unwrap(),
        )
    }
}

pub fn deblur_image(event_adder: &mut EventAdder) -> Option<DeblurReturn> {
    if let Some(blur_info) = &event_adder.blur_info {
        event_adder.interval_count += 1;
        // The beginning time for interval 0. Probably before the blurred image exposure beginning time
        // TODO: Why? Events outside the exposure time aren't included then...
        // let interval_beginning_start =
        //     ((blur_info.exposure_begin_t) / event_adder.interval_t) * event_adder.interval_t;
        let interval_beginning_start = blur_info.exposure_begin_t;
        let interval_end_start =
            // ((blur_info.exposure_end_t) / event_adder.interval_t) * event_adder.interval_t;
            blur_info.exposure_end_t;
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
            let mut current_ts =
                intermediate_interval_start_timestamps[0].0 + event_adder.interval_t;
            loop {
                if current_ts < interval_beginning_start && !event_adder.deblur_only {
                    intermediate_interval_start_timestamps.push((current_ts, Mat::default()));
                    current_ts += event_adder.interval_t;
                } else {
                    break;
                }
            }

            if !event_adder.deblur_only && !event_adder.event_before_queue.is_empty() {
                intermediate_interval_start_timestamps
                    .par_iter_mut()
                    .for_each(|(timestamp_start, mat)| {
                        // let c = optimize_c()
                        *mat = event_adder
                            .get_intermediate_image(event_adder.current_c, *timestamp_start);
                    });

                for elem in intermediate_interval_start_timestamps {
                    ret_vec.push(elem.1)
                }
            }
        }

        ////////////////////////

        // Naturally handle the case where the input image is relatively sharp
        if interval_beginning_start > blur_info.exposure_end_t {
            println!("Bad interval");
            return None;
        }

        // Make a vec of these timestamps so we can iterate them concurrently
        let mut interval_start_timestamps = vec![(interval_beginning_start, Mat::default(), 0.0)];
        let mut current_ts = interval_beginning_start + event_adder.interval_t;
        loop {
            if current_ts <= interval_end_start && !event_adder.deblur_only {
                interval_start_timestamps.push((current_ts, Mat::default(), event_adder.current_c));
                current_ts += event_adder.interval_t;
            } else {
                break;
            }
        }

        // Optimize c just once, relative to the temporal middle of the APS frame
        let new_c = match event_adder.optimize_c
            && event_adder.interval_count % event_adder.optimize_c_frequency == 0
        {
            true => {
                event_adder.interval_count = 0;
                event_adder
                    .optimize_c(interval_start_timestamps[interval_start_timestamps.len() / 2].0)
            }
            false => event_adder.current_c,
        };

        interval_start_timestamps
            .par_iter_mut()
            .for_each(|(timestamp_start, mat, found_c)| {
                // let c = match event_adder.optimize_c {
                //     true => {event_adder.optimize_c(*timestamp_start)},
                //     false => {event_adder.current_c}
                // };
                *found_c = new_c;
                *mat = event_adder
                    .get_latent_and_edge(*found_c, *timestamp_start)
                    .0
            });

        let mut last_interval = interval_start_timestamps.last().unwrap().clone();
        if event_adder.deblur_only {
            assert_eq!(interval_start_timestamps.len(), 1);
            last_interval.0 += event_adder.interval_t;
        }

        for elem in interval_start_timestamps {
            ret_vec.push(elem.1)
        }

        Some(DeblurReturn {
            last_interval_start_timestamp: last_interval.0,
            ret_vec,
            found_c: last_interval.2,
        })
    } else {
        None
    }
}

fn event_polarity_float(event: &Event) -> f64 {
    match event.on() {
        true => 1.0,
        false => -1.0,
    }
}

use opencv::imgproc::{sobel, threshold, THRESH_BINARY};

pub struct BlurInfo {
    pub blurred_image: OMatrix<f64, Dynamic, Dynamic>,
    pub exposure_begin_t: i64,
    pub exposure_end_t: i64,
    pub init: bool, // TODO: not very rusty
    pub packet_timestamp: Instant,
}

impl BlurInfo {
    pub fn new(
        image: OMatrix<f64, Dynamic, Dynamic>,
        exposure_begin_t: i64,
        exposure_end_t: i64,
        packet_timestamp: Instant,
    ) -> BlurInfo {
        BlurInfo {
            blurred_image: image,
            exposure_begin_t,
            exposure_end_t,
            init: true,
            packet_timestamp,
        }
    }
}
