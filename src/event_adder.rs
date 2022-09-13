use std::mem;
use std::ops::{AddAssign, DivAssign, MulAssign, Sub, SubAssign};
use nalgebra::{DMatrix, Dynamic, OMatrix};
use aedat::base::Packet;
use aedat::events_generated::Event;
use opencv::core::{ElemMul, Mat, MatExprTraitConst, CV_64F, BORDER_DEFAULT, no_array, normalize, NORM_MINMAX, sum_elems, sqrt, mean};
use cv_convert::{TryFromCv};


use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

const FIB: [f64; 22] = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0, 377.0, 610.0, 987.0, 1597.0, 2584.0,
    4181.0, 6765.0, 10946.0, 17711.0];

pub struct DeblurReturn {
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) ret_vec: Vec<Mat>,
    pub(crate) found_c: f64,
}

pub struct EventAdder {
    /// The time span of each reconstructed frame
    interval_t: i64,

    /// Events occurring before the current blurred image
    event_before_queues: [Vec<Event>; 3],

    /// Events occurring during the current blurred image
    event_during_queues: [Vec<Event>; 3],

    /// Events occurring after the current blurred image
    event_after_queue: Vec<Event>,
    height: i32,
    width: i32,
    pub(crate) last_interval_start_timestamp: i64,
    pub(crate) latent_image: Mat,
    pub(crate) blur_infos: [Option<BlurInfo>; 3],
    pub(crate) next_blur_info: Option<BlurInfo>,
    pub(crate) current_c: f64,
    optimize_c: bool,
    pub(crate) mode: Mode
}

pub(crate) enum Mode {
    Edi,
    MEdi,
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
        m_edi: bool,
    ) -> EventAdder {
        let mode = match m_edi {
            false => {Edi},
            true => {MEdi}
        };
        EventAdder {
            interval_t: output_frame_length,
            event_before_queues: [Vec::new(), Vec::new(), Vec::new()],
                event_during_queues: [Vec::new(), Vec::new(), Vec::new()],
            event_after_queue: Vec::new(),
            height: height as i32,
            width: width as i32,
            last_interval_start_timestamp: 0,
            latent_image: Mat::zeros(height as i32, width as i32, CV_64F)
                .unwrap()
                .to_mat()
                .unwrap(),
            blur_infos: [None, None, None],
            next_blur_info: Default::default(),
            current_c: start_c,
            optimize_c,
            mode
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
            self.sort_event(event, 0);
        }
    }

    fn sort_event(&mut self, event: &Event, blur_period: usize) {
        let blur_info = match &self.blur_infos[blur_period] {
            None => { panic!("blur_info not initialized")}
            Some(a) => {a}
        };
        match event.t() {
            a if a < blur_info.exposure_begin_t => {
                self.event_before_queues[blur_period].push(*event);
            }
            a if a > blur_info.exposure_end_t => {
                match self.mode {
                    Edi => {
                        self.event_after_queue.push(*event);
                    }
                    MEdi => {
                        if blur_period < 2 {
                            self.sort_event(event, blur_period + 1);
                        }
                        else {
                            self.event_after_queue.push(*event);
                        }
                    }
                }

            }
            _ => {
                self.event_during_queues[blur_period].push(*event);
            }
        }
    }

    pub fn reset_event_queues(&mut self) {
        match self.mode {
            Edi => {
                mem::swap(&mut self.event_before_queues[0], &mut self.event_after_queue);
                self.event_after_queue.clear();
                self.event_during_queues[0].clear();
            }
            MEdi => {
                self.event_before_queues.swap(0,1);
                self.event_before_queues.swap(1,2);
                mem::swap(&mut self.event_before_queues[2], &mut self.event_after_queue);
                self.event_after_queue.clear();

                self.event_during_queues.swap(0,1);
                self.event_during_queues.swap(1,2);
                self.event_during_queues[2].clear();
            }
        }

    }

    fn get_intermediate_image(&self, c: f64, timestamp_start: i64) -> Mat {
        if self.event_before_queues[0].is_empty() {
            panic!("Empty before queue");
        }

        // TODO: Need to avoid having to traverse the whole queue each time?
        let start_index = 0;
        let mut end_index = 0;
        loop {
            if end_index + 1 == self.event_before_queues[0].len()
                || self.event_before_queues[0][end_index + 1].t() > timestamp_start + self.interval_t {
                break;
            }
            end_index += 1;
        }

        let mut event_counter = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        let (mut y, mut x);
        for event in &self.event_before_queues[0][start_index..end_index] {
            y = event.y() as usize;
            x = event.x() as usize;
            event_counter[(y, x)] += event_polarity_float(event);
        }

        // L^tilde(t) = L^tilde(f) + cE(t)
        // Take the exp of L^tilde(t) to get L(t), the final latent image
        event_counter.mul_assign(c);
        event_counter = event_counter.map(|x: f64| x.exp());
        let event_counter_mat = Mat::try_from_cv(event_counter).unwrap();


        

        self
            .latent_image
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

        let mut x1 = a + FIB[fib_index - 2] / FIB[fib_index] * (b-a);
        let mut x2 = b - FIB[fib_index - 2] / FIB[fib_index] * (b-a);
        let mut fx1 = self.get_phi(x1, timestamp_start);
        let mut fx2 = self.get_phi(x2, timestamp_start);

        for k in 1..fib_index-2 {
            if fx1 < fx2 {
                b = x2;
                x2 = x1;
                fx2 = fx1;
                x1 = a + FIB[fib_index - k - 1] / FIB[fib_index - k + 1] * (b-a);
                fx1 = self.get_phi(x1, timestamp_start);
            } else {
                a = x1;
                x1 = x2;
                fx1 = fx2;
                x2 = b - FIB[fib_index - k - 1] / FIB[fib_index - k + 1] * (b-a);
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
        let (latent_image, mt_image, _) = self.get_latent_and_edge(c, timestamp_start, 0);
        // _show_display_force("mt_image", &mt_image, 1, true);

        let (latent_grad, latent_edges) = self.get_gradient_and_edges(latent_image);
        // _show_display_force("grad", &latent_grad, 1, false);
        // _show_display_force("grad_edges", &latent_edges, 1, false);
        let (_mt_grad, mt_edges) = self.get_gradient_and_edges(mt_image);

        let phi_edge = sum_elems(
            &latent_edges.elem_mul(mt_edges)
                .into_result().unwrap().to_mat().unwrap()).unwrap().0[0];
        // dbg!(phi_edge);

        let phi_tv = sum_elems(&latent_grad).unwrap().0[0];
        // dbg!(phi_tv);

        
        // dbg!(phi);
        0.14 * phi_tv - phi_edge
    }

    fn get_gradient_and_edges(&self, image: Mat) -> (Mat, Mat) {
        let mut image_sobel_x = Mat::default();
        sobel(&image, &mut image_sobel_x, CV_64F, 1, 0, 3,
              1.0, 0.0, BORDER_DEFAULT).expect("Sobel error");

        let mut image_sobel_y = Mat::default();
        sobel(&image, &mut image_sobel_y, CV_64F, 0, 1, 3,
              1.0, 0.0, BORDER_DEFAULT).expect("Sobel error");
        let tmp = (image_sobel_x.clone().elem_mul(&image_sobel_x) + image_sobel_y.clone().elem_mul(&image_sobel_y))
            .into_result().unwrap().to_mat().unwrap();

        let mut grad = Mat::default();
        sqrt(&tmp, &mut grad).unwrap();

        let mut grad_norm = Mat::default();
        normalize(&grad, &mut grad_norm, 0.0, 1.0,
                  NORM_MINMAX, -1, &no_array()).expect("Norm error");

        let mut thresholded = Mat::default();
        let mut threshold_val = mean(&grad_norm, &no_array()).unwrap().0[0];
        threshold_val += (1.0 - threshold_val) / 3.0;
        threshold(&grad_norm, &mut thresholded, threshold_val, 1.0, THRESH_BINARY).unwrap();

        (grad, thresholded)
    }

    ////////////////////////////////
    // mEDI sketch
    // -- Deblur the image (for a certain timestamp, t) to get a latent intensity image L_i
    //      -- the deblurring is different than before. Calculated from the next two latent images
    // -- Multiply L_i by (1/T int(exp(cE(t))dt) to get the reblurred image
    // -- Calculate the square of the L2 norm. Get the c that minimizes this value









    fn get_latent_and_edge(&self, c: f64, timestamp_start: i64, blur_index: usize) -> (Mat, Mat, OMatrix<f64, Dynamic, Dynamic>) {
        assert!(c > 0.0 && c <= 1.0);
        if self.event_during_queues[blur_index].is_empty() {
            panic!("No during queue")
        }
        // TODO: Need to avoid having to traverse the whole queue each time?
        let mut start_index = 0;
        loop {
            if start_index + 1 == self.event_during_queues[blur_index].len()
                || self.event_during_queues[blur_index][start_index + 1].t() > timestamp_start {
                break;
            }
            start_index += 1;
        }

        let mut latent_image = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);
        let mut edge_image = latent_image.clone();
        //
        let mut event_counter = latent_image.clone();
        let mut timestamps = latent_image.clone();
        timestamps.add_scalar_mut(timestamp_start as f64);

        let (mut y, mut x);
        // Events occurring AFTER this timestamp
        for event in &self.event_during_queues[blur_index][start_index..] {
            y = event.y() as usize;
            x = event.x() as usize;
            latent_image[(y, x)] +=
                (c * event_counter[(y, x)]).exp()
                    * (event.t() as f64 - timestamps[(y, x)]);

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
        timestamps.add_scalar_mut(self.event_during_queues[blur_index].last().unwrap().t() as f64);
        event_counter.component_mul_assign(&timestamps);
        latent_image.add_assign(&event_counter);


        // Events occurring BEFORE this timestamp

        timestamps = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);
        timestamps.add_scalar_mut(timestamp_start as f64);
        event_counter = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        for event in &self.event_during_queues[blur_index][..start_index] {
            y = event.y() as usize;
            x = event.x() as usize;
            latent_image[(y, x)] +=
                (c * event_counter[(y, x)]).exp()
                    * (timestamps[(y, x)] - event.t() as f64);

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

        timestamps.add_scalar_mut(-self.event_during_queues[blur_index][0].t() as f64);
        event_counter.component_mul_assign(&timestamps);
        latent_image.add_assign(&event_counter);

        latent_image.div_assign( self.event_during_queues[blur_index].last().unwrap().t() as f64
            - self.event_during_queues[blur_index][0].t() as f64);
        let blurred_image = &self.blur_infos[blur_index].as_ref().unwrap().blurred_image;
        let a_exp = latent_image.clone_owned();
        latent_image = blurred_image.component_div(&latent_image);

        // The last gathered latent image might get completely black pixels if there are some
        // negative polarity events right near the end of the exposure time. This looks unreasonably
        // bad, so I'm fixing it manually here. It's likely due to some DVS pixels firing slightly
        // sooner than others for the same kind of intensity change.
        for (latent_px, blurred_px) in latent_image.iter_mut().zip(blurred_image.iter()) {
            // if *latent_px > 1.1 {
            //     *latent_px = 1.0;
            // } else if *latent_px <= 0.0 {
            //     if *blurred_px == 1.0 {
            //         *latent_px = 1.0;
            //     } else {
            //         *latent_px = 0.0;
            //     }
            // }
        }

        // show_display_force("latent", &latent_image, 1, false);
        (Mat::try_from_cv(latent_image).unwrap(), Mat::try_from_cv(edge_image).unwrap(), a_exp)

    }

    fn get_bi(&self, c: f64, blur_index: usize, timestamp_start_1: i64, timestamp_start_2: i64) -> OMatrix<f64, Dynamic, Dynamic> {
        let mut bi = DMatrix::<f64>::zeros(self.height as usize, self.width as usize);

        let mut start_index = 0;
        loop {
            if start_index + 1 == self.event_during_queues[blur_index].len()
                || self.event_during_queues[blur_index][start_index + 1].t() > timestamp_start_1 {
                break;
            }
            start_index += 1;
        }

        let mut end_index = 0;
        loop {
            if end_index + 1 == self.event_during_queues[blur_index+1].len()
                || self.event_during_queues[blur_index + 1][end_index + 1].t() > timestamp_start_2 {
                break;
            }
            end_index += 1;
        }


        let (mut y, mut x);

        // Events occurring during the blurred image
        for event in &self.event_during_queues[blur_index][start_index..] {
            y = event.y() as usize;
            x = event.x() as usize;
            bi[(y, x)] += event_polarity_float(event);
        }

        // Events occurring between the blurred image and the next blurred image
        for event in &self.event_before_queues[blur_index + 1][0..] {
            y = event.y() as usize;
            x = event.x() as usize;
            bi[(y, x)] += event_polarity_float(event);
        }

        // Events occurring during the first part of the next blurred image
        for event in &self.event_during_queues[blur_index + 1][0..end_index] {
            y = event.y() as usize;
            x = event.x() as usize;
            bi[(y, x)] += event_polarity_float(event);
        }

        for (px) in bi.iter_mut() {
            *px = *px * c;
        }

        bi
    }

    // Simplify math?
    // orig:
    // L1 = 3L2 - L3 -TB2 - a2 - b2 + b1
    // L1 = L2 - L3 - 2a2 - b2 + b1
    //
    // b_i should be ALL the events occurring between the start of latent image L_i and the start of
    // latent image L_i+1
    pub(crate) fn get_latent_and_edge_medi(&self, c: f64, timestamp_start_l1: i64, timestamp_start_l2: i64, timestamp_start_l3: i64) -> (Mat) {
        let (l_2_mat, b2_mat, mut a2) = self.get_latent_and_edge(c, timestamp_start_l2, 1);

        let mut l_2: OMatrix<f64, Dynamic, Dynamic> = OMatrix::<f64, Dynamic, Dynamic>::try_from_cv(l_2_mat).unwrap();
        let l2_copy = l_2.clone_owned();
        let mut l2_ln = l_2.clone_owned();

        // let mut b2: OMatrix<f64, Dynamic, Dynamic> = OMatrix::<f64, Dynamic, Dynamic>::try_from_cv(b2_mat).unwrap();
        // for (px) in b2.iter_mut() {
        //     *px = *px * c;
        // }
        // let b2 = self.get_bi(c, 1, timestamp_start_l2, timestamp_start_l3);

        let mut B2 = self.blur_infos[1].as_ref().unwrap().blurred_image.clone_owned();
        for (px) in B2.iter_mut() {
            if *px == 0.0 {
                *px = 0.0001;
            }
            *px = (255.0 * *px).ln() / 255.0;
            let tmp = *px;
            print!("");
        }

        for (px) in l2_ln.iter_mut() {
            if *px == 0.0 {
                *px = 0.000001;
            }
            *px = (255.0 * *px).ln() / 255.0;
            let tmp = *px;
            print!("");
        }

        // need to take the log of a2_exp
        for (px) in a2.iter_mut() {
            if *px == 0.0 {
                *px = 0.0001;
            }
            *px = (255.0 * *px).ln() / 255.0;
        }


        let (l_3_mat, _, _) = self.get_latent_and_edge(c, timestamp_start_l3, 2);

        let l_3: OMatrix<f64, Dynamic, Dynamic> = OMatrix::<f64, Dynamic, Dynamic>::try_from_cv(l_3_mat).unwrap();
        let tmp = Mat::try_from_cv(l_3.clone()).unwrap();
        _show_display_force("l_3", &tmp, 1, false);


        // let b1 = self.get_bi(c, 0, timestamp_start_l1, timestamp_start_l2);

        let mut l3_ln = l_3.clone_owned();
        for (px) in l3_ln.iter_mut() {
            if *px == 0.0 {
                *px = 0.000001;
            }
            *px = (255.0 * *px).ln() / 255.0;
            let tmp = *px;
            print!("");
        }

        let b2 = l3_ln -l2_ln.clone_owned();









        let (l1_mat, b1_mat, _) = self.get_latent_and_edge(c, timestamp_start_l1, 0);
        let mut l_1: OMatrix<f64, Dynamic, Dynamic> = OMatrix::<f64, Dynamic, Dynamic>::try_from_cv(l1_mat).unwrap();

        let mut l1_ln = l_1.clone_owned();

        for (px) in l1_ln.iter_mut() {
            if *px == 0.0 {
                *px = 0.000001;
            }
            *px = (255.0 * *px).ln() / 255.0;
            let tmp = *px;
            print!("");
        }

        let b1 = l2_ln -l1_ln;
        // let mut b1: OMatrix<f64, Dynamic, Dynamic> = OMatrix::<f64, Dynamic, Dynamic>::try_from_cv(b1_mat).unwrap();
        // for (px) in b1.iter_mut() {
        //     *px = *px * c;
        // }

        l_2.add_assign(l2_copy.clone_owned());
        l_2.add_assign(l2_copy);
        // l_2.sub_assign(l_3.clone_owned());
        l_2.sub_assign(l_3);
        l_2.sub_assign(B2);

        // l_2.sub_assign(l2_ln);
        // l_2.sub_assign(a2.clone_owned());
        l_2.sub_assign(a2);
        let tmp = Mat::try_from_cv(l_2.clone()).unwrap();
        _show_display_force("l_2_mid", &tmp, 1, true);
        l_2.sub_assign(b2);
        l_2.add_assign(b1);

        for (latent_px, blurred_px) in l_2.iter_mut().zip(self.blur_infos[0].as_ref().unwrap().blurred_image.iter()) {
            if *latent_px > 1.1 {
                *latent_px = 1.0;
            } else if *latent_px <= 0.0 {
                // if *blurred_px == 1.0 {
                //     *latent_px = 1.0;
                // } else {
                    *latent_px = 0.0;
                // }
            }
        }

        let tmp = Mat::try_from_cv(l_2).unwrap();
        _show_display_force("tmp", &tmp, 1, false);
        (tmp)

        // (Mat::try_from_cv(l_2).unwrap())
    }
}

pub fn deblur_image(event_adder: &EventAdder) -> Option<DeblurReturn> {
    if let Some(blur_info) = &event_adder.blur_infos[0] {
        match event_adder.mode {
            Edi => {}
            MEdi => {
                assert!(event_adder.blur_infos[1].is_some());
                assert!(event_adder.blur_infos[2].is_some());
            }
        }



        // The beginning time for interval 0. Probably before the blurred image exposure beginning time
        let interval_beginning_start =
            ((blur_info.exposure_begin_t) / event_adder.interval_t) * event_adder.interval_t;
        let interval_end_start =
            ((blur_info.exposure_end_t) / event_adder.interval_t) * event_adder.interval_t;
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
                    // *mat = event_adder.get_intermediate_image(event_adder.current_c, *timestamp_start);
                });

            for elem in intermediate_interval_start_timestamps {
                // ret_vec.push(elem.1)
            }
        }

        ////////////////////////

        // Naturally handle the case where the input image is relatively sharp
        if interval_beginning_start >= blur_info.exposure_end_t {
            panic!("Bad interval")
        }

        // Make a vec of these timestamps so we can iterate them concurrently
        let mut interval_start_timestamps = vec![(interval_beginning_start, Mat::default(), event_adder.current_c)];
        let mut current_ts = interval_beginning_start + event_adder.interval_t;
        loop {
            if current_ts <= interval_end_start {
                interval_start_timestamps.push((current_ts, Mat::default(), event_adder.current_c));
                current_ts += event_adder.interval_t;
            } else {
                break;
            }
        }


        // TODO: better structure


        match event_adder.mode {
            Edi => {
                interval_start_timestamps
                .par_iter_mut()
                .for_each(|(timestamp_start, mat, found_c)| {
                    let c = match event_adder.optimize_c {
                        true => { event_adder.optimize_c(*timestamp_start) },
                        false => { event_adder.current_c }
                    };
                    *found_c = c;
                    *mat = event_adder.get_latent_and_edge(c, *timestamp_start, 0).0
                });
            }
            MEdi => {
                let interval_1_beginning_start =
                    ((blur_info.exposure_begin_t) / event_adder.interval_t) * event_adder.interval_t;
                let interval_1_end_start =
                    ((blur_info.exposure_end_t) / event_adder.interval_t) * event_adder.interval_t;
                if interval_1_beginning_start >= event_adder.blur_infos[1].as_ref().unwrap().exposure_end_t {
                    panic!("Bad interval")
                }
                let mut interval_start_timestamps_1 = vec![(interval_1_beginning_start, Mat::default(), 0.0)];
                let mut current_ts = interval_1_beginning_start + event_adder.interval_t;
                loop {
                    if current_ts <= interval_1_end_start {
                        interval_start_timestamps_1.push((current_ts, Mat::default(), event_adder.current_c));
                        current_ts += event_adder.interval_t;
                    } else {
                        break;
                    }
                }

                let interval_2_beginning_start =
                    ((blur_info.exposure_begin_t) / event_adder.interval_t) * event_adder.interval_t;
                let interval_2_end_start =
                    ((blur_info.exposure_end_t) / event_adder.interval_t) * event_adder.interval_t;
                if interval_2_beginning_start >= event_adder.blur_infos[2].as_ref().unwrap().exposure_end_t {
                    panic!("Bad interval")
                }
                let mut interval_start_timestamps_2 = vec![(interval_2_beginning_start, Mat::default(), 0.0)];
                let mut current_ts = interval_2_beginning_start + event_adder.interval_t;
                loop {
                    if current_ts <= interval_2_end_start {
                        interval_start_timestamps_2.push((current_ts, Mat::default(), event_adder.current_c));
                        current_ts += event_adder.interval_t;
                    } else {
                        break;
                    }
                }

                let found_c = &interval_start_timestamps[0].2;
                let idx = interval_start_timestamps.len()/2;
                interval_start_timestamps[idx].1 =
                            event_adder.get_latent_and_edge_medi(
                                *found_c,
                                interval_start_timestamps[idx].0,
                                interval_start_timestamps_1[interval_start_timestamps_1.len()/2].0,
                                interval_start_timestamps_2[interval_start_timestamps_2.len()/2].0,
                            )
                // interval_start_timestamps[0].1 =
                //     event_adder.get_latent_and_edge_medi(
                //         *found_c,
                //         interval_start_timestamps[0].0,
                //         interval_start_timestamps_1[0].0,
                //         interval_start_timestamps_2[0].0,
                //     )

                // for i in 0..interval_start_timestamps.len() {
                //     let found_c = &interval_start_timestamps[i].2;
                //     if i == interval_start_timestamps.len() - 1 {
                //         interval_start_timestamps[i].1 =
                //             event_adder.get_latent_and_edge_medi(
                //                 *found_c,
                //                 interval_start_timestamps[i].0,
                //                 interval_start_timestamps[i].0,
                //                 interval_start_timestamps[i].0,
                //             )
                //     }
                //     else if i == interval_start_timestamps.len() - 2 {
                //         interval_start_timestamps[i].1 =
                //             event_adder.get_latent_and_edge_medi(
                //                 *found_c,
                //                 interval_start_timestamps[i].0,
                //                 interval_start_timestamps[i+1].0,
                //                 interval_start_timestamps[i+1].0,
                //             )
                //     }
                //     else {
                //         interval_start_timestamps[i].1 =
                //             event_adder.get_latent_and_edge_medi(
                //                 *found_c,
                //                 interval_start_timestamps[i].0,
                //                 interval_start_timestamps[i+1].0,
                //                 interval_start_timestamps[i+2].0,
                //             )
                //     }
                // }
            }
        }

        // let mut ret_vec = Vec::with_capacity(interval_start_timestamps.len());
        // self.last_interval_start_timestamp = interval_start_timestamps.last().unwrap().0;
        let last_interval = interval_start_timestamps.last().unwrap().clone();
        for elem in interval_start_timestamps {
            ret_vec.push(elem.1)
        }

        // self.latent_image = ret_vec.last().unwrap().clone();

        Some(DeblurReturn {
            last_interval_start_timestamp: last_interval.0,
            ret_vec,
            found_c: last_interval.2
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


use opencv::imgproc::{sobel, THRESH_BINARY, threshold};
use crate::event_adder::Mode::{Edi, MEdi};
use crate::reconstructor::_show_display_force;


pub struct BlurInfo {
    pub blurred_image: OMatrix<f64, Dynamic, Dynamic>,
    exposure_begin_t: i64,
    exposure_end_t: i64,
    pub init: bool, // TODO: not very rusty
}

impl BlurInfo {
    pub fn new(
        image: OMatrix<f64, Dynamic, Dynamic>,
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
