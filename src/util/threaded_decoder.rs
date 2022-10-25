use std::time::{Duration, Instant};
use aedat::base::{Decoder, Packet, StreamContent};
use num_traits::FromPrimitive;
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};
use tokio::time::sleep;

pub(crate) struct TimestampedPacket {
    pub timestamp: Instant,
    pub packet: Packet,
}

pub(crate) struct PacketReceiver {
    bounded_receiver: Option<Receiver<TimestampedPacket>>,
    unbounded_receiver: Option<UnboundedReceiver<TimestampedPacket>>,
}

struct PacketTimingSim {
    last_packet_embedded_timestamp: u64,
    last_packet_decoded_timestamp: Instant,
}

impl PacketReceiver {
    pub(crate) async fn next(&mut self) -> Option<TimestampedPacket> {
        if self.bounded_receiver.is_some() {
            return self.bounded_receiver.as_mut().unwrap().recv().await;
        }
        if self.unbounded_receiver.is_some() {
            return self.unbounded_receiver.as_mut().unwrap().recv().await;
        }
        None
    }
}

pub(crate) fn setup_packet_threads(
    aedat_decoder_0: Decoder,
    aedat_decoder_1: Option<Decoder>,
    simulate_latency: bool
) -> PacketReceiver {
    let mut packet_receiver = PacketReceiver {
        bounded_receiver: None,
        unbounded_receiver: None,
    };
    match aedat_decoder_1 {
        None => {
            let (sender, receiver): (
                tokio::sync::mpsc::Sender<TimestampedPacket>,
                tokio::sync::mpsc::Receiver<TimestampedPacket>,
            ) = tokio::sync::mpsc::channel(500);
            setup_file_threads(sender, aedat_decoder_0, simulate_latency);
            packet_receiver.bounded_receiver = Some(receiver);
        }
        Some(decoder_1) => {
            let (sender, receiver): (
                tokio::sync::mpsc::UnboundedSender<TimestampedPacket>,
                tokio::sync::mpsc::UnboundedReceiver<TimestampedPacket>,
            ) = tokio::sync::mpsc::unbounded_channel();
            setup_socket_threads(sender, aedat_decoder_0, decoder_1);
            packet_receiver.unbounded_receiver = Some(receiver);
        }
    };
    packet_receiver
}

/// Use a bounded channel for a file source, so that we don't just read in the whole file at once
fn setup_file_threads(
    sender: tokio::sync::mpsc::Sender<TimestampedPacket>,
    mut decoder_0: Decoder,
                      simulate_latency: bool) {
    tokio::spawn(async move {
        let mut timing_sim: Option<PacketTimingSim> = None;
        let mut packet_end_time: u64 = 0;
        loop {
            match decoder_0.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break;
                }
                Some(Ok(p)) => {
                    if simulate_latency {
                        latency_sim_update(&mut timing_sim, &mut packet_end_time, &p).await;
                    }

                    if (sender.send(TimestampedPacket { timestamp: Instant::now(), packet: p}).await).is_err() {
                        println!("receiver dropped");
                        return;
                    }
                }
                Some(Err(e)) => panic!("{}", e),
            }
        }
    });
}

fn setup_socket_threads(
    sender_main: tokio::sync::mpsc::UnboundedSender<TimestampedPacket>,
    mut decoder_0: Decoder,
    mut decoder_1: Decoder,
) {
    let sender_0 = sender_main;
    let sender_1 = sender_0.clone();
    // Create thread for decoder_0
    tokio::spawn(async move {


        loop {
            match decoder_0.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break;
                }
                Some(Ok(mut p)) => {
                    p.stream_id = decoder_0.id_to_stream.get(&p.stream_id).unwrap().content as u32;
                    if sender_0.send(TimestampedPacket { timestamp: Instant::now(), packet: p}).is_err() {
                        println!("receiver dropped");
                        return;
                    }
                }
                Some(Err(e)) => panic!("{}", e),
            }
        }
    });

    // let (sender_1, mut receiver_1): (tokio::sync::mpsc::UnboundedSender<Packet>, tokio::sync::mpsc::UnboundedReceiver<Packet>)
    //     = tokio::sync::mpsc::unbounded_channel();
    // Create thread for decoder_1
    tokio::spawn(async move {
        loop {
            match decoder_1.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break;
                }
                Some(Ok(mut p)) => {
                    p.stream_id = decoder_1.id_to_stream.get(&p.stream_id).unwrap().content as u32;
                    if sender_1.send(TimestampedPacket { timestamp: Instant::now(), packet: p}).is_err() {
                        println!("receiver dropped");
                        return;
                    }
                }
                Some(Err(e)) => panic!("{}", e),
            }
        }
    });

    // // create listener thread to collate packet messages
    // tokio::spawn(async move {
    //     loop {
    //         match receiver_0.recv().await {
    //             Some(p) => {
    //                 if let Err(_) = sender_main.send(p) {
    //                     println!("receiver dropped");
    //                     return;
    //                 }
    //             }
    //             _ => {}
    //         }
    //         match receiver_1.recv().await {
    //             Some(p) => {
    //                 if let Err(_) = sender_main.send(p) {
    //                     println!("receiver dropped");
    //                     return;
    //                 }
    //             }
    //             _ => {}
    //         }
    //     }
    // });
}


async fn latency_sim_update(timing_sim: &mut Option<PacketTimingSim>, packet_end_time: &mut u64, p: &Packet) {
    *packet_end_time = match FromPrimitive::from_u32(p.stream_id) {
        Some(StreamContent::Frame) => {
            let frame = match aedat::frame_generated::size_prefixed_root_as_frame(&p.buffer)
            {
                Ok(result) => result,
                Err(_) => {
                    panic!("the packet does not have a size prefix");
                }
            };
            frame.exposure_end_t() as u64
        }
        Some(StreamContent::Events) => {
            let event_packet =
                match aedat::events_generated::size_prefixed_root_as_event_packet(&p.buffer) {
                    Ok(result) => result,
                    Err(_) => {
                        panic!("the packet does not have a size prefix");
                    }
                };
            match event_packet.elements() {
                None => {
                    *packet_end_time
                }
                Some(elems) => {
                    elems.last().unwrap().t() as u64
                }
            }
        }
        _ => {
            *packet_end_time
        }
    };

    match timing_sim {
        None => {
            if *packet_end_time > 0 {
                *timing_sim = Some(PacketTimingSim { last_packet_embedded_timestamp: *packet_end_time, last_packet_decoded_timestamp: Instant::now() })
            }
        }
        Some(s) => {
            let time_to_sleep = ((packet_end_time.saturating_sub(s.last_packet_embedded_timestamp)) as u64).saturating_sub((Instant::now() - s.last_packet_decoded_timestamp).as_micros() as u64);
            assert!(time_to_sleep < 10e7 as u64);  // Sanity check
            // println!("Sleeping for {} us", time_to_sleep);
            sleep(Duration::from_micros(time_to_sleep)).await;
            s.last_packet_embedded_timestamp = *packet_end_time;
            s.last_packet_decoded_timestamp = Instant::now();
        }
    }
}
