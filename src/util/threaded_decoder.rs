use std::time::Instant;
use aedat::base::{Decoder, Packet};
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};

pub(crate) struct TimestampedPacket {
    pub timestamp: Instant,
    pub packet: Packet,
}

pub(crate) struct PacketReceiver {
    bounded_receiver: Option<Receiver<TimestampedPacket>>,
    unbounded_receiver: Option<UnboundedReceiver<TimestampedPacket>>,
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
            ) = tokio::sync::mpsc::channel(2000);
            setup_file_threads(sender, aedat_decoder_0);
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
fn setup_file_threads(sender: tokio::sync::mpsc::Sender<TimestampedPacket>, mut decoder_0: Decoder) {
    tokio::spawn(async move {
        loop {
            match decoder_0.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break;
                }
                Some(Ok(p)) => {
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
