use std::future::Future;
use std::thread;
use aedat::base::{Decoder, Packet, ParseError};
use crossbeam::channel::bounded;
use tokio::sync::mpsc::{UnboundedReceiver, Receiver};

struct PacketReceiver {
    bounded_receiver: Option<Receiver<Packet>>,
    unbounded_receiver: Option<UnboundedReceiver<Packet>>,
}

impl PacketReceiver {
    pub(crate) async fn next(&mut self) -> Option<Packet> {
        if self.bounded_receiver.is_some() {
            return self.bounded_receiver.as_mut().unwrap().recv().await
        }
        if self.unbounded_receiver.is_some() {
            return self.unbounded_receiver.as_mut().unwrap().recv().await
        }
        None
    }
}

fn setup_threads(aedat_decoder_0: Decoder,
                        aedat_decoder_1: Option<Decoder>) -> PacketReceiver {

    let mut packet_receiver = PacketReceiver { bounded_receiver: None, unbounded_receiver: None };
    match aedat_decoder_1 {
        None => {
            let (sender, receiver): (tokio::sync::mpsc::Sender<Packet>, tokio::sync::mpsc::Receiver<Packet>)
                = tokio::sync::mpsc::channel(2000);
            setup_file_threads(sender, aedat_decoder_0);
            packet_receiver.bounded_receiver = Some(receiver);
        }
        Some(decoder_1) => {
            let (sender, receiver): (tokio::sync::mpsc::UnboundedSender<Packet>, tokio::sync::mpsc::UnboundedReceiver<Packet>)
                = tokio::sync::mpsc::unbounded_channel();
            setup_socket_threads(sender, aedat_decoder_0, decoder_1);
            packet_receiver.unbounded_receiver = Some(receiver);
        }
    };
    packet_receiver
}

/// Use a bounded channel for a file source, so that we don't just read in the whole file at once
fn setup_file_threads(sender: tokio::sync::mpsc::Sender<Packet>, mut decoder_0: Decoder) {
    tokio::spawn(async move {
        loop {
            match decoder_0.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break
                },
                Some(Ok(p)) => {
                    if let Err(_) = sender.send(p).await {
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
    sender_main: tokio::sync::mpsc::UnboundedSender<Packet>,
    mut decoder_0: Decoder,
    mut decoder_1: Decoder) {

    // TODO: There's probably a better way of doing this... Use non-blocking socket/tcp connections?


    let (sender_0, mut receiver_0): (tokio::sync::mpsc::UnboundedSender<Packet>, tokio::sync::mpsc::UnboundedReceiver<Packet>)
        = tokio::sync::mpsc::unbounded_channel();
    // Create thread for decoder_0
    tokio::spawn(async move {
        loop {
            match decoder_0.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break
                },
                Some(Ok(p)) => {
                    if let Err(_) = sender_0.send(p) {
                        println!("receiver dropped");
                        return;
                    }
                }
                Some(Err(e)) => panic!("{}", e),
            }
        }
    });

    let (sender_1, mut receiver_1): (tokio::sync::mpsc::UnboundedSender<Packet>, tokio::sync::mpsc::UnboundedReceiver<Packet>)
        = tokio::sync::mpsc::unbounded_channel();
    // Create thread for decoder_1
    tokio::spawn(async move {
        loop {
            match decoder_1.next() {
                None => {
                    eprintln!("End of file. Leaving reader thread");
                    break
                },
                Some(Ok(p)) => {
                    if let Err(_) = sender_1.send(p) {
                        println!("receiver dropped");
                        return;
                    }
                }
                Some(Err(e)) => panic!("{}", e),
            }
        }
    });

    // create listener thread to collate packet messages
    tokio::spawn(async move {
        loop {
            match receiver_0.recv().await {
                Some(p) => {
                    if let Err(_) = sender_main.send(p) {
                        println!("receiver dropped");
                        return;
                    }
                }
                _ => {}
            }
            match receiver_1.recv().await {
                Some(p) => {
                    if let Err(_) = sender_main.send(p) {
                        println!("receiver dropped");
                        return;
                    }
                }
                _ => {}
            }
        }
    });
}
