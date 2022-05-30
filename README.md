# davis-EDI-rs
A fast, Rust-based, open-source implementation of the paper "Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera" (2019) by Pan et al.

## About
This project aims to elucidate the paper above and move towards real-time software systems which take advantage of synthesized event and frame data. The original paper's [code](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) is largely obfuscated, making it impossible to make improvements. Furthermore, the original code base is written in MATLAB, making it relatively slow, and it uses a converted MATLAB data format for the DAVIS files. 

This implementation operates on .aedat4 files generated directly by an [iniVation](https://inivation.com/) DAVIS camera. This removes a lot of the headache with getting started, and allows you to easily run the program on extremely large files (that is, long recordings), and down the line should be able to easily process a live camera feed in real time as well. If we're going to move towards practical systems with event cameras, we need practical processing! I've included a sample file recorded with my camera.

I did my best to understand and implement the mathematics presented in the original paper, while taking a few liberties for the sake of speed gains. At present, this project only implements the **_Event-based Double Integral_** from the [2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Bringing_a_Blurry_Frame_Alive_at_High_Frame-Rate_With_an_CVPR_2019_paper.pdf), **not** the *multiple Event-based Double Integral* from the [2020 paper](https://ieeexplore.ieee.org/abstract/document/9252186). I plan to implement that method when some important remaining work is completed (see below).

## To-do list
There are some major things left before I can start implementing mEDI. Any assistance from the community would be greatly appreciated

- [ ] Add documentation
- [ ] Improve performance
- [ ] Make it more Rusty (follow conventions)
- [ ] **Fix the contrast threshold optimization**
  - I could really use help with this. My energy function isn't producing great or consistent results. I did my best to replicate the MATLAB functions for getting gradient magnitude and the edge cross correlation, but it's not quite looking like it does in the original paper. ~~For now, the program just uses a hard-coded contrast threshold value.~~ There's a command-line option to enable the contrast threshold optimization, but it works best when input APS images are blurry.

## Requirements
- Rust 2021 or higher
- Cargo
- OpenCV and its Rust bindings (installation instructions [here](https://github.com/twistedfall/opencv-rust))
- Other dependencies will download and install automatically when building with Cargo

![](https://github.com/ac-freeman/davis-EDI-rs/blob/main/demo.gif)
