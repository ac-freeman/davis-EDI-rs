# davis-EDI-rs
A fast, Rust-based, open-source implementation of the paper "Bringing a Blurry Frame Alive at High Frame-Rate with an Event Camera" (2019) by Pan et al.

![](https://github.com/ac-freeman/davis-EDI-rs/blob/main/output.gif) ![](https://github.com/ac-freeman/davis-EDI-rs/blob/main/output_recon.gif)

## About
This project aims to elucidate the paper above and move towards real-time software systems which take advantage of synthesized event and frame data. The original paper's [code](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera) is largely obfuscated, making it impossible to make improvements. Furthermore, the original code base is written in MATLAB, making it relatively slow, and it uses a converted MATLAB data format for the DAVIS files. 

This implementation operates on .aedat4 files generated directly by an [iniVation](https://inivation.com/) DAVIS camera _and_ a live feed from a DAVIS camera. If we're going to move towards practical systems with event cameras, we need practical processing! I've included a sample file recorded with my camera.

I did my best to understand and implement the mathematics presented in the original paper, while taking a few liberties for the sake of speed gains. At present, this project only implements the **_Event-based Double Integral_** from the [2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pan_Bringing_a_Blurry_Frame_Alive_at_High_Frame-Rate_With_an_CVPR_2019_paper.pdf), **not** the *multiple Event-based Double Integral* from the [2020 paper](https://ieeexplore.ieee.org/abstract/document/9252186).

## Usage
Since there are a lot of command-line arguments available, my preferred method of serving them is in a file. Refer to [`Args.toml`](Args.toml) for an example. You can give it a run by cloning this repository and executing:

`cargo run --release -- --args-filename "./Args.toml"`

### Deblur from an .aedat4 file
You can deblur a pre-existing file by providing the directory of the file in `--base-path`, the file name in `--event-filename-0`, and `--mode` to "file".

### Deblur from a live camera feed
You can also deblur the data coming straight from a camera, in real time! I've provided the [config file](dataset/dv_sockets.xml) for iniVation's DV software which lets you publish the APS frames and event packets to two Unix sockets. For this approach, you should set `--base-path` to "/tmp", `--event-filename-0` to the name of the _events_ socket, `--event-filename-1` to the name of the _frames_ socket, and `--mode` to "socket". This should work pretty much the same way for a TCP connection, but additional configuration may be required.

### Other parameters
`--output-fps`: The reconstruction output frame rate. Increasing this parameter has a marginal effect on the processing speed.

`--show-display`: Whether or not to show a live view of the reconstruction, using OpenCV display windows.

`--write-video`: If true, writes the reconstructed frames to an .avi file.

`--optimize-c`: If true, will dynamically choose the optimal contrast threshold for deblurring each frame. Causes a significant slow down, especially for higher frame-rate inputs, and can make the reconstruction perform slightly less than real time. If false, then the `--start-c` value provided will be the contrast threshold used for deblurring all frames.

`--optimize-controller`: If true, will attempt to maintain real-time reconstruction performance. The controller dynamically toggles whether contrast threshold optimization is performed (unless `--optimize-c` is false), and adjusts the reconstruction frame rate. If false, will maintain a constant reconstruction frame rate, but may fall behind real-time performance. The reconstructed video files will be much smoother with this disabled. If the scene dynamics won't change much, and you have the ability to dial in settings ahead of time, it's best to keep this disabled and find (through trail and error) the optimal `--output-fps` value which maintains good performance.

## To-do list
There are some major things left before I can start implementing mEDI. Any assistance from the community would be greatly appreciated

- [ ] Add documentation
- [x] Improve speed
- [ ] Make it more Rusty (follow conventions)
- [x] Fix the contrast threshold optimization
  - Special thanks to [Chen Song](https://github.com/chensong1995) for his assistance with this
- [x] Add reconstruction from a live camera
- [x] Create a controller for when to perform c-value optimization, based on the desired reconstructed frame rate, for real-time applications
  - Needs improvement, however
- [x] Be able to save reconstructed frames as a video easily

## Requirements
- Rust 2021 or higher
- Cargo
- OpenCV and its Rust bindings (installation instructions [here](https://github.com/twistedfall/opencv-rust))
- Other dependencies will download and install automatically when building with Cargo

## Compatibilitity
I've only tested on Ubuntu 20.04. If you go to the effort of getting things running on Windows, let me know what the additional requirements are and I'll add them here. Eventually I'll get a full Windows/Linux/Mac CI pipeline setup.
