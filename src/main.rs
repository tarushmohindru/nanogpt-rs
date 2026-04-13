use crate::train::train;
use burn::backend::{Autodiff, LibTorch};
use burn::tensor::bf16;

mod config;
mod data;
mod model;
mod train;
mod wandb;

fn main() {
    type MyBackend = LibTorch<bf16>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    // let gpt = GPTConfig::new(1024, 50257, 12, 12, 768).init::<MyBackend>(&device);

    // println!("{gpt}");

    train::<MyAutodiffBackend>(device, "checkpoints", false);
}
