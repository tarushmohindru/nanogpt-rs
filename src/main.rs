use crate::config::{CausalSelfAttentionConfig, GPTConfig};
use crate::train::train;
use burn::backend::Autodiff;
use burn::tensor::bf16;
use burn::{Tensor, backend::LibTorch, tensor::Device};
use std::fs::File;
use std::io::Write;

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
