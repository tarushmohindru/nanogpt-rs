use burn::{Tensor, backend::LibTorch, tensor::Device};
use std::fs::File;
use std::io::Write;

use crate::config::{CausalSelfAttentionConfig, GPTConfig};

mod config;
mod model;

fn main() {
    type MyBackend = LibTorch;
    let device = burn::backend::libtorch::LibTorchDevice::Cuda(0);
    // let gpt = GPTConfig::new(1024, 50257, 12, 12, 768).init::<MyBackend>(&device);

    // println!("{gpt}");

    let attn = CausalSelfAttentionConfig::new(12, 3).init::<MyBackend>(&device);
    let x = Tensor::<MyBackend, 3>::random(
        [5, 8, 12],
        burn::tensor::Distribution::Normal(0., 1.),
        &device,
    );

    let mut file = File::create("output.txt").unwrap();
    writeln!(file, "x: {}", x.to_data()).unwrap();
    writeln!(file, "c_attn (qkv): {}", attn.c_attn.weight.val().to_data()).unwrap();
    writeln!(file, "c_proj: {}", attn.c_proj.weight.val().to_data()).unwrap();
    writeln!(file, "y: {}", attn.forward(x).to_data()).unwrap();
}
