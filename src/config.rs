use burn::{
    Tensor,
    config::Config,
    nn::{Embedding, EmbeddingConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
};

use crate::model::{Block, CausalSelfAttention, GPT, MLP, Transformer};

#[derive(Config, Debug)]
pub struct CausalSelfAttentionConfig {
    pub n_embd: usize,
    pub n_head: usize,
}

impl CausalSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalSelfAttention<B> {
        CausalSelfAttention {
            c_attn: LinearConfig::new(self.n_embd, self.n_embd * 3).init(device),
            c_proj: LinearConfig::new(self.n_embd, self.n_embd).init(device),
            n_embd: self.n_embd,
            n_head: self.n_head,
        }
    }
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    pub n_embd: usize,
    pub block_size: usize,
    pub n_head: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLP<B> {
        MLP {
            c_fc: LinearConfig::new(self.n_embd, self.n_embd * 4).init(device),
            act: Gelu::new(),
            c_proj: LinearConfig::new(self.n_embd * 4, self.n_embd).init(device),
        }
    }
}

#[derive(Config, Debug)]
pub struct BlockConfig {
    pub n_embd: usize, //
    pub n_head: usize,
    pub block_size: usize,
    pub n_layer: usize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            ln1: LayerNormConfig::new(self.n_embd).init(device),
            att: CausalSelfAttentionConfig::new(self.n_embd, self.n_head).init(device),
            ln2: LayerNormConfig::new(self.n_embd).init(device),
            mlp: MLPConfig::new(self.n_embd, self.block_size, self.n_head).init(device),
        }
    }
}

#[derive(Config, Debug)]
pub struct TransformerConfig {
    pub n_embd: usize,
    pub vocab_size: usize,
    pub block_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Transformer<B> {
        let layers = (0..self.n_layer)
            .map(|_| {
                BlockConfig::new(self.n_embd, self.n_head, self.block_size, self.n_layer)
                    .init(device)
            })
            .collect();

        Transformer {
            wte: EmbeddingConfig::new(self.vocab_size, self.n_embd).init(device),
            wpe: EmbeddingConfig::new(self.block_size, self.n_embd).init(device),
            h: layers,
            ln_f: LayerNormConfig::new(self.n_embd).init(device),
            block_size: self.block_size,
        }
    }
}

#[derive(Config, Debug)]
pub struct GPTConfig {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
}

impl GPTConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GPT<B> {
        GPT {
            transformer: TransformerConfig::new(
                self.n_embd,
                self.vocab_size,
                self.block_size,
                self.n_layer,
                self.n_head,
            )
            .init(device),
        }
    }
}
