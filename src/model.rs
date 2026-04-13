use core::f32;

use burn::{
    nn::{
        Embedding, Gelu, LayerNorm, Linear,
        attention::{MhaInput, MultiHeadAttention},
    },
    prelude::*,
    tensor::activation::softmax,
};

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    pub mha: MultiHeadAttention<B>,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = x.device();
        let [b, t, c] = x.dims();

        // let qkv = self.c_attn.forward(x);
        // let q = qkv.clone().slice([0..b, 0..t, 0..self.n_embd]);
        // let k = qkv
        //     .clone()
        //     .slice([0..b, 0..t, self.n_embd..2 * self.n_embd]);
        // let v = qkv
        //     .clone()
        //     .slice([0..b, 0..t, 2 * self.n_embd..3 * self.n_embd]);

        // let q = q
        //     .reshape([b, t, self.n_head, c / self.n_head])
        //     .swap_dims(1, 2);
        // let k = k
        //     .reshape([b, t, self.n_head, c / self.n_head])
        //     .swap_dims(1, 2);
        // let v = v
        //     .reshape([b, t, self.n_head, c / self.n_head])
        //     .swap_dims(1, 2);

        // let mask = Tensor::tril_mask([1, 1, t, t], 0, &device);

        // let scale = 1.0 / ((c / self.n_head) as f64).sqrt();
        // let mut att = (q.matmul(k.swap_dims(2, 3))) * (scale);
        // att = att.mask_fill(mask, f32::NEG_INFINITY);
        // let scores = softmax(att, 3);
        // let mut y = scores.matmul(v).swap_dims(1, 2).reshape([b, t, c]);
        // y = self.c_proj.forward(y);
        // return y;

        let mask = Tensor::tril_mask([t, t], 0, &device).bool_not();
        let mha_input = MhaInput::self_attn(x).mask_attn(mask);
        self.mha.forward(mha_input).context
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    pub c_fc: Linear<B>,
    pub act: Gelu,
    pub c_proj: Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = self.act.forward(x);
        let x = self.c_proj.forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub ln1: LayerNorm<B>,
    pub att: CausalSelfAttention<B>,
    pub ln2: LayerNorm<B>,
    pub mlp: MLP<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.att.forward(self.ln1.forward(x));
        let x = x.clone() + self.mlp.forward(self.ln2.forward(x));
        x
    }
}

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    pub wte: Embedding<B>,
    pub wpe: Embedding<B>,
    pub h: Vec<Block<B>>,
    pub ln_f: LayerNorm<B>,
    pub block_size: usize,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = idx.device();
        let [b, t] = idx.dims();

        assert!(t <= self.block_size);

        let pos = Tensor::<B, 1, Int>::arange(0..(t as i64), &device).unsqueeze::<2>();
        let pos_embd = self.wpe.forward(pos);
        let tok_embd = self.wte.forward(idx);
        let mut x = tok_embd + pos_embd;

        for block in &self.h {
            x = block.forward(x);
        }

        x = self.ln_f.forward(x);
        x
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    pub transformer: Transformer<B>,
}

impl<B: Backend> GPT<B> {
    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.transformer.forward(idx);

        let wte_weight = self.transformer.wte.weight.val();
        let vocab_size = wte_weight.dims()[0];
        let [b, t, c] = x.dims();
        let x = x.reshape([b * t, c]);
        let logits = x.matmul(wte_weight.transpose());
        let logits = logits.reshape([b, t, vocab_size]);

        logits
    }

    pub fn apply_weight_init(mut self, n_layer: usize) -> Self {
        let scale = (1.0 / (2.0 * n_layer as f64).sqrt()) as f32;

        for block in self.transformer.h.iter_mut() {
            // Scale MLP output projection
            let w = block.mlp.c_proj.weight.val();
            let scaled = w * scale;
            block.mlp.c_proj.weight = burn::nn::Param::from_tensor(scaled);
        }

        self
    }
}
