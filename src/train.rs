use burn::optim::Adam;
use burn::optim::GradientsAccumulator;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::{
    Tensor,
    config::Config,
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::Int,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use serde_json::json;

use crate::{
    config::GPTConfig,
    data::{TextBatch, create_dataloader},
    model::GPT,
    wandb::WandbRun,
};

impl<B: Backend> GPT<B> {
    pub fn forward_classification(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(idx);

        let [b, t, vocab_size] = output.dims();
        let output = output.clone().reshape([b * t, vocab_size]);
        let targets = targets.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for GPT<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> burn::train::TrainOutput<Self::Output> {
        let item = self.forward_classification(item.tokens, item.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for GPT<B> {
    type Input = TextBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item.tokens, item.targets)
    }
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: GPTConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1)]
    pub num_epochs: usize,
    #[config(default = 256)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    #[config(default = 2000)]
    warmup_steps: usize,
    #[config(default = 3e-5)]
    learning_rate_min: f64,
    #[config(default = 2)]
    grad_accum_steps: usize,
    #[config(default = 5000)]
    pub checkpoint_interval: usize,
}

pub fn get_lr(step: usize, training_config: &TrainingConfig, total_steps: usize) -> f64 {
    if step < training_config.warmup_steps {
        return training_config.learning_rate * (step as f64 / training_config.warmup_steps as f64);
    }

    let progress = (step - training_config.warmup_steps) as f64
        / (total_steps - training_config.warmup_steps) as f64;
    let cosine = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    training_config.learning_rate_min
        + cosine * (training_config.learning_rate - training_config.learning_rate_min)
}

fn save_checkpoint<B: AutodiffBackend>(model: &GPT<B>, step: usize, artifact_dir: &str) {
    let checkpoint_dir = format!("{artifact_dir}/checkpoints/step-{step}");
    std::fs::create_dir_all(&checkpoint_dir).ok();

    model
        .clone()
        .save_file(format!("{checkpoint_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save model checkpoint");

    std::fs::write(
        format!("{checkpoint_dir}/meta.json"),
        serde_json::json!({ "step": step }).to_string(),
    )
    .ok();

    println!("[Checkpoint] Saved at step {step}");
}

fn load_checkpoint<B: AutodiffBackend>(
    artifact_dir: &str,
    config: &TrainingConfig,
    device: &B::Device,
) -> Option<(GPT<B>, OptimizerAdaptor<Adam, GPT<B>, B>, usize)> {
    let checkpoint_dir = format!("{artifact_dir}/checkpoints");
    let (step, path) = std::fs::read_dir(&checkpoint_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            let step = name.strip_prefix("step-")?.parse::<usize>().ok()?;
            Some((step, e.path()))
        })
        .max_by_key(|(step, _)| *step)?;

    let path = path.to_str()?;

    let model = config
        .model
        .init::<B>(device)
        .load_file(format!("{path}/model"), &CompactRecorder::new(), device)
        .expect("Failed to load model checkpoint");

    // Optimizer restarts fresh — loses momentum but model weights are restored
    let optim: OptimizerAdaptor<Adam, GPT<B>, B> = config.optimizer.init();

    println!("[Checkpoint] Resuming model from step {step} (optimizer state reset)");
    Some((model, optim, step))
}

pub fn train<B: AutodiffBackend>(device: B::Device, artifact_dir: &str, resume: bool) {
    let config_model = GPTConfig::new(1024, 50257, 12, 12, 768);
    let config_optimizer = AdamConfig::new();
    let config = TrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    let total_steps = 100_000;
    let val_interval = 500;

    // Resume from checkpoint or start fresh
    let (mut model, mut optim, mut step) = if resume {
        load_checkpoint::<B>(artifact_dir, &config, &device).unwrap_or_else(|| {
            println!("No checkpoint found, starting fresh");
            let optim: OptimizerAdaptor<Adam, GPT<B>, B> = config.optimizer.init();
            (
                config
                    .model
                    .init::<B>(&device)
                    .apply_weight_init(config.model.n_layer),
                optim,
                0,
            )
        })
    } else {
        let optim: OptimizerAdaptor<Adam, GPT<B>, B> = config.optimizer.init();
        (
            config
                .model
                .init::<B>(&device)
                .apply_weight_init(config.model.n_layer),
            optim,
            0,
        )
    };

    let (train_dataset, mut test_dataset) = create_dataloader(
        config.model.block_size,
        config.batch_size,
        config.num_workers,
        config.seed,
        &device,
    );

    let mut accum_step = 0;
    let mut accumulator = GradientsAccumulator::new();

    let mut wandb = WandbRun::init(
        "tarushmohindru",
        "nanogpt-rs-pretrain-1.0",
        json!({
            "batch_size": config.batch_size,
            "grad_accum_steps": config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "warmup_steps": config.warmup_steps,
            "block_size": config.model.block_size,
            "n_layer": config.model.n_layer,
            "n_head": config.model.n_head,
            "n_embd": config.model.n_embd,
        }),
    );

    for batch in train_dataset.iter() {
        let [b, t] = batch.tokens.dims();
        let logits = model.forward(batch.tokens);

        let vocab_size = logits.dims()[2];
        let logits = logits.reshape([b * t, vocab_size]);
        let targets = batch.targets.reshape([b * t]);

        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets);

        // Scale, backward immediately, accumulate — graph freed each micro-step
        let loss_scaled = loss.clone() / config.grad_accum_steps as f64;
        let step_grads = GradientsParams::from_grads(loss_scaled.backward(), &model);
        accumulator.accumulate(&model, step_grads);
        accum_step += 1;

        if accum_step % config.grad_accum_steps == 0 {
            let lr = get_lr(step, &config, total_steps);
            model = optim.step(lr, model, accumulator.grads());
            accumulator = GradientsAccumulator::new(); // reset for next window

            let loss_val = loss.clone().into_scalar().to_f64();
            let perplexity = loss_val.exp();
            let tokens_seen =
                step * config.batch_size * config.model.block_size * config.grad_accum_steps;

            println!(
                "[Step {step} | Tokens {tokens_seen}] Loss {loss_val:.4} | \
                 Perplexity {perplexity:.2} | LR {lr:.2e}"
            );

            wandb.log(json!({
                "train/loss": loss_val,
                "train/perplexity": perplexity,
                "train/lr": lr,
                "tokens_seen": tokens_seen,
            }));

            if step % val_interval == 0 {
                let model_valid = model.valid();
                let mut val_iter = test_dataset.iter();
                if let Some(val_batch) = val_iter.next() {
                    let tokens = val_batch.tokens.inner();
                    let targets = val_batch.targets.inner();
                    let [b, t] = tokens.dims();
                    let logits = model_valid.forward(tokens);
                    let vocab_size = logits.dims()[2];
                    let loss = CrossEntropyLossConfig::new().init(&device).forward(
                        logits.reshape([b * t, vocab_size]),
                        targets.reshape([b * t]),
                    );
                    let val_loss = loss.into_scalar().to_f64();
                    println!(
                        "[Val - Step {step}] Loss {val_loss:.4} | Perplexity {:.2}",
                        val_loss.exp()
                    );
                    wandb.log(json!({
                        "val/loss": val_loss,
                        "val/perplexity": val_loss.exp(),
                    }));
                }
            }

            if step % config.checkpoint_interval == 0 && step > 0 {
                save_checkpoint(&model, step, artifact_dir);
            }

            step += 1;
            accum_step = 0;

            if step >= total_steps {
                break;
            }
        }
    }

    // Final checkpoint
    save_checkpoint(&model, step, artifact_dir);
    wandb.finish();
}
