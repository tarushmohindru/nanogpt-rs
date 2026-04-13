use std::sync::Arc;

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataloader::batcher::Batcher,
        dataset::{Dataset, HuggingfaceDatasetLoader},
    },
    prelude::Backend,
    tensor::{Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};
use tiktoken_rs::r50k_base;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FineWebItem {
    pub text: String,
}

use burn::data::dataset::SliceDataset;

pub fn load_fineweb() -> impl Dataset<FineWebItem> {
    HuggingfaceDatasetLoader::new("HuggingFaceFW/fineweb-edu")
        .with_subset("sample-10BT")
        .dataset("train")
        .unwrap()
}

pub fn load_fineweb_train_test() -> (impl Dataset<FineWebItem>, impl Dataset<FineWebItem>) {
    let dataset = load_fineweb(); // full train set
    let total = dataset.len();
    let split = (total as f64 * 0.9) as usize;

    let train = SliceDataset::new(Arc::new(dataset), 0, split);
    let test = SliceDataset::new(Arc::new(dataset), split, total);
    (train, test)
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Clone)]
pub struct TextBatcher {
    pub block_size: usize,
    pub tokenizer: Arc<tiktoken_rs::CoreBPE>,
}

impl TextBatcher {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            tokenizer: Arc::new(r50k_base().unwrap()),
        }
    }
}

impl<B: Backend> Batcher<B, FineWebItem, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<FineWebItem>, device: &B::Device) -> TextBatch<B> {
        let tokens = items
            .iter()
            .filter_map(|item| {
                let encoded = self.tokenizer.encode_with_special_tokens(&item.text);
                if encoded.len() < self.block_size + 1 {
                    return None; // skip sequences that are too short
                }
                Some(encoded[..self.block_size + 1].to_vec())
            })
            .map(|t| {
                let data =
                    TensorData::from(t.iter().map(|x| *x as i64).collect::<Vec<_>>().as_slice());
                Tensor::<B, 1, Int>::from_data(data, device)
            })
            .map(|tensor| tensor.reshape([1, self.block_size + 1]))
            .collect::<Vec<_>>();

        let tokens = Tensor::cat(tokens, 0); // [b, block_size+1]
        let b = tokens.dims()[0];

        let inputs = tokens.clone().slice([0..b, 0..self.block_size]);
        let targets = tokens.slice([0..b, 1..self.block_size + 1]);

        TextBatch {
            tokens: inputs, // [b, block_size]
            targets,        // [b, block_size]
        }
    }
}

pub fn create_dataloader<B: Backend>(
    block_size: usize,
    batch_size: usize,
    num_workers: usize,
    seed: u64,
    device: &B::Device,
) -> (
    impl Iterator<Item = TextBatch<B>>,
    impl Iterator<Item = TextBatch<B>>,
) {
    let (train_dataset, test_dataset) = load_fineweb_train_test();
    let batcher = TextBatcher::new(block_size);

    let train_loader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(train_dataset)
        .iter();

    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(test_dataset)
        .iter();

    (train_loader, test_loader)
}
