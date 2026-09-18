mod dims;
mod gpt;
mod gpu;
mod shape;
mod tape;
mod tokenizer;

use std::path::PathBuf;

use crate::tape::Device;
use clap::Parser;
use gpt::Gpt;
use tape::Tape;
use tokenizer::Tokenizer;

#[derive(Parser, Clone)]
pub struct Args {
    #[arg(long, default_value = "gpu")]
    device: Device,
    #[arg(long, default_value_t = 128)]
    d_model: usize,
    #[arg(long, default_value_t = 2)]
    n_head: usize,
    #[arg(long, default_value_t = 4)]
    n_layer: usize,
    #[arg(long, default_value_t = 128)]
    n_steps: usize,
    #[arg(long, default_value_t = 16)]
    n_samples: usize,
    #[arg(long, default_value_t = 64)]
    n_context: usize,
    #[arg(long, default_value_t = 128)]
    n_tokens: usize,
    #[arg(long, default_value_t = 0.01)]
    learning_rate: f32,
    #[arg(long, default_value_t = 0.85)]
    beta_1: f32,
    #[arg(long, default_value_t = 0.99)]
    beta_2: f32,
    #[arg(long, default_value_t = 1e-8)]
    eps_adam: f32,
    #[arg(long, default_value = "tinystories-1000.txt")]
    input: PathBuf,
}

fn main() {
    let args = Args::parse();
    let input = args.input.clone();
    let mut tokenizer = Tokenizer::new(args.n_tokens);
    println!("Tokenizing...");
    tokenizer.train(&input);
    println!("Vocab: {}", tokenizer.vocab.len() + 1);

    let mut tape = Tape::new(args.device);
    let mut gpt = Gpt::new(&mut tape, &tokenizer, args);
    println!("Params: {}", gpt.n_params);
    println!("Weights: {}", gpt.n_weights);

    gpt.train(&mut tape, &mut tokenizer, &input);
    gpt.infer(&mut tape, &mut tokenizer);
}
