mod dims;
mod gpt;
mod shape;
mod tape;
mod tokenizer;

use gpt::Gpt;
use tape::Tape;
use tokenizer::Tokenizer;

fn main() {
    let mut tokenizer = Tokenizer::new();
    tokenizer.train("tinystories-1000.txt");
    println!("Vocab: {}", tokenizer.vocab.len() + 1);

    let mut tape = Tape::new();
    let mut gpt = Gpt::new(&mut tape, &tokenizer);
    println!("Params: {}", gpt.n_params);
    println!("Weights: {}", gpt.n_weights);

    gpt.train(&mut tape, &mut tokenizer, "tinystories-full.txt");
    gpt.infer(&mut tape, &mut tokenizer);
}
