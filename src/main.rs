mod gpt;
mod tape;
mod tokenizer;

use gpt::Gpt;
use tape::Tape;
use tokenizer::Tokenizer;

fn main() {
    let mut tokenizer = Tokenizer::new();
    tokenizer.train("tinystories-10.txt");
    println!("Vocab: {}", tokenizer.vocab.len() + 1);

    let mut tape = Tape::new();
    let mut gpt = Gpt::new(&mut tape, &tokenizer);
    println!("Size: {}", gpt.size);

    gpt.train(&mut tape, &mut tokenizer, "tinystories-full.txt");
    gpt.infer(&mut tape, &mut tokenizer);
}
