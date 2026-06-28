use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use rand::{distributions::Uniform};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::ops::Add;
use std::{fs, vec};

const N_EMBED: usize = 16; // embedding dimension
const N_HEAD: usize = 4; // Number of attention heads
const N_LAYER: usize = 4; // number of layers
const BLOCK_SIZE: usize = 16; // Context window
const HEAD_DIM: usize = N_EMBED / N_HEAD;
const TRAINING_STEPS: usize = 100_00;
const N_SAMPLES: usize = 16;
const LEARNING_RATE: f32 = 0.01;
const BETA_1: f32 = 0.85;
const BETA_2: f32 = 0.99;
const EPS_ADAM: f32 = 1e-8;
const N_RULES: usize = 64;
const N_CHARS: usize = 6;

// TODO: GPU

fn main() {
    let mut tokenizer = Tokenizer::new();
    tokenizer.train("input.txt");
    println!("Vocab: {}", tokenizer.vocab.len() + 1);

    let mut tape = Tape::new();
    let mut gpt = Gpt::new(&mut tape, &tokenizer);
    println!("Params: {}", gpt.size);

    gpt.train(&mut tape, &mut tokenizer, "input.txt");
    gpt.infer(&mut tape, &tokenizer);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct Token {
    bytes: [Option<u8>; N_CHARS],
}

impl Token {
    fn new(b: u8) -> Self {
        let mut bytes: [Option<u8>; N_CHARS] = [None; N_CHARS];
        bytes[0] = Some(b);

        Self { bytes }
    }
}

impl Add for Token {
    type Output = Token;

    fn add(self, other: Self) -> Token {
        let mut bytes = self.bytes;

        let mut j = 0;
        for i in 0..N_CHARS {
            if self.bytes[i].is_some() {
                continue;
            }

            if other.bytes[j].is_none() {
                break;
            }

            bytes[i] = other.bytes[j];
            j += 1;
        }

        Token { bytes }
    }
}

struct Tokenizer {
    rules: HashMap<(Token, Token), usize>,
    encoder: HashMap<Token, usize>,
    vocab: Vec<Token>,
    sequence: Vec<Token>,
    prev: Vec<Option<usize>>,
    next: Vec<Option<usize>>,
    dead: Vec<bool>,
    pairs: HashMap<(Token, Token), HashSet<usize>>,
    heap: BinaryHeap<(usize, (Token, Token))>,
    bos: usize,
}

impl Tokenizer {
    fn add_input(&mut self, input: &str) {
        let bytes = input.as_bytes();
        for (i, pair) in bytes.windows(2).enumerate() {
            let left = Token::new(pair[0]);
            let right = Token::new(pair[1]);
            let pos = self.sequence.len();

            self.sequence.push(left);
            self.pairs.entry((left, right)).or_default().insert(pos);
            self.prev.push(if i == 0 { None } else { Some(pos - 1) });
            self.next.push(Some(pos + 1));
        }

        if let Some(&b) = bytes.last() {
            let token = Token::new(b);
            self.prev.push(Some(self.sequence.len() - 1));
            self.sequence.push(token);
            self.next.push(None);
        }

        self.dead.resize(self.sequence.len(), false);
    }

    // Move position from one key (pair) to another
    fn rekey_position(
        &mut self,
        from_key: (Token, Token),
        to_key: (Token, Token),
        old_pos: usize,
        new_pos: usize,
    ) {
        if let Some(positions) = self.pairs.get_mut(&from_key) {
            positions.remove(&old_pos);
        }

        self.pairs.entry(to_key).or_default().insert(new_pos);

        if self.bos == 0 {
            let freq = self.pairs[&to_key].len();
            self.heap.push((freq, to_key));
        }
    }

    fn merge_pair(&mut self, pair: (Token, Token)) {
        let mut rule_added = self.bos != 0;
        if let Some(positions) = self.pairs.remove(&pair) {
            for start in positions {
                if self.dead[start] {
                    continue;
                }

                if let Some(end) = self.next[start] {
                    // Tokens might have been mutated invalidating the pair
                    if (self.sequence[start], self.sequence[end]) != pair {
                        continue;
                    }

                    let merged = self.sequence[start] + self.sequence[end];
                    if rule_added == false {
                        let rank = self.rules.len();
                        self.rules.insert(pair, rank);
                        rule_added = true
                    };

                    if let Some(left) = self.prev[start] {
                        let from_key = (self.sequence[left], self.sequence[start]);
                        let to_key = (self.sequence[left], merged);
                        self.rekey_position(from_key, to_key, left, left);
                    }

                    // Unlink end from the chain start -> end -> right
                    let right = self.next[end];
                    self.next[start] = right;
                    self.dead[end] = true;
                    if let Some(right) = right {
                        self.prev[right] = Some(start);

                        let from_key = (self.sequence[end], self.sequence[right]);
                        let to_key = (merged, self.sequence[right]);
                        self.rekey_position(from_key, to_key, end, start);
                    }

                    self.sequence[start] = merged;
                }
            }
        }
    }

    fn learn_rules(&mut self) {
        while let Some(rule) = &self.heap.pop() {
            let freq = rule.0;
            let pair = rule.1;

            if let Some(positions) = self.pairs.get(&pair) {
                let current_freq = positions.len();
                if freq != current_freq {
                    self.heap.push((current_freq, rule.1));
                    continue;
                }

                self.merge_pair(pair);
            }

            if self.rules.len() == N_RULES {
                break;
            }
        }
    }

    fn reset_scratch(&mut self) {
        self.sequence.clear();
        self.prev.clear();
        self.next.clear();
        self.pairs.clear();
        self.heap.clear();
    }

    fn encode(&mut self, input: &str) -> [usize; BLOCK_SIZE] {
        self.reset_scratch();
        self.add_input(input);
        self.dead.fill(false);

        for key in self.pairs.keys() {
            if let Some(&rank) = self.rules.get(key) {
                self.heap.push((self.rules.len() - rank, *key));
            }
        }

        while let Some(rule) = self.heap.pop() {
            let token = rule.1;
            self.merge_pair(token);
        }

        let mut doc = [self.bos; BLOCK_SIZE];
        let mut pos = 0;
        for (i, &t) in self.sequence.iter().enumerate() {
            if self.dead[i] {
                continue;
            }

            if pos >= BLOCK_SIZE {
                break;
            }

            if let Some(&j) = self.encoder.get(&t) {
                doc[pos] = j;
                pos += 1;
            };
        }

        doc
    }

    fn get_vocab(&self) -> Vec<Token> {
        let mut vocab: Vec<_> = self
            .sequence
            .iter()
            .enumerate()
            .filter(|(i, _)| !self.dead[*i])
            .map(|(_, c)| *c)
            .collect();

        vocab.sort();
        vocab.dedup();

        vocab
    }

    fn get_encoder(&self) -> HashMap<Token, usize> {
        self.vocab
            .iter()
            .copied()
            .enumerate()
            .map(|(i, t)| (t, i))
            .collect()
    }

    fn train(&mut self, path: &str) {
        let file = fs::File::open(path).expect("Input file not found");
        let reader = std::io::BufReader::new(file);

        for line in reader.lines() {
            let input = line.expect("Failed to read line");
            self.add_input(&input);
        }

        self.dead.resize(self.sequence.len(), false);
        self.dead.fill(false);

        for p in &self.pairs {
            let freq = p.1.len();
            self.heap.push((freq, *p.0));
        }

        self.learn_rules();

        self.vocab = self.get_vocab();
        self.encoder = self.get_encoder();
        self.bos = self.vocab.len();
    }

    fn new() -> Self {
        let sequence = vec![];
        let prev = vec![];
        let next = vec![];
        let dead = vec![];
        let vocab = vec![];
        let encoder = HashMap::new();
        let pairs = HashMap::new();
        let rules = HashMap::with_capacity(N_RULES);
        let heap = BinaryHeap::new();

        Self {
            sequence,
            prev,
            next,
            dead,
            vocab,
            encoder,
            pairs,
            rules,
            heap,
            bos: 0,
        }
    }
}

// TODO: Tensor
#[derive(Clone, Copy)]
struct Value {
    // Output of the forward pass
    data: f32,
    // Gradient accumulated during the backward pass.
    // How much nudging this value affects the final loss
    grad: f32,
    // Input position and grad for this value
    inputs: [Option<(usize, f32)>; 2],
}

impl Value {
    fn new(data: f32, inputs: [Option<(usize, f32)>; 2]) -> Self {
        Self {
            data,
            grad: 0.0,
            inputs,
        }
    }
}

struct Tape {
    values: Vec<Value>,
}

impl Tape {
    fn new() -> Self {
        Self { values: vec![] }
    }

    fn backward(&mut self) {
        let n = self.values.len();
        assert!(n > 0);

        self.values[n - 1].grad = 1.0;

        for i in (0..n).rev() {
            let v = self.values[i].clone();
            for input in v.inputs {
                if let Some((j, grad)) = input {
                    self.values[j].grad += grad * v.grad;
                }
            }
        }
    }

    fn rmsnorm(&mut self, x: &[usize]) -> Vec<usize> {
        let mut sum = self.value(1e-5);
        for &v in x {
            let i = self.mul(v, v);
            sum = self.add(sum, i);
        }

        let inv_n = self.value(1.0 / x.len() as f32);
        let mean = self.mul(sum, inv_n);
        let scale = self.pow(mean, -0.5);
        Vec::from_iter(x.iter().map(|&v| self.mul(v, scale)))
    }

    fn softmax(&mut self, x: &[usize]) -> Vec<usize> {
        let data = x.iter().map(|&i| self.values[i].data);
        let max = data.reduce(f32::max).unwrap();
        let neg_max = self.value(-max);

        let mut exps = vec![];
        let mut total = self.value(0.0);
        for &i in x {
            let scaled = self.add(i, neg_max);
            let exp = self.exp(scaled);
            exps.push(exp);
            total = self.add(total, exp);
        }

        let div = self.pow(total, -1.0);
        Vec::from_iter(exps.iter().map(|&e| self.mul(e, div)))
    }

    fn push(&mut self, v: Value) -> usize {
        self.values.push(v);
        self.values.len() - 1
    }

    fn value(&mut self, data: f32) -> usize {
        self.push(Value::new(data, [None, None]))
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.values[a].data + self.values[b].data;
        self.push(Value::new(data, [Some((a, 1.0)), Some((b, 1.0))]))
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let data = self.values[a].data * self.values[b].data;
        let left = Some((a, self.values[b].data));
        let right = Some((b, self.values[a].data));
        self.push(Value::new(data, [left, right]))
    }

    fn neg(&mut self, i: usize) -> usize {
        self.push(Value::new(-1.0, [None, None]));
        self.mul(i, self.values.len() - 1)
    }

    fn pow(&mut self, i: usize, n: f32) -> usize {
        let grad = n * self.values[i].data.powf(n - 1.0);
        let inputs = [Some((i, grad)), None];
        self.push(Value::new(self.values[i].data.powf(n), inputs))
    }

    fn log(&mut self, i: usize) -> usize {
        let grad = 1.0 / self.values[i].data;
        let inputs = [Some((i, grad)), None];
        self.push(Value::new(self.values[i].data.ln(), inputs))
    }

    fn exp(&mut self, i: usize) -> usize {
        let data = self.values[i].data.exp();
        self.push(Value::new(data, [Some((i, data)), None]))
    }

    fn relu(&mut self, i: usize) -> usize {
        let grad = if self.values[i].data > 0.0 { 1.0 } else { 0.0 };
        let inputs = [Some((i, grad)), None];
        self.push(Value::new(self.values[i].data.max(0.0), inputs))
    }
}

struct Matrix {
    values: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(tape: &mut Tape, rows: usize, cols: usize) -> Self {
        // Kaiming initialization
        let mut rng = rand::thread_rng();
        let k = (6.0 / cols as f32).sqrt();
        let dist = Uniform::new(-k, k);

        let n = rows * cols;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let d = dist.sample(&mut rng);
            let v = tape.value(d);
            values.push(v);
        }

        Self { rows, cols, values }
    }

    fn row(&self, r: usize) -> &[usize] {
        let s = r * self.cols;
        &self.values[s..s + self.cols]
    }

    fn linear(&self, tape: &mut Tape, v: &[usize]) -> Vec<usize> {
        let mut out = vec![];

        for r in 0..self.rows {
            let i0 = self.values[r * self.cols];
            let mut acc = tape.mul(i0, v[0]);

            for c in 1..self.cols {
                let i = self.values[r * self.cols + c];
                let p = tape.mul(i, v[c]);
                acc = tape.add(acc, p);
            }

            out.push(acc);
        }

        out
    }
}

struct Cache {
    keys: Vec<Vec<usize>>,
    values: Vec<Vec<usize>>,
}

struct Layer {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
    cache: Cache,
}

impl Layer {
    fn new(tape: &mut Tape, n_embed: usize) -> Self {
        Self {
            attn_wq: Matrix::new(tape, n_embed, n_embed),
            attn_wk: Matrix::new(tape, n_embed, n_embed),
            attn_wv: Matrix::new(tape, n_embed, n_embed),
            attn_wo: Matrix::new(tape, n_embed, n_embed),
            mlp_fc1: Matrix::new(tape, 4 * n_embed, n_embed),
            mlp_fc2: Matrix::new(tape, n_embed, n_embed * 4),
            cache: Cache {
                keys: vec![],
                values: vec![],
            },
        }
    }
}

struct Gpt {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: Vec<Layer>,
    size: usize,
}

impl Gpt {
    fn new(tape: &mut Tape, tokenizer: &Tokenizer) -> Self {
        let vocab_size = tokenizer.vocab.len() + 1;
        Self {
            wte: Matrix::new(tape, vocab_size, N_EMBED),
            wpe: Matrix::new(tape, BLOCK_SIZE, N_EMBED),
            lm_head: Matrix::new(tape, vocab_size, N_EMBED),
            layers: Vec::from_iter((0..N_LAYER).map(|_| Layer::new(tape, N_EMBED))),
            size: tape.values.len(),
        }
    }

    fn forward(&mut self, tape: &mut Tape, token_id: usize, pos_id: usize) -> Vec<usize> {
        // Combined token and position embedding.
        let wte = self.wte.row(token_id);
        let wpe = self.wpe.row(pos_id);

        let mut x = Vec::from_iter((0..N_EMBED).map(|i| tape.add(wte[i], wpe[i])));
        x = tape.rmsnorm(&x);

        for layer in &mut self.layers {
            let x_residual = x.clone();
            x = tape.rmsnorm(&x);

            // Query, key, and value vectors for the current embedding.
            let q = layer.attn_wq.linear(tape, &x);
            let k = layer.attn_wk.linear(tape, &x);
            let v = layer.attn_wv.linear(tape, &x);

            // Cache the key and value embeddings.
            layer.cache.keys.push(k.clone());
            layer.cache.values.push(v.clone());
            let n_ctx = layer.cache.keys.len();

            let mut x_attn = vec![];
            for h in 0..N_HEAD {
                let h_start = h * HEAD_DIM;
                let h_end = h_start + HEAD_DIM;
                let q_head = &q[h_start..h_end];

                // The dot product between the query and all keys determines how
                // relevant each token is to the query. Scale to make variance roughly 1.
                let scale = tape.value(1.0 / (HEAD_DIM as f32).powf(0.5));
                let attn_logits = Vec::from_iter((0..n_ctx).map(|t| {
                    let mut dot = tape.value(0.0);
                    for i in 0..HEAD_DIM {
                        let q = q_head[i];
                        let k = layer.cache.keys[t][h_start + i];
                        let p = tape.mul(k, q);
                        dot = tape.add(dot, p)
                    }
                    tape.mul(dot, scale)
                }));

                // Convert attention to probability distribution (adds up to 1).
                let attn_weights = tape.softmax(&attn_logits);

                // Weighted sum of values determines how much each token contributes.
                let mut head_out = Vec::from_iter((0..HEAD_DIM).map(|i| {
                    let mut sum = tape.value(0.0);
                    for t in 0..n_ctx {
                        let v = layer.cache.values[t][h_start + i];
                        let p = tape.mul(attn_weights[t], v);
                        sum = tape.add(sum, p)
                    }
                    sum
                }));

                x_attn.append(&mut head_out);
            }

            // Reproject heads into one space and add residual
            x = layer.attn_wo.linear(tape, &x_attn);
            for i in 0..x.len() {
                x[i] = tape.add(x[i], x_residual[i])
            }

            // MLP block
            let x_residual = x.clone();
            x = tape.rmsnorm(&x);

            // Project up to 4x the N_EMBED and RELU to make a non-linear transform
            x = layer.mlp_fc1.linear(tape, &x);
            for i in 0..x.len() {
                x[i] = tape.relu(x[i]);
            }

            // Project down and add residual
            x = layer.mlp_fc2.linear(tape, &x);
            for i in 0..x.len() {
                x[i] = tape.add(x[i], x_residual[i]);
            }
        }

        // Output logits
        self.lm_head.linear(tape, &x)
    }

    fn train(&mut self, tape: &mut Tape, tokenizer: &mut Tokenizer, path: &str) {
        let mut file = fs::File::open(path).expect("Input file not found");
        let reader = std::io::BufReader::new(&file);

        let mut offsets = vec![];
        let mut pos = 0u64;
        for line in reader.lines() {
            let line = line.expect("failed to read line");
            offsets.push(pos);
            pos += line.len() as u64 + 1;
        }

        let num_lines = offsets.len();
        let step_width = TRAINING_STEPS.to_string().len();

        let mut m: Vec<f32> = vec![0.0; self.size];
        let mut v: Vec<f32> = vec![0.0; self.size];
        for step in 0..TRAINING_STEPS {
            let offset = offsets[step % num_lines];
            file.seek(SeekFrom::Start(offset)).unwrap_or_else(|e| {
                panic!("Failed to seek to {offset}: {e}");
            });

            let reader = BufReader::new(&file);
            let input = reader
                .lines()
                .next()
                .expect("no line at offset")
                .expect("failed to read line");

            let encoded = tokenizer.encode(&input);

            let mut sum = tape.value(0.0);
            let mut n = 0;
            for (pos_id, pair) in encoded.windows(2).enumerate() {
                let token_id = pair[0];
                let target_id = pair[1];
                let logits = self.forward(tape, token_id, pos_id);
                let probs = tape.softmax(&logits);
                // TODO: Get rid of epsilon hack
                let eps = tape.value(1e-9);
                let prob = tape.add(probs[target_id], eps);
                let loss_t = tape.log(prob);
                let loss = tape.neg(loss_t);

                sum = tape.add(sum, loss);
                n += 1;

                if target_id == tokenizer.bos {
                    break;
                }
            }

            let inv_n = tape.value(1.0 / n as f32);
            let loss = tape.mul(sum, inv_n);

            println!(
                "Loss {:0width$} / {}: {}",
                step + 1,
                TRAINING_STEPS,
                tape.values[loss].data,
                width = step_width
            );

            tape.backward();

            let stepf = step as f32;
            // Linear learning rate decay
            let lr_t = LEARNING_RATE * (1.0 - (stepf / TRAINING_STEPS as f32));
            for i in 0..self.size {
                let p = &mut tape.values[i];
                m[i] = BETA_1 * m[i] + (1.0 - BETA_1) * p.grad;
                v[i] = BETA_2 * v[i] + (1.0 - BETA_2) * p.grad.powf(2.0);
                let m_hat = m[i] / (1.0 - BETA_1.powf(stepf + 1.0));
                let v_hat = v[i] / (1.0 - BETA_2.powf(stepf + 1.0));
                p.data -= lr_t * m_hat / (v_hat.powf(0.5) + EPS_ADAM);
                p.grad = 0.0;
            }

            tape.values.truncate(self.size);
            for layer in &mut self.layers {
                layer.cache.keys.clear();
                layer.cache.values.clear();
            }
        }
    }

    fn infer(&mut self, tape: &mut Tape, tokenizer: &Tokenizer) {
        for _ in 0..N_SAMPLES {
            let mut rng = thread_rng();
            let mut token_id = tokenizer.bos;
            let mut sample = vec![];

            for pos_id in 0..BLOCK_SIZE {
                let logits = self.forward(tape, token_id, pos_id);
                let probs = tape.softmax(&logits);
                let weights = Vec::from_iter(probs.iter().map(|&v| tape.values[v].data));
                let dist = WeightedIndex::new(&weights).unwrap();
                token_id = dist.sample(&mut rng);

                if token_id == tokenizer.bos {
                    break;
                }

                let t = tokenizer.vocab[token_id];
                sample.extend(t.bytes.iter().flatten().copied());
            }

            tape.values.truncate(self.size);
            for layer in &mut self.layers {
                layer.cache.keys.clear();
                layer.cache.values.clear();
            }

            let s = String::from_utf8(sample).expect("invalid utf-8");
            println!("Sample: {}", s);
        }
    }
}
