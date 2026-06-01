// TODO
// - Have the tape as Rc<Refcell<T>> on Value
// - Vector type that implements dot, softmax, rmsnorm

use rand::distributions::Distribution;
use rand::thread_rng;
use rand::{distributions::Uniform, seq::SliceRandom};
use std::{
    fs,
    io::{self, BufRead},
    vec,
};

const N_EMBED: usize = 16; // embedding dimension
const N_HEAD: usize = 4; // Number of attention heads
const N_LAYER: usize = 1; // number of layers
const BLOCK_SIZE: usize = 16; // Context window
const HEAD_DIM: usize = N_EMBED / N_HEAD;

fn main() -> std::io::Result<()> {
    let file = fs::File::open("input.txt")?;
    let reader = io::BufReader::new(file);

    let mut docs = vec![];
    for doc in reader.lines() {
        docs.push(doc.unwrap());
    }
    let mut rng = thread_rng();
    docs.shuffle(&mut rng);

    // Each unique character becomes a token assigned a unique integer
    let mut uchars: Vec<char> = docs.join("").chars().collect();
    uchars.sort();
    uchars.dedup();

    // ID for the special beginning of sequence character
    let bos = uchars.len();
    let vocab_size = bos + 1;

    let mut tape = Tape::new();
    let gpt = Gpt::new(&mut tape, vocab_size);

    println!("Tape len: {}", tape.values.len());

    Ok(())
}

#[derive(Clone)]
struct Value {
    // Output of the forward pass
    data: f64,
    // Gradient accumulated during the backward pass.
    // How much nudging this value affects the final loss
    grad: f64,
    // Inputs for this value
    children: Vec<usize>,
    // How changing each input affects this value
    local_grads: Vec<f64>,
}

impl Value {
    fn new(data: f64, children: Vec<usize>, local_grads: Vec<f64>) -> Self {
        Self {
            data,
            grad: 0.0,
            children,
            local_grads,
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
            for j in 0..v.children.len() {
                let child_i = v.children[j];
                let local_grad = v.local_grads[j];

                self.values[child_i].grad += local_grad * v.grad;
            }
        }
    }

    fn rmsnorm(&mut self, x: &[usize]) -> Vec<usize> {
        let mut ms = 0.0;
        for &v in x {
            let i = self.mul(v, v);
            ms += self.values[i].data;
        }

        let scale = self.value((ms + 1e-5).powf(-0.5));
        Vec::from_iter(x.iter().map(|&v| self.mul(v, scale)))
    }

    fn softmax(&mut self, x: &[usize]) -> Vec<usize> {
        let max = x
            .iter()
            .map(|&i| self.values[i].data)
            .fold(0.0f64, f64::max);
        let neg_max = self.value(-max);

        let mut exps = vec![];
        let mut total = 0.0;
        for i in x {
            let scaled = self.add(*i, neg_max);
            let exp = self.exp(scaled);
            total += self.values[exp].data;
            exps.push(exp);
        }

        let div = self.value(1.0 / total);
        Vec::from_iter(exps.iter().map(|&e| self.mul(e, div)))
    }

    fn push(&mut self, v: Value) -> usize {
        self.values.push(v);
        self.values.len() - 1
    }

    fn value(&mut self, data: f64) -> usize {
        self.push(Value::new(data, vec![], vec![]))
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let data = self.values[a].data + self.values[b].data;
        self.push(Value::new(data, vec![a, b], vec![1.0, 1.0]))
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let local_grads = vec![self.values[b].data, self.values[a].data];
        let data = self.values[a].data * self.values[b].data;
        self.push(Value::new(data, vec![a, b], local_grads))
    }

    fn neg(&mut self, i: usize) -> usize {
        self.push(Value::new(-1.0, vec![], vec![]));
        self.mul(i, self.values.len() - 1)
    }

    fn pow(&mut self, i: usize, n: f64) -> usize {
        let local_grads = vec![n * self.values[i].data.powf(n - 1.0)];
        self.push(Value::new(
            self.values[i].data.powf(n),
            vec![i],
            local_grads,
        ))
    }

    fn log(&mut self, i: usize) -> usize {
        let local_grads = vec![1.0 / self.values[i].data];
        self.push(Value::new(self.values[i].data.ln(), vec![i], local_grads))
    }

    fn exp(&mut self, i: usize) -> usize {
        let data = self.values[i].data.exp();
        self.push(Value::new(data, vec![i], vec![data]))
    }

    fn relu(&mut self, i: usize) -> usize {
        let local_grads = vec![if self.values[i].data > 0.0 { 1.0 } else { 0.0 }];
        self.push(Value::new(
            self.values[i].data.max(0.0),
            vec![i],
            local_grads,
        ))
    }
}

struct Matrix {
    values: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn new(tape: &mut Tape, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-1.0, 1.0);

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
        &self.values[r..self.cols]
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

struct Layer {
    attn_wq: Matrix,
    attn_wk: Matrix,
    attn_wv: Matrix,
    attn_wo: Matrix,
    mlp_fc1: Matrix,
    mlp_fc2: Matrix,
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
        }
    }
}

struct Gpt {
    wte: Matrix,
    wpe: Matrix,
    lm_head: Matrix,
    layers: Vec<Layer>,
    size: usize,
    keys: Vec<Vec<usize>>,
    values: Vec<Vec<usize>>,
}

impl Gpt {
    fn new(tape: &mut Tape, vocab_size: usize) -> Self {
        Self {
            wte: Matrix::new(tape, vocab_size, N_EMBED),
            wpe: Matrix::new(tape, BLOCK_SIZE, N_EMBED),
            lm_head: Matrix::new(tape, vocab_size, N_EMBED),
            layers: Vec::from_iter((0..N_LAYER).map(|_| Layer::new(tape, N_EMBED))),
            size: tape.values.len(),
            keys: vec![],
            values: vec![],
        }
    }

    fn forward(&mut self, tape: &mut Tape, token_id: usize, pos_id: usize) -> Vec<usize> {
        // Combined token and position embedding.
        let wte = self.wte.row(token_id);
        let wpe = self.wpe.row(pos_id);
        let mut x: Vec<usize> = Vec::from_iter((0..N_EMBED).map(|i| tape.add(wte[i], wpe[i])));

        for layer in &mut self.layers {
            let x_residual = x.clone();
            x = tape.rmsnorm(&x);

            // Query, key, and value vectors for the current embedding.
            let q = layer.attn_wq.linear(tape, &x);
            let k = layer.attn_wk.linear(tape, &x);
            let v = layer.attn_wv.linear(tape, &x);

            // Cache the key and value embeddings.
            self.keys.push(k.clone());
            self.values.push(v.clone());
            let n_ctx = self.keys.len();

            let mut x_attn = vec![];
            for h in 0..N_HEAD {
                let h_start = h * HEAD_DIM;
                let h_end = h_start + HEAD_DIM;
                let q_head = &q[h_start..h_end];

                // The dot product between the query and all keys determines how 
                // relevant each token is to the query. Scale to make variance roughly 1.
                let scale = tape.value(1.0 / (HEAD_DIM as f64).powf(0.5));
                let attn_logits = Vec::from_iter((0..n_ctx).map(|t| {
                    let mut dot = tape.value(0.0);
                    for i in 0..HEAD_DIM {
                        let q = q_head[i];
                        let k = self.keys[t][h_start + i];
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
                        let v = self.values[t][h_start + i];
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
}
