use crate::Args;
use crate::shape::Shape;
use crate::tape::Tape;
use crate::tokenizer::Tokenizer;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::time::Instant;
use std::{fs, usize, vec};

struct Layer {
    attn_wq: usize,
    attn_wk: usize,
    attn_wv: usize,
    attn_wo: usize,
    mlp_fc1: usize,
    mlp_fc2: usize,
}

impl Layer {
    fn new(tape: &mut Tape, d_model: usize, n_layer: usize) -> Self {
        // GPT-2 trick to stop the residual stream's standard deviation
        // from growing with depth at initialization
        let wo_fc2_scale = 1.0 / (2.0 * n_layer as f32).sqrt();
        let attn_shape = &[d_model, d_model];
        let attn_wq = tape.random(attn_shape, 1.0);
        let attn_wk = tape.random(attn_shape, 1.0);
        let attn_wv = tape.random(attn_shape, 1.0);
        let attn_wo = tape.random(attn_shape, wo_fc2_scale);

        let mlp_fc1 = tape.random(&[d_model, d_model * 4], 1.0);
        let mlp_fc2 = tape.random(&[d_model * 4, d_model], wo_fc2_scale);

        Self {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            mlp_fc1,
            mlp_fc2,
        }
    }
}

pub struct Gpt {
    wte: usize,
    lm_head: usize,
    layers: Vec<Layer>,
    pub n_params: usize,
    pub n_weights: usize,
    pub args: Args,
}

impl Gpt {
    pub fn new(tape: &mut Tape, tokenizer: &Tokenizer, args: Args) -> Self {
        let d_model = args.d_model;
        let n_layer = args.n_layer;
        let vocab_size = tokenizer.vocab.len() + 1;
        let wte = tape.random(&[vocab_size, d_model], 1.0);
        let lm_head = tape.random(&[d_model, vocab_size], 1.0);
        let layers = Vec::from_iter((0..n_layer).map(|_| Layer::new(tape, d_model, n_layer)));
        let n_params = tape.data.len();
        let n_weights = tape.weights.len();

        // Add moment buffers
        let n = tape.weights.len();
        for _ in 0..2 {
            for i in 0..n {
                let data_offset = tape.data.len();
                let shape = tape.weights[i].shape;

                for _ in 0..shape.product() {
                    tape.data.push(0.0);
                }

                tape.push_weight(data_offset, shape, None, None, true);
            }
        }

        Self {
            wte,
            lm_head,
            layers,
            n_params,
            n_weights,
            args,
        }
    }

    // Times 3 to account for moment buffers
    fn truncate(&mut self, tape: &mut Tape) {
        tape.weights.truncate(self.n_weights * 3);
        tape.data.truncate(self.n_params * 3);
        tape.inputs.truncate(0);
    }

    // TODO: Get rid of the one hot materialization
    fn cross_entropy(&mut self, tape: &mut Tape, logits: usize, target_ids: &[usize]) -> usize {
        let shape = tape.weights[logits].shape;
        let vocab_size = shape.get_dim(-1);
        let one_hot = tape.zeros(shape);
        let eps = tape.scalar(1e-9);

        for (i, token_id) in target_ids.iter().copied().enumerate() {
            let d_i = tape.weights[one_hot].offset(i * vocab_size + token_id);
            tape.data[d_i] = 1.0;
        }

        // TODO: Get rid of epsilon hack and implement log_softmax
        let mut x = tape.softmax(logits);
        x = tape.add(x, eps);
        x = tape.log(x);
        x = tape.mul(x, one_hot);
        x = tape.sum(x, None);
        x = tape.neg(x);

        let seq_len = target_ids.len() as f32;
        let n = tape.scalar(seq_len);
        tape.div(x, n)
    }

    pub fn causal_mask(tape: &mut Tape, seq_len: usize) -> usize {
        let shape = Shape::new(&[seq_len, seq_len]);
        let init = tape.scalar(f32::NEG_INFINITY);
        let out = tape.broadcast(init, shape.dims);
        let out = tape.materialize(out);

        for i in 0..shape.product() {
            let index = shape.unravel_index(i);
            let row = index.0[0];
            let col = index.0[1];

            if col <= row {
                let d_i = tape.weights[out].offset(i);
                tape.data[d_i] = 1.0;
            }
        }

        out
    }

    fn embedding(&mut self, tape: &mut Tape, tokenizer: &Tokenizer, token_ids: &[usize]) -> usize {
        let vocab_size = tokenizer.vocab.len() + 1; // Include BOS
        let seq_len = token_ids.len();
        let shape = Shape::new(&[seq_len, vocab_size]);
        let one_hot = tape.zeros(shape);

        for (i, token_id) in token_ids.iter().copied().enumerate() {
            let d_i = tape.weights[one_hot].offset(i * vocab_size + token_id);
            tape.data[d_i] = 1.0;
        }

        tape.matmul(one_hot, self.wte)
    }

    fn forward(&mut self, tape: &mut Tape, tokenizer: &Tokenizer, token_ids: &[usize]) -> usize {
        let head_dim = self.args.d_model / self.args.n_head;
        let seq_len = token_ids.len();
        let head_shape = Shape::new(&[1, self.args.n_head, seq_len, head_dim]);

        // Score standard deviation grows roughly with sqrt(head_dim)
        let head_dim_w = tape.scalar(head_dim as f32);
        let score_scale = tape.pow(head_dim_w, 0.5);
        let causal_mask = Self::causal_mask(tape, seq_len);
        let out_shape = Shape::new(&[seq_len, self.args.d_model]);

        let mut x = self.embedding(tape, tokenizer, token_ids);
        x = tape.rmsnorm(x);

        for layer in &mut self.layers {
            let x_residual = x.clone();
            x = tape.rmsnorm(x);

            let k = tape.matmul(x, layer.attn_wk);
            let k = tape.reshape(k, head_shape);

            let v = tape.matmul(x, layer.attn_wv);
            let v = tape.reshape(v, head_shape);

            let q = tape.matmul(x, layer.attn_wq);
            let q = tape.reshape(q, head_shape);

            let k_t = tape.transpose(k, -2, -1);
            let mut scores = tape.matmul(q, k_t);
            scores = tape.div(scores, score_scale);
            scores = tape.add(scores, causal_mask);
            scores = tape.softmax(scores);
            scores = tape.matmul(scores, v);
            scores = tape.transpose(scores, -3, -2);
            scores = tape.reshape(scores, out_shape);

            x = tape.matmul(scores, layer.attn_wo);
            x = tape.add(x, x_residual);

            // MLP block
            let x_residual = x.clone();
            x = tape.rmsnorm(x);

            // Project up to 4x the D_MODEL and RELU to make a non-linear transform
            x = tape.matmul(x, layer.mlp_fc1);
            x = tape.relu(x);

            // Project down and add residual
            x = tape.matmul(x, layer.mlp_fc2);
            x = tape.add(x, x_residual);
        }

        // Keep the spread small to play nice with softmax
        x = tape.rmsnorm(x);
        tape.matmul(x, self.lm_head)
    }

    pub fn train(&mut self, tape: &mut Tape, tokenizer: &mut Tokenizer, path: &str) {
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
        let mut token_ids = vec![];

        for step in 0..self.args.n_steps {
            let start = Instant::now();
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

            tokenizer.encode(&input, &mut token_ids);
            let input_ids = &token_ids[..token_ids.len() - 1];
            let target_ids = &token_ids[1..];

            let logits = self.forward(tape, tokenizer, &input_ids);
            let loss = self.cross_entropy(tape, logits, &target_ids);
            let loss_f = tape.data[tape.weights[loss].offset];

            tape.backward();

            // Linear learning rate decay
            let stepf = step as f32;
            let stepsf = self.args.n_steps as f32;
            let lr_t = tape.scalar(self.args.learning_rate * (1.0 - (stepf / stepsf)));
            let eps_adam = tape.scalar(self.args.eps_adam);
            let one = tape.scalar(1.0);

            let b1 = tape.scalar(self.args.beta_1);
            let b1_complement = tape.sub(one, b1);
            let m_pow = tape.pow(b1, stepf + 1.0);
            let m_numerator = tape.sub(one, m_pow);

            let b2 = tape.scalar(self.args.beta_2);
            let b2_complement = tape.sub(one, b2);
            let v_pow = tape.pow(b2, stepf + 1.0);
            let v_numerator = tape.sub(one, v_pow);

            for i in 0..self.n_weights {
                if let Some(grad) = tape.weights[i].grad {
                    let mi = self.n_weights + i;
                    let m1 = tape.mul(mi, b1);
                    let m2 = tape.mul(grad, b1_complement);
                    let m3 = tape.add(m1, m2);
                    tape.copy_data(m3, mi);

                    let grad_squared = tape.pow(grad, 2.0);
                    let vi = self.n_weights * 2 + i;
                    let v1 = tape.mul(vi, b2);
                    let v2 = tape.mul(grad_squared, b2_complement);
                    let v3 = tape.add(v1, v2);
                    tape.copy_data(v3, vi);

                    let m_hat = tape.div(mi, m_numerator);
                    let v_hat = tape.div(vi, v_numerator);

                    let v_hat_sqrt = tape.pow(v_hat, 0.5);
                    let denominator = tape.add(v_hat_sqrt, eps_adam);
                    let numerator = tape.mul(m_hat, lr_t);
                    let change = tape.div(numerator, denominator);
                    let neg_change = tape.neg(change);
                    let d = tape.add(i, neg_change);

                    tape.copy_data(d, i);
                    tape.weights[i].grad = None;
                }
            }

            let w = self.args.n_steps.to_string().len();
            let elapsed = start.elapsed();

            println!(
                "[{:02}.{:03}] Loss {:0>w$} / {}: {}",
                elapsed.as_secs(),
                elapsed.subsec_millis(),
                step,
                self.args.n_steps,
                loss_f,
                w = w
            );

            self.truncate(tape);
        }
    }

    pub fn infer(&mut self, tape: &mut Tape, tokenizer: &mut Tokenizer) {
        for _ in 0..self.args.n_samples {
            let mut rng = thread_rng();
            let mut token_ids = vec![tokenizer.bos];

            for _ in 0..self.args.n_context {
                let logits = self.forward(tape, tokenizer, &token_ids);
                let logits = tape.select(logits, 0, token_ids.len() - 1);
                let probs = tape.softmax(logits);

                let offset = tape.weights[probs].offset;
                let shape = tape.weights[probs].shape;
                let weights = &tape.data[offset..offset + shape.product()];
                let dist = WeightedIndex::new(weights).unwrap();
                let token_id = dist.sample(&mut rng);

                if token_id == tokenizer.bos {
                    break;
                }

                token_ids.push(token_id);
                self.truncate(tape);
            }

            let mut sample = vec![];
            for token_id in token_ids {
                if let Some(t) = tokenizer.vocab.get(token_id) {
                    sample.extend(t.bytes.iter().flatten().copied());
                };
            }

            let s = String::from_utf8(sample).expect("invalid utf-8");
            println!("Sample: {}", s);
        }
    }
}
