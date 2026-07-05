use rand::distributions::Uniform;
use rand::distributions::{Distribution, WeightedIndex};
use rand::thread_rng;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::ops::Add;
use std::{fs, vec};

const N_EMBED: usize = 16; // embedding dimension
const N_HEAD: usize = 4; // Number of attention heads
const N_LAYER: usize = 4; // number of layers
const BLOCK_SIZE: usize = 16; // Context window
const TRAINING_STEPS: usize = 1_000;
const N_SAMPLES: usize = 16;
const LEARNING_RATE: f32 = 0.01;
const BETA_1: f32 = 0.85;
const BETA_2: f32 = 0.99;
const EPS_ADAM: f32 = 1e-8;
const N_RULES: usize = 256;
const N_CHARS: usize = 6;
const MAX_RANK: usize = 4;
const HEAD_DIM: usize = N_EMBED / N_HEAD;

fn main() {
    let mut tokenizer = Tokenizer::new();
    tokenizer.train("input.txt");
    println!("Vocab: {}", tokenizer.vocab.len() + 1);

    let mut tape = Tape::new();
    let mut gpt = Gpt::new(&mut tape, &tokenizer);
    println!("Params: {}", gpt.size);

    gpt.train(&mut tape, &mut tokenizer, "input.txt");
    gpt.infer(&mut tape, &mut tokenizer);
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
        train: bool,
    ) {
        if let Some(positions) = self.pairs.get_mut(&from_key) {
            positions.remove(&old_pos);
        }

        self.pairs.entry(to_key).or_default().insert(new_pos);

        if train {
            let freq = self.pairs[&to_key].len();
            self.heap.push((freq, to_key));
        } else {
            if let Some(&rank) = self.rules.get(&to_key) {
                self.heap.push((self.rules.len() - rank, to_key));
            }
        }
    }

    fn merge_pair(&mut self, pair: (Token, Token), train: bool) {
        let mut rule_added = train == false;
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
                        self.rekey_position(from_key, to_key, left, left, train);
                    }

                    // Unlink end from the chain start -> end -> right
                    let right = self.next[end];
                    self.next[start] = right;
                    self.dead[end] = true;
                    if let Some(right) = right {
                        self.prev[right] = Some(start);

                        let from_key = (self.sequence[end], self.sequence[right]);
                        let to_key = (merged, self.sequence[right]);
                        self.rekey_position(from_key, to_key, end, start, train);
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

                self.merge_pair(pair, true);
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
            self.merge_pair(token, false);
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
            .rules
            .keys()
            .copied()
            .flat_map(|(a, b)| [a, b, a + b])
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

#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Pow { n: f32 },
    Log,
    Exp,
    Relu,
    Neg,
    Squeeze,
    MatMul,
    Mul,
    Select { dim: usize, i: usize },
    Sum { dim: Option<usize> },
    // Argmax is a tensor containing flat logical indices
    // to the first max value of each reduction
    Max { dim: Option<usize>, argmax: usize },
    Concat { dim: usize },
    Transpose { outer: isize, inner: isize },
    Reshape,
    Broadcast,
}

type Dims = [Option<usize>; MAX_RANK];

trait ShapeIndex {
    fn to_isize(self) -> isize;
}

impl ShapeIndex for usize {
    fn to_isize(self) -> isize {
        self as isize
    }
}

impl ShapeIndex for i32 {
    fn to_isize(self) -> isize {
        self as isize
    }
}

impl ShapeIndex for isize {
    fn to_isize(self) -> isize {
        self
    }
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    dims: Dims,
    strides: Dims,
}

// Shape strides should be used if there is not a need to recompute
fn get_strides(dims: Dims) -> Dims {
    let mut strides = Dims::default();
    let n = dims.iter().flatten().count();
    for i in 0..n {
        strides[i] = Some(dims[i + 1..].iter().flatten().product());
    }
    strides
}

fn dot_dims(a: Dims, b: Dims) -> usize {
    let mut dot = 0;
    for (i, v) in a.iter().flatten().enumerate() {
        dot += v * b[i].unwrap();
    }
    dot
}

fn add_dims(base_dims: Dims, new_dims: &[usize]) -> Dims {
    let mut out_dims = base_dims;
    let mut out_slots = out_dims.iter_mut().filter(|x| x.is_none());
    for &v in new_dims {
        match out_slots.next() {
            Some(slot) => *slot = Some(v),
            None => panic!("Failed to add dims"),
        }
    }
    out_dims
}

fn insert_dim(dims: Dims, dim: usize, i: usize) -> Dims {
    let mut out = dims;
    for j in (dim..dims.len() - 1).rev() {
        out[j + 1] = out[j];
    }
    out[dim] = Some(i);
    out
}

impl Shape {
    fn new(a: &[usize]) -> Self {
        let mut dims = Dims::default();
        for i in 0..a.len() {
            dims[i] = Some(a[i]);
        }

        let strides = get_strides(dims);
        Self { dims, strides }
    }

    fn from_dims(dims: Dims) -> Self {
        let strides = get_strides(dims);
        Self { dims, strides }
    }

    fn rank(&self) -> usize {
        self.dims.iter().flatten().count()
    }

    fn real_index(&self, i: isize) -> usize {
        if i >= 0 {
            i as usize
        } else {
            let r = self.rank() as isize;
            (r + i) as usize
        }
    }

    fn unravel_index(&self, i: usize) -> Dims {
        let mut index = Dims::default();
        let mut i = i.clone();
        for j in (0..self.dims.iter().flatten().count()).rev() {
            let dim = self.dims[j].unwrap();
            index[j] = Some(i % dim);
            i = i / dim;
        }
        index
    }

    fn expand(&self, n: usize) -> Self {
        let rank = self.rank();
        if rank >= n {
            self.clone()
        } else {
            let mut out = self.clone();
            let diff = n - rank;
            out.dims.copy_within(0..rank, diff);
            out.dims[..diff].fill(Some(1));
            out.strides.copy_within(0..rank, diff);
            out.strides[..diff].fill(Some(0));
            out
        }
    }

    fn get_dim<I: ShapeIndex>(&self, i: I) -> usize {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.dims[i].unwrap()
    }

    fn get_stride<I: ShapeIndex>(&self, i: I) -> usize {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.strides[i].unwrap()
    }

    fn squeeze<I: ShapeIndex>(&self, i: I) -> Self {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.drop(i)
    }

    fn broadcast(&self, other: Self) -> Dims {
        let n = self.rank().max(other.rank());

        let mut self_dims = self.dims.iter().flatten().copied().rev();
        let mut other_dims = other.dims.iter().flatten().copied().rev();
        let mut out_dims = Dims::default();
        for i in (0..n).rev() {
            let a = self_dims.next();
            let b = other_dims.next();
            out_dims[i] = a.max(b);
        }

        out_dims
    }

    fn batches(&self) -> Self {
        let r = self.rank();
        if r <= 2 {
            Shape::new(&[])
        } else {
            let mut out = self.clone();
            for i in r - 2..r {
                out.dims[i] = None;
                out.strides[i] = None;
            }
            out
        }
    }

    fn product(&self) -> usize {
        self.dims.iter().flatten().product()
    }

    fn drop(&self, i: usize) -> Self {
        let mut out = self.clone();
        out.dims[i..].rotate_left(1);
        out.strides[i..].rotate_left(1);
        out.dims[MAX_RANK - 1] = None;
        out.strides[MAX_RANK - 1] = None;
        out
    }
}

#[derive(Copy, Clone, Debug)]
struct View {
    offset: usize,
    len: usize,
}

impl View {
    fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

#[derive(Clone, Copy, Debug)]
struct Tensor {
    // View into tape inputs
    inputs: Option<View>,
    // Tape data position
    offset: usize,
    // Tensor dimensions
    shape: Shape,
    // Index of another tensor holding gradient values
    grad: Option<usize>,
    // Operation that produced this tensor used for VJP computation
    op: Option<Op>,
}

impl Tensor {
    fn new(offset: usize, shape: Shape, inputs: Option<View>, op: Option<Op>) -> Self {
        Self {
            shape,
            offset,
            op,
            inputs,
            grad: None,
        }
    }

    fn offset(&self, i: usize) -> usize {
        let index = self.shape.unravel_index(i);
        self.offset + dot_dims(index, self.shape.strides)
    }
}

struct Tape {
    values: Vec<Tensor>,
    data: Vec<f32>,
    inputs: Vec<usize>,
}

impl Tape {
    fn new() -> Self {
        Self {
            values: vec![],
            data: vec![],
            inputs: vec![],
        }
    }

    fn backward(&mut self) {
        let n = self.values.len();
        assert!(n > 0);

        let seed = Some(self.scalar(1.0));
        self.values[n - 1].grad = seed;

        for output in (0..n).rev() {
            if self.values[output].grad.is_none() {
                continue;
            };

            if let Some(inputs) = self.values[output].inputs {
                let offset = inputs.offset;
                for i in 0..inputs.len {
                    let input = self.inputs[offset + i];
                    let input_grad = self.vjp(input, output, i);

                    if let Some(curr) = self.values[input].grad {
                        let grad = Some(self.add(curr, input_grad));
                        self.values[input].grad = grad
                    } else {
                        self.values[input].grad = Some(input_grad)
                    }
                }
            }
        }
    }

    // Assumes (in, out) weights and uses Kaiming/He init
    fn random(&mut self, shape: Shape) -> usize {
        let offset = self.data.len();
        let in_dims = shape.dims[0].unwrap();
        let k = (6.0 / in_dims as f32).sqrt();
        let dist = Uniform::new(-k, k);
        let len = shape.product();

        let mut rng = rand::thread_rng();
        for _ in 0..len {
            self.data.push(dist.sample(&mut rng));
        }

        self.push(Tensor::new(offset, shape, None, None))
    }

    fn select(&mut self, a: usize, dim: usize, i: usize) -> usize {
        let op = Some(Op::Select { dim, i });
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        // Dropping dim from in_shape produces the output shape
        let in_shape = self.values[a].shape;
        let out_shape = in_shape.drop(dim);

        let in_offset = self.values[a].offset;
        let dim_offset = in_shape.get_stride(dim) * i;
        let data_offset = in_offset + dim_offset;
        self.push(Tensor::new(data_offset, out_shape, inputs, op))
    }

    fn scalar(&mut self, value: f32) -> usize {
        let offset = self.data.len();
        let shape = Shape::new(&[]);

        self.data.push(value);
        self.push(Tensor::new(offset, shape, None, None))
    }

    // Chain rule gives dL/dx = dL/dy dy/dx and dL/dy = output.grad
    fn vjp(&mut self, input: usize, output: usize, slot: usize) -> usize {
        let op = self.values[output].op.unwrap();
        let v = self.values[output].grad.unwrap();

        match op {
            Op::Reshape => {
                let input_shape = self.values[input].shape;
                self.reshape(v, input_shape)
            }
            Op::Transpose { outer, inner } => self.transpose(v, inner, outer),
            Op::Broadcast => {
                let output_shape = self.values[output].shape;
                let mut input_shape = self.values[input].shape;
                input_shape = input_shape.expand(output_shape.rank());

                let mut x = v;
                for (i, dim) in output_shape.dims.iter().copied().enumerate() {
                    if dim != input_shape.dims[i] {
                        x = self.sum(x, Some(i));
                    };
                }

                self.reshape(x, input_shape)
            }
            Op::Select { dim, i } => {
                let v_shape = self.values[v].shape;
                let input_shape = self.values[input].shape;
                let zero = self.scalar(0.0);
                let x = self.reshape(zero, input_shape);
                let x_strides = get_strides(input_shape.dims);

                for v_k in 0..v_shape.product() {
                    // Select sliced the input along dim at index i and
                    // removed dim from the output shape. To get the logical
                    // index on the jacobian we need to step along the slice
                    // on the input shape.
                    let v_index = v_shape.unravel_index(v_k);
                    let x_index = insert_dim(v_index, dim, i);
                    let x_k = dot_dims(x_index, x_strides);

                    let v_i = self.values[v].offset(v_k);
                    let x_i = self.values[x].offset + x_k;
                    self.data[x_i] = self.data[v_i];
                }

                x
            }
            Op::Concat { dim } => {
                let input_shape = self.values[input].shape;
                let input_offset = self.values[output].inputs.unwrap().offset;
                let v_shape = self.values[v].shape;
                let v_strides = get_strides(v_shape.dims);

                let mut slot_start = 0;
                for k in input_offset..input_offset + slot {
                    let k_shape = self.values[self.inputs[k]].shape;
                    slot_start += k_shape.get_dim(dim);
                }

                let zero = self.scalar(0.0);
                let x = self.reshape(zero, input_shape);
                for x_k in 0..input_shape.product() {
                    let mut v_index = input_shape.unravel_index(x_k);
                    *v_index[dim].as_mut().unwrap() += slot_start;
                    let v_k = dot_dims(v_index, v_strides);

                    let v_i = self.values[v].offset(v_k);
                    let x_i = self.values[x].offset + x_k;
                    self.data[x_i] = self.data[v_i];
                }

                x
            }
            Op::Squeeze => {
                let input_shape = self.values[input].shape;
                self.reshape(v, input_shape)
            }
            Op::MatMul => {
                let input_offset = self.values[output].inputs.unwrap().offset;

                if slot == 0 {
                    let b = self.inputs[input_offset + 1];
                    let b_t = self.transpose(b, -2, -1);
                    self.matmul(v, b_t)
                } else {
                    let a = self.inputs[input_offset];
                    let a_t = self.transpose(a, -2, -1);
                    self.matmul(a_t, v)
                }
            }
            // Gradient passed to the first occurance of
            // max for each reduction by convention.
            Op::Max { argmax, .. } => {
                let zero = self.scalar(0.0);
                let input_shape = self.values[input].shape;
                let j = self.reshape(zero, input_shape);

                let argmax_offset = self.values[argmax].offset;
                let argmax_shape = self.values[argmax].shape;
                for k in 0..argmax_shape.product() {
                    let k_i = self.data[argmax_offset + k] as usize;
                    let d_i = self.values[j].offset(k_i);
                    self.data[d_i] = 1.0;
                }

                self.mul(v, j)
            }
            Op::Neg => {
                let j = self.scalar(-1.0);
                self.mul(v, j)
            }
            Op::Add => {
                let j = self.scalar(1.0);
                self.mul(v, j)
            }
            Op::Exp => {
                let j = self.exp(input);
                self.mul(v, j)
            }
            Op::Log => {
                let j = self.pow(input, -1.0);
                self.mul(v, j)
            }
            Op::Pow { n } => {
                let s = self.scalar(n);
                let p = self.pow(input, n - 1 as f32);
                let j = self.mul(s, p);
                self.mul(v, j)
            }
            Op::Mul => {
                let j = if slot == 0 { input + 1 } else { input - 1 };
                self.mul(v, j)
            }
            Op::Sum { dim: _ } => {
                let ones = self.scalar(1.0);
                let j = self.reshape(ones, self.values[input].shape);
                self.mul(v, j)
            }
            Op::Relu => {
                let input_shape = self.values[input].shape;
                let zero = self.scalar(0.0);
                let j = self.reshape(zero, input_shape);

                for k in 0..input_shape.product() {
                    let k_i = self.values[input].offset(k);
                    if self.data[k_i] > 0.0 {
                        let x_i = self.values[j].offset(k);
                        self.data[x_i] = 1.0;
                    }
                }

                self.mul(v, j)
            }
        }
    }

    fn broadcast(&mut self, a: usize, dims: Dims) -> usize {
        let op = Some(Op::Broadcast);
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let in_shape = self.values[a].shape;
        let in_dims = in_shape.dims;
        let in_rank = in_shape.rank();

        let mut strides = in_shape.strides;
        for i in 0..in_rank {
            if in_dims[i] == Some(1) {
                strides[i] = Some(0);
            }
        }

        let out_rank = dims.iter().flatten().count();
        let rank_diff = out_rank - in_rank;
        if rank_diff > 0 {
            strides.copy_within(0..in_rank, rank_diff);
            strides[..rank_diff].fill(Some(0));
        }

        let out_shape = Shape { dims, strides };
        let offset = self.values[a].offset;
        self.push(Tensor::new(offset, out_shape, inputs, op))
    }

    fn reshape(&mut self, a: usize, to_shape: Shape) -> usize {
        let data_offset = self.data.len();
        let input_offset = self.inputs.len();
        let inputs = Some(View::new(input_offset, 1));
        self.inputs.push(a);

        for i in 0..to_shape.product() {
            let a_i = self.values[a].offset(i);
            self.data.push(self.data[a_i]);
        }

        let op = Some(Op::Reshape);
        let shape = Shape::from_dims(to_shape.dims);
        self.push(Tensor::new(data_offset, shape, inputs, op))
    }

    fn transpose(&mut self, a: usize, outer: isize, inner: isize) -> usize {
        let op = Some(Op::Transpose { outer, inner });
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let offset = self.values[a].offset;
        let in_shape = self.values[a].shape;
        let mut out_shape = in_shape.clone();

        let inner = in_shape.real_index(inner);
        let outer = in_shape.real_index(outer);
        out_shape.dims[inner] = in_shape.dims[outer];
        out_shape.dims[outer] = in_shape.dims[inner];
        out_shape.strides[inner] = in_shape.strides[outer];
        out_shape.strides[outer] = in_shape.strides[inner];

        self.push(Tensor::new(offset, out_shape, inputs, op))
    }

    // Assumes all inputs have the same shape outside of dim
    fn concat(&mut self, a: &[usize], dim: usize) -> usize {
        let op = Some(Op::Concat { dim });
        let data_offset = self.data.len();
        let inputs_offset = self.inputs.len();
        let inputs_len = a.len();
        let inputs = Some(View::new(inputs_offset, inputs_len));
        self.inputs.extend_from_slice(a);

        let outer_dims = &self.values[a[0]].shape.dims[..dim];
        let outer_len = outer_dims.iter().flatten().product();
        for i in 0..outer_len {
            for b in a.iter().copied() {
                let inner_dims = &self.values[b].shape.dims[dim..];
                let inner_len = inner_dims.iter().flatten().product();

                for k in 0..inner_len {
                    let l_i = i * inner_len + k;
                    let d_i = self.values[b].offset(l_i);
                    self.data.push(self.data[d_i]);
                }
            }
        }

        let out_dim = Some(a.iter().copied().fold(0, |acc, next| {
            let x_dim = self.values[next].shape.get_dim(dim);
            acc + x_dim
        }));

        let mut out_shape = self.values[a[0]].shape.clone();
        out_shape.dims[dim] = out_dim;
        out_shape.strides = get_strides(out_shape.dims);

        self.push(Tensor::new(data_offset, out_shape, inputs, op))
    }

    // Keeps the dimension if provided
    fn reduce<F: Fn(f32, f32) -> f32>(&mut self, a: usize, op: Op, f: F, init: f32) -> usize {
        let data_offset = self.data.len();
        let a_shape = self.values[a].shape;

        let dim = match op {
            Op::Sum { dim } | Op::Max { dim, .. } => dim,
            _ => unreachable!("Reduce called with incorrect op"),
        };

        let inner_len = match dim {
            None => 1,
            Some(dim) => {
                let inner_dims = &a_shape.dims[dim + 1..];
                inner_dims.iter().flatten().product()
            }
        };

        let dim_len = match dim {
            None => a_shape.product(),
            Some(dim) => a_shape.get_dim(dim),
        };

        let outer_len = match dim {
            None => 1,
            Some(dim) => {
                let outer_dims = &a_shape.dims[..dim];
                outer_dims.iter().flatten().product()
            }
        };

        for i in 0..outer_len {
            for k in 0..inner_len {
                let mut acc = init;
                for j in 0..dim_len {
                    let l_i = (i * dim_len + j) * inner_len + k;
                    let d_i = self.values[a].offset(l_i);
                    acc = f(acc, self.data[d_i]);
                }
                self.data.push(acc);
            }
        }

        let shape = match dim {
            None => Shape::new(&[]),
            Some(dim) => {
                let mut dims = a_shape.dims.clone();
                dims[dim] = Some(1);
                Shape::from_dims(dims)
            }
        };

        let inputs = Some(self.push_inputs(&[a]));
        self.push(Tensor::new(data_offset, shape, inputs, Some(op)))
    }

    fn sum(&mut self, a: usize, dim: Option<usize>) -> usize {
        let f = |acc: f32, next: f32| acc + next;
        self.reduce(a, Op::Sum { dim }, f, 0.0)
    }

    fn max(&mut self, a: usize, dim: Option<usize>) -> usize {
        let f = |acc: f32, next: f32| acc.max(next);
        let zero = self.scalar(0.0); // TODO: Real argmax
        self.reduce(a, Op::Max { dim, argmax: zero }, f, f32::NEG_INFINITY)
    }

    fn rmsnorm(&mut self, a: usize) -> usize {
        let a_shape = self.values[a].shape;
        let dim = a_shape.rank() - 1;
        let n = self.scalar(a_shape.get_dim(dim) as f32);

        let epsilon = self.scalar(1e-5);
        let pow = self.pow(a, 2.0);
        let sum = self.sum(pow, Some(dim));
        let mean = self.div(sum, n);
        let mean = self.add(mean, epsilon);
        let scale = self.pow(mean, -0.5);

        self.mul(a, scale)
    }

    fn softmax(&mut self, a: usize) -> usize {
        let a_shape = self.values[a].shape;
        let dim = a_shape.rank() - 1;
        let max = self.max(a, Some(dim));
        let sub = self.sub(a, max);
        let exps = self.exp(sub);
        let sum = self.sum(exps, Some(dim));
        self.div(exps, sum)
    }

    fn copy_data(&mut self, from: usize, to: usize) {
        let from = self.values[from];
        let to = self.values[to];

        let n = from.shape.product();
        assert!(n == to.shape.product());

        let src = from.offset..from.offset + n;
        self.data.copy_within(src, to.offset);
    }

    fn push(&mut self, v: Tensor) -> usize {
        self.values.push(v);
        self.values.len() - 1
    }

    fn push_inputs(&mut self, inputs: &[usize]) -> View {
        let offset = self.inputs.len();
        let len = inputs.iter().len();

        for &input in inputs {
            self.inputs.push(input);
        }

        View::new(offset, len)
    }

    fn bmm(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.values[a].shape;
        let b_shape = self.values[b].shape;

        let m = a_shape.get_dim(-2);
        let n = b_shape.get_dim(-1);
        let k = a_shape.get_dim(-1);

        let a_batches = a_shape.batches();
        let b_batches = b_shape.batches();
        let out_batches = a_batches.broadcast(b_batches);
        let batch_count = out_batches.iter().flatten().product();

        let a_target = add_dims(out_batches, &[m, k]);
        let b_target = add_dims(out_batches, &[k, n]);
        let a = self.broadcast(a, a_target);
        let b = self.broadcast(b, b_target);

        let data_offset = self.data.len();
        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        for batch in 0..batch_count {
            let a_offset = batch * m * k;
            let b_offset = batch * k * n;

            for i in 0..m * n {
                let row = i / n;
                let col = i % n;

                let mut dot = 0.0;
                for j in 0..k {
                    let a_i = a_offset + row * k + j;
                    let a_i = self.values[a].offset(a_i);
                    let a_data = self.data[a_i];

                    let b_i = b_offset + col + n * j;
                    let b_i = self.values[b].offset(b_i);
                    let b_data = self.data[b_i];

                    dot += a_data * b_data;
                }

                self.data.push(dot);
            }
        }

        let dims = add_dims(out_batches, &[m, n]);
        let strides = get_strides(dims);
        let out_shape = Shape { dims, strides };
        let op = Some(Op::MatMul);
        self.push(Tensor::new(data_offset, out_shape, inputs, op))
    }

    // https://docs.pytorch.org/docs/2.13/generated/torch.matmul.html
    fn matmul(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.values[a].shape;
        let b_shape = self.values[b].shape;
        let a_rank = a_shape.rank();
        let b_rank = b_shape.rank();

        match (a_rank, b_rank) {
            (1, 1) => {
                let inputs = Some(View::new(self.inputs.len(), 2));
                let data_offset = self.data.len();
                self.inputs.push(a);
                self.inputs.push(b);

                let mut dot = 0.0;
                for i in 0..a_shape.get_dim(0) {
                    let a_i = self.values[a].offset(i);
                    let b_i = self.values[b].offset(i);

                    dot += self.data[a_i] * self.data[b_i];
                }

                self.data.push(dot);

                let out_shape = Shape::new(&[]);
                let op = Some(Op::MatMul);
                self.push(Tensor::new(data_offset, out_shape, inputs, op))
            }
            (1, _) => {
                let k = a_shape.get_dim(0);
                let c = self.reshape(a, Shape::new(&[1, k]));
                let d = self.bmm(c, b);
                self.squeeze(d, -2)
            }
            (_, 1) => {
                let k = b_shape.get_dim(0);
                let c = self.reshape(b, Shape::new(&[k, 1]));
                let d = self.bmm(a, c);
                self.squeeze(d, -1)
            }
            _ => self.bmm(a, b),
        }
    }

    fn squeeze(&mut self, a: usize, dim: isize) -> usize {
        let a_shape = self.values[a].shape;
        let out_shape = a_shape.squeeze(dim);
        let offset = self.values[a].offset;
        let inputs = Some(self.push_inputs(&[a]));
        let op = Some(Op::Squeeze);

        self.push(Tensor::new(offset, out_shape, inputs, op))
    }

    fn add(&mut self, a: usize, b: usize) -> usize {
        let op = Some(Op::Add);

        let a_shape = self.values[a].shape;
        let b_shape = self.values[b].shape;
        let in_dims = a_shape.broadcast(b_shape);
        let a = self.broadcast(a, in_dims);
        let b = self.broadcast(b, in_dims);

        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        let data_offset = self.data.len();
        let data_len = in_dims.iter().flatten().product();

        for i in 0..data_len {
            let a_i = self.values[a].offset(i);
            let b_i = self.values[b].offset(i);
            let s = self.data[a_i] + self.data[b_i];
            self.data.push(s);
        }

        let dims = self.values[a].shape.dims;
        let strides = get_strides(dims);
        let shape = Shape { dims, strides };
        self.push(Tensor::new(data_offset, shape, inputs, op))
    }

    fn sub(&mut self, a: usize, b: usize) -> usize {
        let c = self.neg(b);
        self.add(a, c)
    }

    fn elementwise<F: Fn(f32) -> f32>(&mut self, a: usize, op: Op, f: F) -> usize {
        let op = Some(op);
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let mut shape = self.values[a].shape;
        let data_offset = self.data.len();

        for i in 0..shape.product() {
            let d_i = self.values[a].offset(i);
            let v = f(self.data[d_i]);
            self.data.push(v);
        }

        shape.strides = get_strides(shape.dims);
        self.push(Tensor::new(data_offset, shape, inputs, op))
    }

    fn neg(&mut self, i: usize) -> usize {
        let f = |x: f32| -x;
        self.elementwise(i, Op::Neg, f)
    }

    fn pow(&mut self, i: usize, n: f32) -> usize {
        let f = |x: f32| x.powf(n);
        self.elementwise(i, Op::Pow { n }, f)
    }

    fn log(&mut self, i: usize) -> usize {
        let f = |x: f32| x.ln();
        self.elementwise(i, Op::Log, f)
    }

    fn exp(&mut self, i: usize) -> usize {
        let f = |x: f32| x.exp();
        self.elementwise(i, Op::Exp, f)
    }

    fn relu(&mut self, i: usize) -> usize {
        let f = |x: f32| x.max(0.0);
        self.elementwise(i, Op::Relu, f)
    }

    fn mul(&mut self, a: usize, b: usize) -> usize {
        let op = Some(Op::Mul);

        let a_shape = self.values[a].shape;
        let b_shape = self.values[b].shape;
        let in_dims = a_shape.broadcast(b_shape);
        let a = self.broadcast(a, in_dims);
        let b = self.broadcast(b, in_dims);

        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        let data_offset = self.data.len();
        let data_len = in_dims.iter().flatten().product();
        for i in 0..data_len {
            let a_i = self.values[a].offset(i);
            let b_i = self.values[b].offset(i);
            let p = self.data[a_i] * self.data[b_i];
            self.data.push(p);
        }

        let dims = self.values[a].shape.dims;
        let strides = get_strides(dims);
        let shape = Shape { dims, strides };
        self.push(Tensor::new(data_offset, shape, inputs, op))
    }

    fn div(&mut self, i: usize, j: usize) -> usize {
        let k = self.pow(j, -1.0);
        self.mul(i, k)
    }
}

struct Cache {
    keys: Vec<usize>,
    values: Vec<usize>,
}

struct Layer {
    attn_wq: usize,
    attn_wk: usize,
    attn_wv: usize,
    attn_wo: usize,
    mlp_fc1: usize,
    mlp_fc2: usize,
    cache: Cache,
}

impl Layer {
    fn new(tape: &mut Tape, n_embed: usize) -> Self {
        let attn_shape = Shape::new(&[n_embed, n_embed]);
        let attn_wq = tape.random(attn_shape);
        let attn_wk = tape.random(attn_shape);
        let attn_wv = tape.random(attn_shape);
        let attn_wo = tape.random(attn_shape);
        let mlp_fc1 = tape.random(Shape::new(&[n_embed, n_embed * 4]));
        let mlp_fc2 = tape.random(Shape::new(&[n_embed * 4, n_embed]));

        let cache = Cache {
            keys: vec![],
            values: vec![],
        };

        Self {
            attn_wq,
            attn_wk,
            attn_wv,
            attn_wo,
            mlp_fc1,
            mlp_fc2,
            cache,
        }
    }
}

struct Gpt {
    wte: usize,
    wpe: usize,
    lm_head: usize,
    layers: Vec<Layer>,
    params: usize,
    size: usize,
}

impl Gpt {
    fn new(tape: &mut Tape, tokenizer: &Tokenizer) -> Self {
        let vocab_size = tokenizer.vocab.len() + 1;
        let wte = tape.random(Shape::new(&[vocab_size, N_EMBED]));
        let wpe = tape.random(Shape::new(&[BLOCK_SIZE, N_EMBED]));
        let lm_head = tape.random(Shape::new(&[N_EMBED, vocab_size]));
        let layers = Vec::from_iter((0..N_LAYER).map(|_| Layer::new(tape, N_EMBED)));
        let params = tape.values.len();
        let size = tape.data.len();

        // Add moment buffers
        let n = tape.values.len();
        for _ in 0..2 {
            for i in 0..n {
                let data_offset = tape.data.len();
                let shape = tape.values[i].shape;

                for _ in 0..shape.product() {
                    tape.data.push(0.0);
                }

                tape.push(Tensor::new(data_offset, shape, None, None));
            }
        }

        Self {
            wte,
            wpe,
            lm_head,
            layers,
            params,
            size,
        }
    }

    fn truncate(&mut self, tape: &mut Tape) {
        // Times 3 to preserve moment buffers
        tape.values.truncate(self.params * 3);
        tape.data.truncate(self.size * 3);
        tape.inputs.truncate(0);

        for layer in &mut self.layers {
            layer.cache.keys.clear();
            layer.cache.values.clear();
        }
    }

    fn forward(&mut self, tape: &mut Tape, token_id: usize, pos_id: usize) -> usize {
        // Score standard deviation grows roughly with sqrt(head_dim)
        let head_dim = tape.scalar(HEAD_DIM as f32);
        let score_scale = tape.pow(head_dim, 0.5);

        // Combined token and position embedding.
        let wte = tape.select(self.wte, 0, token_id);
        let wpe = tape.select(self.wpe, 0, pos_id);
        let mut x = tape.add(wte, wpe);
        x = tape.rmsnorm(x);

        for layer in &mut self.layers {
            let x_residual = x.clone();
            x = tape.rmsnorm(x);

            // Cache the key and value embeddings.
            let k_new = tape.matmul(x, layer.attn_wk);
            let v_new = tape.matmul(x, layer.attn_wv);
            layer.cache.keys.push(k_new);
            layer.cache.values.push(v_new);

            let n_keys = layer.cache.keys.len();
            let kv_shape = Shape::new(&[n_keys, N_HEAD, HEAD_DIM]);

            let k = tape.concat(&layer.cache.keys, 0);
            let k = tape.reshape(k, kv_shape);
            let k = tape.transpose(k, 0, 1);

            let v = tape.concat(&layer.cache.values, 0);
            let v = tape.reshape(v, kv_shape);
            let v = tape.transpose(v, 0, 1);

            let q = tape.matmul(x, layer.attn_wq);
            let q = tape.reshape(q, Shape::new(&[N_HEAD, HEAD_DIM]));

            let mut x_attn = [0; N_HEAD];
            for h in 0..N_HEAD {
                let q_h = tape.select(q, 0, h);
                let k_h = tape.select(k, 0, h);
                let k_h = tape.transpose(k_h, 0, 1);

                let scores = tape.matmul(q_h, k_h);
                let scores = tape.div(scores, score_scale);
                let scores = tape.softmax(scores);

                let v_h = tape.select(v, 0, h);
                x_attn[h] = tape.matmul(scores, v_h);
            }

            // Reproject heads into one space and add residual
            x = tape.concat(&x_attn, 0);
            x = tape.matmul(x, layer.attn_wo);
            x = tape.add(x, x_residual);

            // MLP block
            let x_residual = x.clone();
            x = tape.rmsnorm(x);

            // Project up to 4x the N_EMBED and RELU to make a non-linear transform
            x = tape.matmul(x, layer.mlp_fc1);
            x = tape.relu(x);

            // Project down and add residual
            x = tape.matmul(x, layer.mlp_fc2);
            x = tape.add(x, x_residual);
        }

        // Output logits
        tape.matmul(x, self.lm_head)
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

            let mut sum = tape.scalar(0.0);
            let mut n = 0;
            for (pos_id, pair) in encoded.windows(2).enumerate() {
                let token_id = pair[0];
                let target_id = pair[1];

                let logits = self.forward(tape, token_id, pos_id);
                let probs = tape.softmax(logits);
                let eps = tape.scalar(1e-9);
                let prob = tape.select(probs, 0, target_id);
                let prob = tape.add(prob, eps); // TODO: Get rid of epsilon hack
                let loss_t = tape.log(prob);
                let loss = tape.neg(loss_t);

                sum = tape.add(sum, loss);
                n += 1;

                if target_id == tokenizer.bos {
                    break;
                }
            }

            let inv_n = tape.scalar(1.0 / n as f32);
            let loss = tape.mul(sum, inv_n);
            let loss_f = tape.data[tape.values[loss].offset];

            let w = TRAINING_STEPS.to_string().len();
            println!("Loss {:0>w$} / {}: {}", step, TRAINING_STEPS, loss_f, w = w);

            tape.backward();

            // Linear learning rate decay
            let stepf = step as f32;
            let stepsf = TRAINING_STEPS as f32;
            let lr_t = tape.scalar(LEARNING_RATE * (1.0 - (stepf / stepsf)));
            let eps_adam = tape.scalar(EPS_ADAM);
            let one = tape.scalar(1.0);

            let b1 = tape.scalar(BETA_1);
            let b1_complement = tape.sub(one, b1);
            let m_pow = tape.pow(b1, stepf + 1.0);
            let m_numerator = tape.sub(one, m_pow);

            let b2 = tape.scalar(BETA_2);
            let b2_complement = tape.sub(one, b2);
            let v_pow = tape.pow(b2, stepf + 1.0);
            let v_numerator = tape.sub(one, v_pow);

            for i in 0..self.params {
                if let Some(grad) = tape.values[i].grad {
                    let mi = self.params + i;
                    let m1 = tape.mul(mi, b1);
                    let m2 = tape.mul(grad, b1_complement);
                    let m3 = tape.add(m1, m2);
                    tape.copy_data(m3, mi);

                    let grad_squared = tape.pow(grad, 2.0);
                    let vi = self.params * 2 + i;
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
                    tape.values[i].grad = None;
                }
            }

            self.truncate(tape);
        }
    }

    fn infer(&mut self, tape: &mut Tape, tokenizer: &mut Tokenizer) {
        for _ in 0..N_SAMPLES {
            let mut rng = thread_rng();
            let mut token_id = tokenizer.bos;
            let mut sample = vec![];

            for pos_id in 0..BLOCK_SIZE {
                let logits = self.forward(tape, token_id, pos_id);
                let probs = tape.softmax(logits);
                let offset = tape.values[probs].offset;
                let shape = tape.values[probs].shape;
                let weights = &tape.data[offset..offset + shape.product()];
                let dist = WeightedIndex::new(weights).unwrap();
                token_id = dist.sample(&mut rng);

                if token_id == tokenizer.bos {
                    break;
                }

                let t = tokenizer.vocab[token_id];
                sample.extend(t.bytes.iter().flatten().copied());
            }

            self.truncate(tape);

            let s = String::from_utf8(sample).expect("invalid utf-8");
            println!("Sample: {}", s);
        }
    }
}
