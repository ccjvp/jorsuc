use crate::dims::Dims;
use crate::shape::Shape;
use rand::distributions::Distribution;
use rand::distributions::Uniform;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
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
    _Concat { dim: usize },
    Transpose { outer: isize, inner: isize },
    Reshape,
    Broadcast,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct View {
    offset: usize,
    len: usize,
}

impl View {
    fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    // View into tape inputs
    pub inputs: Option<View>,
    // Tape data position
    pub offset: usize,
    // Tensor dimensions
    pub shape: Shape,
    // Index of another tensor holding gradient values
    pub grad: Option<usize>,
    // Operation that produced this tensor used for VJP computation
    pub op: Option<Op>,
    // Wheter data is laid out contiguously
    pub is_contiguous: bool,
}

impl Tensor {
    pub fn new(
        offset: usize,
        shape: Shape,
        inputs: Option<View>,
        op: Option<Op>,
        is_contiguous: bool,
    ) -> Self {
        Self {
            shape,
            offset,
            op,
            inputs,
            is_contiguous,
            grad: None,
        }
    }

    pub fn offset(self, i: usize) -> usize {
        if self.is_contiguous {
            self.offset + i
        } else {
            let index = self.shape.unravel_index(i);
            self.offset + index.dot(self.shape.strides)
        }
    }

    pub fn nditer(self) -> impl Iterator<Item = usize> {
        self.shape.nditer(self.offset)
    }
}

pub struct Tape {
    pub weights: Vec<Tensor>,
    pub data: Vec<f32>,
    pub inputs: Vec<usize>,
}

// TODO: Numerical tests using generated data
impl Tape {
    pub fn new() -> Self {
        Self {
            weights: vec![],
            data: vec![],
            inputs: vec![],
        }
    }

    pub fn backward(&mut self) {
        let n = self.weights.len();
        assert!(n > 0);

        let seed = Some(self.scalar(1.0));
        self.weights[n - 1].grad = seed;

        for output in (0..n).rev() {
            if self.weights[output].grad.is_none() {
                continue;
            };

            if let Some(inputs) = self.weights[output].inputs {
                let offset = inputs.offset;
                for i in 0..inputs.len {
                    let input = self.inputs[offset + i];
                    let input_grad = self.vjp(input, output, i);

                    if let Some(curr) = self.weights[input].grad {
                        let grad = Some(self.add(curr, input_grad));
                        self.weights[input].grad = grad
                    } else {
                        self.weights[input].grad = Some(input_grad)
                    }
                }
            }
        }
    }

    // Assumes (in, out) weights and uses Kaiming/He init
    pub fn random(&mut self, shape: &[usize], scale: f32) -> usize {
        let shape = Shape::new(shape);
        let offset = self.data.len();
        let in_dims = shape.dims.0[0];
        let k = (6.0 / in_dims as f32).sqrt();
        let dist = Uniform::new(-k, k);
        let len = shape.product();

        let mut rng = rand::thread_rng();
        for _ in 0..len {
            self.data.push(dist.sample(&mut rng) * scale);
        }

        self.push_weight(offset, shape, None, None, true)
    }

    pub fn select(&mut self, a: usize, dim: usize, i: usize) -> usize {
        let op = Some(Op::Select { dim, i });
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        // Dropping dim from in_shape produces the output shape
        let in_shape = self.weights[a].shape;
        let out_shape = in_shape.drop(dim);

        let in_offset = self.weights[a].offset;
        let dim_offset = in_shape.get_stride(dim) * i;
        let data_offset = in_offset + dim_offset;
        self.push_weight(data_offset, out_shape, inputs, op, false)
    }

    pub fn scalar(&mut self, value: f32) -> usize {
        let offset = self.data.len();
        let shape = Shape::new(&[]);

        self.data.push(value);
        self.push_weight(offset, shape, None, None, true)
    }

    pub fn zeros(&mut self, shape: Shape) -> usize {
        let zero = self.scalar(0.0);
        let zeros = self.broadcast(zero, shape.dims);

        self.materialize(zeros)
    }

    pub fn ones(&mut self, shape: Shape) -> usize {
        let one = self.scalar(1.0);
        let ones = self.broadcast(one, shape.dims);

        self.materialize(ones)
    }

    #[cfg(test)]
    pub fn tensor(&mut self, data: &[f32]) -> usize {
        let n = data.len();
        assert!(n > 0);

        let offset = self.data.len();
        let shape = Shape::new(&[n]);
        assert!(n == 0 || shape.product() == n);

        self.data.extend_from_slice(data);
        self.push_weight(offset, shape, None, None, true)
    }

    // Chain rule gives dL/dx = dL/dy dy/dx and dL/dy = output.grad
    fn vjp(&mut self, input: usize, output: usize, slot: usize) -> usize {
        let op = self.weights[output].op.unwrap();
        let v = self.weights[output].grad.unwrap();

        match op {
            Op::Reshape => {
                let input_shape = self.weights[input].shape;
                self.reshape(v, input_shape)
            }
            Op::Transpose { outer, inner } => self.transpose(v, inner, outer),
            Op::Broadcast => {
                let output_shape = self.weights[output].shape;
                let output_rank = output_shape.rank();
                let input_shape = self.weights[input].shape;
                let expanded_input = input_shape.expand(output_rank);

                let mut x = v;
                for i in 0..output_rank {
                    if output_shape.get_dim(i) != expanded_input.get_dim(i) {
                        x = self.sum(x, Some(i));
                    }
                }

                self.reshape(x, input_shape)
            }
            Op::Select { dim, i } => {
                let v_shape = self.weights[v].shape;
                let input_shape = self.weights[input].shape;
                let x = self.zeros(input_shape);
                let x_strides = input_shape.dims.strides();

                for v_k in 0..v_shape.product() {
                    // Select sliced the input along dim at index i and
                    // removed dim from the output shape. To get the logical
                    // index on the jacobian we need to step along the slice
                    // on the input shape.
                    let v_index = v_shape.unravel_index(v_k);
                    let x_index = v_index.insert(dim, i);
                    let x_k = x_index.dot(x_strides);

                    let v_i = self.weights[v].offset(v_k);
                    let x_i = self.weights[x].offset + x_k;
                    self.data[x_i] = self.data[v_i];
                }

                x
            }
            Op::_Concat { dim } => {
                let input_shape = self.weights[input].shape;
                let input_offset = self.weights[output].inputs.unwrap().offset;
                let v_shape = self.weights[v].shape;
                let v_strides = v_shape.dims.strides();

                let mut slot_start = 0;
                for k in input_offset..input_offset + slot {
                    let k_shape = self.weights[self.inputs[k]].shape;
                    slot_start += k_shape.get_dim(dim);
                }

                let x = self.zeros(input_shape);
                for x_k in 0..input_shape.product() {
                    let mut v_index = input_shape.unravel_index(x_k);
                    v_index.0[dim] += slot_start;
                    let v_k = v_index.dot(v_strides);

                    let v_i = self.weights[v].offset(v_k);
                    let x_i = self.weights[x].offset + x_k;
                    self.data[x_i] = self.data[v_i];
                }

                x
            }
            Op::Squeeze => {
                let input_shape = self.weights[input].shape;
                self.reshape(v, input_shape)
            }
            Op::MatMul => {
                let input_offset = self.weights[output].inputs.unwrap().offset;

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
                let input_shape = self.weights[input].shape;
                let j = self.zeros(input_shape);

                let argmax_offset = self.weights[argmax].offset;
                let argmax_shape = self.weights[argmax].shape;
                for k in 0..argmax_shape.product() {
                    let k_i = self.data[argmax_offset + k] as usize;
                    let d_i = self.weights[j].offset(k_i);
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
                let shape = self.weights[input].shape;
                let j = self.ones(shape);
                self.mul(v, j)
            }
            Op::Relu => {
                let input_shape = self.weights[input].shape;
                let j = self.zeros(input_shape);

                for k in 0..input_shape.product() {
                    let k_i = self.weights[input].offset(k);
                    if self.data[k_i] > 0.0 {
                        let x_i = self.weights[j].offset(k);
                        self.data[x_i] = 1.0;
                    }
                }

                self.mul(v, j)
            }
        }
    }

    pub fn broadcast(&mut self, a: usize, dims: Dims) -> usize {
        let op = Some(Op::Broadcast);
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let in_shape = self.weights[a].shape;
        let in_dims = in_shape.dims;
        let in_rank = in_shape.rank();

        let mut strides = in_shape.strides;
        for i in 0..in_rank {
            if in_dims.0[i] == 1 {
                strides.0[i] = 0;
            }
        }

        let out_rank = dims.rank();
        let rank_diff = out_rank - in_rank;
        if rank_diff > 0 {
            strides.0.copy_within(0..in_rank, rank_diff);
            strides.0[..rank_diff].fill(0);
        }

        let out_shape = Shape { dims, strides };
        let offset = self.weights[a].offset;
        self.push_weight(offset, out_shape, inputs, op, false)
    }

    pub fn reshape(&mut self, a: usize, to_shape: Shape) -> usize {
        let from_shape = self.weights[a].shape;
        assert_eq!(from_shape.product(), to_shape.product());

        let data_offset = self.data.len();
        let input_offset = self.inputs.len();
        let inputs = Some(View::new(input_offset, 1));
        self.inputs.push(a);

        let mut iter = self.weights[a].nditer();
        for _ in 0..to_shape.product() {
            let a_i = iter.next().unwrap();
            self.data.push(self.data[a_i]);
        }

        let op = Some(Op::Reshape);
        let shape = Shape::from_dims(to_shape.dims);
        self.push_weight(data_offset, shape, inputs, op, true)
    }

    pub fn materialize(&mut self, a: usize) -> usize {
        let data_offset = self.data.len();
        let in_shape = self.weights[a].shape;

        let mut iter = self.weights[a].nditer();
        for _ in 0..in_shape.product() {
            let a_i = iter.next().unwrap();
            self.data.push(self.data[a_i]);
        }

        let op = self.weights[a].op;
        let shape = Shape::from_dims(in_shape.dims);
        let inputs = self.weights[a].inputs;
        self.push_weight(data_offset, shape, inputs, op, true)
    }

    pub fn transpose(&mut self, a: usize, outer: isize, inner: isize) -> usize {
        let op = Some(Op::Transpose { outer, inner });
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let offset = self.weights[a].offset;
        let in_shape = self.weights[a].shape;
        let mut out_shape = in_shape.clone();

        let inner = in_shape.real_index(inner);
        let outer = in_shape.real_index(outer);
        out_shape.dims.0[inner] = in_shape.dims.0[outer];
        out_shape.dims.0[outer] = in_shape.dims.0[inner];
        out_shape.strides.0[inner] = in_shape.strides.0[outer];
        out_shape.strides.0[outer] = in_shape.strides.0[inner];

        self.push_weight(offset, out_shape, inputs, op, false)
    }

    // Assumes all inputs have the same shape outside of dim
    pub fn _concat(&mut self, a: &[usize], dim: usize) -> usize {
        let op = Some(Op::_Concat { dim });
        let data_offset = self.data.len();
        let inputs_offset = self.inputs.len();
        let inputs_len = a.len();
        let inputs = Some(View::new(inputs_offset, inputs_len));
        self.inputs.extend_from_slice(a);

        let outer_dims = &self.weights[a[0]].shape.dims.0[..dim];
        let outer_len = Dims::product(outer_dims);
        for i in 0..outer_len {
            for b in a.iter().copied() {
                let inner_dims = &self.weights[b].shape.dims.0[dim..];
                let inner_len = Dims::product(inner_dims);

                for k in 0..inner_len {
                    let l_i = i * inner_len + k;
                    let d_i = self.weights[b].offset(l_i);
                    self.data.push(self.data[d_i]);
                }
            }
        }

        let out_dim = a.iter().copied().fold(0, |acc, next| {
            let x_dim = self.weights[next].shape.get_dim(dim);
            acc + x_dim
        });

        let mut out_shape = self.weights[a[0]].shape.clone();
        out_shape.dims.0[dim] = out_dim;
        out_shape.strides = out_shape.dims.strides();

        self.push_weight(data_offset, out_shape, inputs, op, true)
    }

    // Keeps the dimension if provided
    fn reduce<F: Fn(f32, f32) -> f32>(&mut self, a: usize, op: Op, f: F, init: f32) -> usize {
        let data_offset = self.data.len();
        let a_shape = self.weights[a].shape;

        let dim = match op {
            Op::Sum { dim } | Op::Max { dim, .. } => dim,
            _ => unreachable!("Reduce called with incorrect op"),
        };

        let inner_len = match dim {
            None => 1,
            Some(dim) => {
                let inner_dims = &a_shape.dims.0[dim + 1..];
                Dims::product(inner_dims)
            }
        };

        let dim_len = match dim {
            None => a_shape.product(),
            Some(dim) => a_shape.get_dim(dim),
        };

        let outer_len = match dim {
            None => 1,
            Some(dim) => {
                let outer_dims = &a_shape.dims.0[..dim];
                Dims::product(outer_dims)
            }
        };

        for i in 0..outer_len {
            for k in 0..inner_len {
                let mut acc = init;
                for j in 0..dim_len {
                    let l_i = (i * dim_len + j) * inner_len + k;
                    let d_i = self.weights[a].offset(l_i);
                    acc = f(acc, self.data[d_i]);
                }
                self.data.push(acc);
            }
        }

        let shape = match dim {
            None => Shape::new(&[]),
            Some(dim) => {
                let mut dims = a_shape.dims.clone();
                dims.0[dim] = 1;
                Shape::from_dims(dims)
            }
        };

        let inputs = Some(self.push_inputs(&[a]));
        self.push_weight(data_offset, shape, inputs, Some(op), true)
    }

    pub fn sum(&mut self, a: usize, dim: Option<usize>) -> usize {
        let f = |acc: f32, next: f32| acc + next;
        self.reduce(a, Op::Sum { dim }, f, 0.0)
    }

    fn max(&mut self, a: usize, dim: Option<usize>) -> usize {
        let f = |acc: f32, next: f32| acc.max(next);
        let zero = self.scalar(0.0); // TODO: Real argmax
        self.reduce(a, Op::Max { dim, argmax: zero }, f, f32::NEG_INFINITY)
    }

    pub fn rmsnorm(&mut self, a: usize) -> usize {
        let a_shape = self.weights[a].shape;
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

    pub fn softmax(&mut self, a: usize) -> usize {
        let a_shape = self.weights[a].shape;
        let dim = a_shape.rank() - 1;
        let max = self.max(a, Some(dim));
        let sub = self.sub(a, max);
        let exps = self.exp(sub);
        let sum = self.sum(exps, Some(dim));
        self.div(exps, sum)
    }

    pub fn copy_data(&mut self, from: usize, to: usize) {
        let from = self.weights[from];
        let to = self.weights[to];

        let n = from.shape.product();
        assert!(n == to.shape.product());

        let src = from.offset..from.offset + n;
        self.data.copy_within(src, to.offset);
    }

    pub fn push_weight(
        &mut self,
        offset: usize,
        shape: Shape,
        inputs: Option<View>,
        op: Option<Op>,
        is_contiguous: bool,
    ) -> usize {
        let w = Tensor::new(offset, shape, inputs, op, is_contiguous);
        self.weights.push(w);
        self.weights.len() - 1
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
        let a_shape = self.weights[a].shape;
        let b_shape = self.weights[b].shape;

        let m = a_shape.get_dim(-2);
        let n = b_shape.get_dim(-1);
        let k = a_shape.get_dim(-1);

        let a_batches = a_shape.batches();
        let b_batches = b_shape.batches();
        let out_batches = a_batches.broadcast(b_batches);
        let batch_count = Dims::product(&out_batches.0);

        let a_target = out_batches.add(&[m, k]);
        let b_target = out_batches.add(&[k, n]);
        let a = self.broadcast(a, a_target);
        let b = self.broadcast(b, b_target);

        let data_offset = self.data.len();
        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        let a_offset = self.weights[a].offset;
        let b_offset = self.weights[b].offset;
        let iter_shape = Shape::from_dims(out_batches);
        let mut a_batch_iter = iter_shape.nditer(a_offset);
        let mut b_batch_iter = iter_shape.nditer(b_offset);

        for _ in 0..batch_count {
            let a_batch_offset = a_batch_iter.next().unwrap();
            let b_batch_offset = b_batch_iter.next().unwrap();

            for i in 0..m * n {
                let row = i / n;
                let col = i % n;

                let mut dot = 0.0;
                for j in 0..k {
                    let row_offset = row * a_shape.get_stride(-2);
                    let a_offset = row_offset + j * a_shape.get_stride(-1);
                    let a_data = self.data[a_batch_offset + a_offset];

                    let col_offset = col * b_shape.get_stride(-1);
                    let b_offset = col_offset + j * b_shape.get_stride(-2);
                    let b_data = self.data[b_batch_offset + b_offset];

                    dot += a_data * b_data;
                }

                self.data.push(dot);
            }
        }

        let dims = out_batches.add(&[m, n]);
        let out_shape = Shape::from_dims(dims);
        let op = Some(Op::MatMul);
        self.push_weight(data_offset, out_shape, inputs, op, true)
    }

    // https://docs.pytorch.org/docs/2.13/generated/torch.matmul.html
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.weights[a].shape;
        let b_shape = self.weights[b].shape;
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
                    let a_i = self.weights[a].offset(i);
                    let b_i = self.weights[b].offset(i);

                    dot += self.data[a_i] * self.data[b_i];
                }

                self.data.push(dot);

                let out_shape = Shape::new(&[]);
                let op = Some(Op::MatMul);
                self.push_weight(data_offset, out_shape, inputs, op, true)
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
        let a_shape = self.weights[a].shape;
        let out_shape = a_shape.squeeze(dim);
        let offset = self.weights[a].offset;
        let inputs = Some(self.push_inputs(&[a]));
        let op = Some(Op::Squeeze);

        self.push_weight(offset, out_shape, inputs, op, false)
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let op = Some(Op::Add);

        let a_shape = self.weights[a].shape;
        let b_shape = self.weights[b].shape;
        let in_dims = a_shape.broadcast(b_shape);
        let a = self.broadcast(a, in_dims);
        let b = self.broadcast(b, in_dims);

        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        let data_offset = self.data.len();
        let mut a_iter = self.weights[a].nditer();
        let mut b_iter = self.weights[b].nditer();

        for _ in 0..Dims::product(&in_dims.0) {
            let a_i = a_iter.next().unwrap();
            let b_i = b_iter.next().unwrap();
            let s = self.data[a_i] + self.data[b_i];
            self.data.push(s);
        }

        let dims = self.weights[a].shape.dims;
        let shape = Shape::from_dims(dims);
        self.push_weight(data_offset, shape, inputs, op, true)
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let c = self.neg(b);
        self.add(a, c)
    }

    fn elementwise<F: Fn(f32) -> f32>(&mut self, a: usize, op: Op, f: F) -> usize {
        let op = Some(op);
        let inputs = Some(View::new(self.inputs.len(), 1));
        self.inputs.push(a);

        let mut shape = self.weights[a].shape;
        let data_offset = self.data.len();

        let mut iter = self.weights[a].nditer();
        for _ in 0..shape.product() {
            let d_i = iter.next().unwrap();
            let v = f(self.data[d_i]);
            self.data.push(v);
        }

        shape.strides = shape.dims.strides();
        self.push_weight(data_offset, shape, inputs, op, true)
    }

    pub fn neg(&mut self, i: usize) -> usize {
        let f = |x: f32| -x;
        self.elementwise(i, Op::Neg, f)
    }

    pub fn pow(&mut self, i: usize, n: f32) -> usize {
        let f = |x: f32| x.powf(n);
        self.elementwise(i, Op::Pow { n }, f)
    }

    pub fn log(&mut self, i: usize) -> usize {
        let f = |x: f32| x.ln();
        self.elementwise(i, Op::Log, f)
    }

    pub fn exp(&mut self, i: usize) -> usize {
        let f = |x: f32| x.exp();
        self.elementwise(i, Op::Exp, f)
    }

    pub fn relu(&mut self, i: usize) -> usize {
        let f = |x: f32| x.max(0.0);
        self.elementwise(i, Op::Relu, f)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let op = Some(Op::Mul);

        let a_shape = self.weights[a].shape;
        let b_shape = self.weights[b].shape;
        let in_dims = a_shape.broadcast(b_shape);
        let a = self.broadcast(a, in_dims);
        let b = self.broadcast(b, in_dims);

        let inputs = Some(View::new(self.inputs.len(), 2));
        self.inputs.push(a);
        self.inputs.push(b);

        let data_offset = self.data.len();
        let mut a_iter = self.weights[a].nditer();
        let mut b_iter = self.weights[b].nditer();

        for _ in 0..Dims::product(&in_dims.0) {
            let a_i = a_iter.next().unwrap();
            let b_i = b_iter.next().unwrap();
            let p = self.data[a_i] * self.data[b_i];
            self.data.push(p);
        }

        let dims = self.weights[a].shape.dims;
        let shape = Shape::from_dims(dims);
        self.push_weight(data_offset, shape, inputs, op, true)
    }

    pub fn div(&mut self, i: usize, j: usize) -> usize {
        let k = self.pow(j, -1.0);
        self.mul(i, k)
    }
}

#[cfg(test)]
mod tests {
    use crate::{dims::Dims, shape::Shape, tape::Tape};
    use pyo3::{PyResult, prelude::*, types::PyDict};
    use quickcheck::{Arbitrary, Gen, TestResult};
    use quickcheck_macros::quickcheck;
    use std::sync::Once;

    static PYTHON_INIT: Once = Once::new();

    fn gen_data(g: &mut Gen, len: usize) -> Vec<f32> {
        let mut data = vec![];

        while data.len() < len {
            let value = f32::arbitrary(g);
            if value.is_finite() {
                // Avoid overflow
                data.push(value % 10.0);
            }
        }

        data
    }

    fn gen_shape(g: &mut Gen) -> Vec<usize> {
        loop {
            let rank = usize::arbitrary(g) % 4 + 1;
            let shape: Vec<_> = (0..rank).map(|_| usize::arbitrary(g) % 4 + 1).collect();
            if shape.iter().product::<usize>() <= 100 {
                return shape;
            }
        }
    }

    #[derive(Debug, Clone)]
    struct UnaryData {
        a: Vec<f32>,
        shape: Vec<usize>,
    }

    impl Arbitrary for UnaryData {
        fn arbitrary(g: &mut Gen) -> Self {
            let shape = gen_shape(g);
            let len = shape.iter().product();

            Self {
                a: gen_data(g, len),
                shape,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct BinaryData {
        a: Vec<f32>,
        b: Vec<f32>,
        shape: Vec<usize>,
    }

    impl Arbitrary for BinaryData {
        fn arbitrary(g: &mut Gen) -> Self {
            let shape = gen_shape(g);
            let len = shape.iter().product();

            Self {
                a: gen_data(g, len),
                b: gen_data(g, len),
                shape,
            }
        }
    }

    #[derive(Debug, Clone)]
    struct MatrixData {
        a: Vec<f32>,
        b: Vec<f32>,
    }

    impl Arbitrary for MatrixData {
        fn arbitrary(g: &mut Gen) -> Self {
            Self {
                a: gen_data(g, 6),
                b: gen_data(g, 6),
            }
        }
    }

    fn tensor<'py>(
        torch: &Bound<'py, PyModule>,
        data: &[f32],
        shape: &[usize],
    ) -> PyResult<Bound<'py, PyAny>> {
        let args = (data.to_vec(),);
        let kwargs = PyDict::new(torch.py());
        kwargs.set_item("device", "cpu")?;
        kwargs.set_item("requires_grad", true)?;

        let t = torch.call_method("tensor", args, Some(&kwargs))?;
        t.call_method1("reshape", (shape.to_vec(),))
    }

    // Returns items in logical order
    fn flat_list(t: Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
        let t = t.call_method0("flatten")?;
        t.call_method0("tolist")?.extract()
    }

    fn init_py() {
        PYTHON_INIT.call_once(Python::initialize);
    }

    fn compare(tape: &mut Tape, expected: Bound<'_, PyAny>, actual: usize) -> PyResult<TestResult> {
        let expected = flat_list(expected).unwrap();

        let mut iter = tape.weights[actual].nditer().enumerate();
        while let Some((expected_i, actual_i)) = iter.next() {
            let expected = expected[expected_i];
            let actual = tape.data[actual_i];

            if expected != actual {
                let tolerance = 1e-5 * expected.abs().max(actual.abs()).max(1.0);
                let diff = (expected - actual).abs();

                assert!(diff <= tolerance);
            }
        }

        Ok(TestResult::passed())
    }

    fn test_unary(
        data: UnaryData,
        method: &str,
        tape_op: fn(&mut Tape, usize) -> usize,
    ) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &data.shape)?;
            let x = a.call_method0(method)?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let y = tape_op(&mut tape, a);

            compare(&mut tape, x, y)
        })
    }

    fn test_binary(
        data: BinaryData,
        method: &str,
        tape_op: fn(&mut Tape, usize, usize) -> usize,
    ) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &data.shape)?;
            let b = tensor(&torch, &data.b, &data.shape)?;
            let x = a.call_method1(method, (&b,))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let b = tape.tensor(&data.b);
            let y = tape_op(&mut tape, a, b);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_neg(data: UnaryData) -> PyResult<TestResult> {
        test_unary(data, "neg", Tape::neg)
    }

    #[quickcheck]
    fn test_relu(data: UnaryData) -> PyResult<TestResult> {
        test_unary(data, "relu", Tape::relu)
    }

    #[quickcheck]
    fn test_exp(data: UnaryData) -> PyResult<TestResult> {
        test_unary(data, "exp", Tape::exp)
    }

    #[quickcheck]
    fn test_pow(mut data: UnaryData) -> PyResult<TestResult> {
        for value in &mut data.a {
            *value = value.abs();
        }

        test_unary(data, "square", |tape, a| tape.pow(a, 2.0))
    }

    #[quickcheck]
    fn test_log(mut data: UnaryData) -> PyResult<TestResult> {
        for value in &mut data.a {
            *value = value.abs() + 1.0;
        }

        test_unary(data, "log", Tape::log)
    }

    #[quickcheck]
    fn test_sum(data: UnaryData) -> PyResult<TestResult> {
        test_unary(data, "sum", |tape, a| tape.sum(a, None))
    }

    #[quickcheck]
    fn test_add(data: BinaryData) -> PyResult<TestResult> {
        test_binary(data, "__add__", Tape::add)
    }

    #[quickcheck]
    fn test_mul(data: BinaryData) -> PyResult<TestResult> {
        test_binary(data, "__mul__", Tape::mul)
    }

    #[quickcheck]
    fn test_sub(data: BinaryData) -> PyResult<TestResult> {
        test_binary(data, "__sub__", Tape::sub)
    }

    #[quickcheck]
    fn test_div(mut data: BinaryData) -> PyResult<TestResult> {
        for value in &mut data.b {
            if value.abs() < 0.1 {
                *value = 1.0;
            }
        }

        test_binary(data, "__truediv__", Tape::div)
    }

    #[quickcheck]
    fn test_sum_dim(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let shape = &[1, data.a.len()];
            let a = tensor(&torch, &data.a[..], shape)?;
            let x = a.call_method1("sum", (1, true))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(shape));
            let y = tape.sum(a, Some(1));

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_broadcast(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let from = &[1, data.a.len()];
            let a = tensor(&torch, &data.a, from)?;
            let args = (vec![2, data.a.len()],);
            let x = a.call_method1("expand", args)?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(from));
            let to = Dims([2, data.a.len(), 0, 0]);
            let y = tape.broadcast(a, to);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_select(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let values = [data.a.as_slice(), data.a.as_slice()].concat();
            let a = tensor(&torch, &values, &[2, data.a.len()])?;
            let x = a.call_method1("select", (0, 1))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&values);
            let a = tape.reshape(a, Shape::new(&[2, data.a.len()]));
            let y = tape.select(a, 0, 1);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_rmsnorm(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &data.shape)?;
            let squared = a.call_method1("pow", (2.0,))?;
            let mean = squared.call_method1("mean", (-1, true))?;
            let mean = mean.call_method1("add", (1e-5,))?;
            let scale = mean.call_method1("pow", (-0.5,))?;
            let x = a.call_method1("mul", (&scale,))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(&data.shape));
            let y = tape.rmsnorm(a);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_softmax(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &data.shape)?;
            let x = a.call_method1("softmax", (-1,))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(&data.shape));
            let y = tape.softmax(a);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_reshape(data: UnaryData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let to = [1, data.a.len()];
            let a = tensor(&torch, &data.a, &[data.a.len()])?;
            let x = a.call_method1("reshape", (to.to_vec(),))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let y = tape.reshape(a, Shape::new(&to));

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_transpose(data: MatrixData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &[2, 3])?;
            let x = a.call_method1("transpose", (0, 1))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(&[2, 3]));
            let y = tape.transpose(a, 0, 1);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_matmul(data: MatrixData) -> PyResult<TestResult> {
        init_py();

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let a = tensor(&torch, &data.a, &[2, 3])?;
            let b = tensor(&torch, &data.b, &[3, 2])?;
            let x = a.call_method1("matmul", (&b,))?;

            let mut tape = Tape::new();
            let a = tape.tensor(&data.a);
            let a = tape.reshape(a, Shape::new(&[2, 3]));
            let b = tape.tensor(&data.b);
            let b = tape.reshape(b, Shape::new(&[3, 2]));
            let y = tape.matmul(a, b);

            compare(&mut tape, x, y)
        })
    }

    #[quickcheck]
    fn test_pipeline_backward(mut data: UnaryData) -> PyResult<TestResult> {
        init_py();

        // Keep the inputs away from ReLU's nondifferentiable point while still
        // exercising a nontrivial graph with broadcasting and a reduction.
        for value in &mut data.a {
            *value = (*value).clamp(-5.0, 5.0);
            if value.abs() < 0.1 {
                *value = 0.2;
            }
        }

        Python::attach(|py| {
            let torch = py.import("torch")?;
            let shape = [1, data.a.len()];

            let x = tensor(&torch, &data.a[..1], &[])?;
            x.call_method0("retain_grad")?;

            let input = tensor(&torch, &data.a, &[data.a.len(), 1])?;
            let expanded = x.call_method1("expand", (shape.to_vec(),))?;
            let projected = expanded.call_method1("matmul", (&input,))?;
            let bias = torch.call_method1("tensor", (0.25_f32,))?;
            let shifted = projected.call_method1("add", (&bias,))?;
            let scaled = shifted.call_method1("mul", (&bias,))?;
            let negated = scaled.call_method0("neg")?;
            let exponentiated = negated.call_method0("exp")?;
            let logged = exponentiated.call_method0("log")?;
            let activated = logged.call_method0("relu")?;
            let squared = activated.call_method1("pow", (2.0_f32,))?;
            let loss = squared.call_method0("sum")?;
            loss.call_method0("backward")?;
            let expected = flat_list(x.getattr("grad")?)?;

            let mut tape = Tape::new();
            let source = tape.scalar(data.a[0]);
            let expanded = tape.broadcast(source, Shape::new(&shape).dims);
            let input = tape.tensor(&data.a);
            let input = tape.reshape(input, Shape::new(&[data.a.len(), 1]));
            let projected = tape.matmul(expanded, input);
            let bias = tape.scalar(0.25);
            let shifted = tape.add(projected, bias);
            let scaled = tape.mul(shifted, bias);
            let negated = tape.neg(scaled);
            let exponentiated = tape.exp(negated);
            let logged = tape.log(exponentiated);
            let activated = tape.relu(logged);
            let squared = tape.pow(activated, 2.0);
            tape.sum(squared, None);
            tape.backward();

            let grad = tape.weights[source].grad.unwrap();
            let weight = tape.weights[grad];
            let actual: Vec<_> = weight.nditer().map(|i| tape.data[i]).collect();

            assert_eq!(expected.len(), actual.len());
            for (expected, actual) in expected.into_iter().zip(actual) {
                let tolerance = 1e-5 * expected.abs().max(actual.abs()).max(1.0);
                let diff = (expected - actual).abs();

                assert!(diff <= tolerance);
            }

            Ok(TestResult::passed())
        })
    }
}
