use rand::distributions::Distribution;
use rand::distributions::Uniform;

const MAX_RANK: usize = 4;

#[derive(Clone, Copy, Debug)]
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
pub struct Shape {
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
    pub fn new(a: &[usize]) -> Self {
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

    pub fn rank(&self) -> usize {
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

    pub fn unravel_index(&self, i: usize) -> Dims {
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

    pub fn get_dim<I: ShapeIndex>(&self, i: I) -> usize {
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

    pub fn product(&self) -> usize {
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
}

impl Tensor {
    pub fn new(offset: usize, shape: Shape, inputs: Option<View>, op: Option<Op>) -> Self {
        Self {
            shape,
            offset,
            op,
            inputs,
            grad: None,
        }
    }

    pub fn offset(&self, i: usize) -> usize {
        let index = self.shape.unravel_index(i);
        self.offset + dot_dims(index, self.shape.strides)
    }
}

pub struct Tape {
    pub values: Vec<Tensor>,
    pub data: Vec<f32>,
    pub inputs: Vec<usize>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            values: vec![],
            data: vec![],
            inputs: vec![],
        }
    }

    // TODO: Should this be on Gpt?
    pub fn causal_mask(&mut self, seq_len: usize) -> usize {
        let shape = Shape::new(&[seq_len, seq_len]);
        let zero = self.scalar(0.0);
        let out = self.reshape(zero, shape);
        for i in 0..shape.product() {
            let index = shape.unravel_index(i);
            let row = index[0].unwrap();
            for col in 0..index[1].unwrap() {
                if col <= row {
                    let d_i = self.values[out].offset(i);
                    self.data[d_i] = 1.0;
                }
            }
        }

        out
    }

    pub fn backward(&mut self) {
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
    pub fn random(&mut self, shape: Shape, scale: f32) -> usize {
        let offset = self.data.len();
        let in_dims = shape.dims[0].unwrap();
        let k = (6.0 / in_dims as f32).sqrt();
        let dist = Uniform::new(-k, k);
        let len = shape.product();

        let mut rng = rand::thread_rng();
        for _ in 0..len {
            self.data.push(dist.sample(&mut rng) * scale);
        }

        self.push(Tensor::new(offset, shape, None, None))
    }

    pub fn select(&mut self, a: usize, dim: usize, i: usize) -> usize {
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

    pub fn scalar(&mut self, value: f32) -> usize {
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

    pub fn reshape(&mut self, a: usize, to_shape: Shape) -> usize {
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

    pub fn transpose(&mut self, a: usize, outer: isize, inner: isize) -> usize {
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
    pub fn concat(&mut self, a: &[usize], dim: usize) -> usize {
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

    pub fn softmax(&mut self, a: usize) -> usize {
        let a_shape = self.values[a].shape;
        let dim = a_shape.rank() - 1;
        let max = self.max(a, Some(dim));
        let sub = self.sub(a, max);
        let exps = self.exp(sub);
        let sum = self.sum(exps, Some(dim));
        self.div(exps, sum)
    }

    pub fn copy_data(&mut self, from: usize, to: usize) {
        let from = self.values[from];
        let to = self.values[to];

        let n = from.shape.product();
        assert!(n == to.shape.product());

        let src = from.offset..from.offset + n;
        self.data.copy_within(src, to.offset);
    }

    pub fn push(&mut self, v: Tensor) -> usize {
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
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
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

    pub fn add(&mut self, a: usize, b: usize) -> usize {
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

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
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

    fn exp(&mut self, i: usize) -> usize {
        let f = |x: f32| x.exp();
        self.elementwise(i, Op::Exp, f)
    }

    pub fn relu(&mut self, i: usize) -> usize {
        let f = |x: f32| x.max(0.0);
        self.elementwise(i, Op::Relu, f)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
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

    pub fn div(&mut self, i: usize, j: usize) -> usize {
        let k = self.pow(j, -1.0);
        self.mul(i, k)
    }
}
