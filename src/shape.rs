use crate::dims::{Dims, MAX_RANK};

pub trait ShapeIndex {
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
    pub dims: Dims,
    pub strides: Dims,
}

impl Shape {
    pub fn new(a: &[usize]) -> Self {
        let mut dims = Dims::default();
        for i in 0..a.len() {
            dims.0[i] = a[i];
        }

        let strides = dims.strides();
        Self { dims, strides }
    }

    pub fn from_dims(dims: Dims) -> Self {
        let strides = dims.strides();
        Self { dims, strides }
    }

    pub fn rank(&self) -> usize {
        self.dims.rank()
    }

    pub fn real_index(&self, i: isize) -> usize {
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
        for j in (0..self.rank()).rev() {
            let dim = self.dims.0[j];
            index.0[j] = i % dim;
            i = i / dim;
        }

        index
    }

    pub fn expand(&self, n: usize) -> Self {
        let rank = self.rank();
        if rank >= n {
            self.clone()
        } else {
            let mut out = self.clone();
            let diff = n - rank;
            out.dims.0.copy_within(0..rank, diff);
            out.dims.0[..diff].fill(1);
            out.strides.0.copy_within(0..rank, diff);
            out.strides.0[..diff].fill(0);
            out
        }
    }

    pub fn get_dim<I: ShapeIndex>(&self, i: I) -> usize {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.dims.0[i]
    }

    pub fn get_stride<I: ShapeIndex>(&self, i: I) -> usize {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.strides.0[i]
    }

    pub fn squeeze<I: ShapeIndex>(&self, i: I) -> Self {
        let i = i.to_isize();
        let i = self.real_index(i);
        self.drop(i)
    }

    pub fn broadcast(&self, other: Self) -> Dims {
        let self_rank = self.rank();
        let other_rank = other.rank();
        let mut self_iter = self.dims.0.iter().take(self_rank).rev();
        let mut other_iter = other.dims.0.iter().take(other_rank).rev();

        let n = self_rank.max(other_rank);
        let mut out_dims = Dims::default();
        for i in (0..n).rev() {
            let a = self_iter.next();
            let b = other_iter.next();
            out_dims.0[i] = a.max(b).copied().unwrap();
        }

        out_dims
    }

    pub fn batches(&self) -> Self {
        let r = self.rank();
        if r <= 2 {
            Shape::new(&[])
        } else {
            let mut out = self.clone();
            for i in r - 2..r {
                out.dims.0[i] = 0;
                out.strides.0[i] = 0;
            }
            out
        }
    }

    pub fn product(&self) -> usize {
        Dims::product(&self.dims.0)
    }

    pub fn drop(&self, i: usize) -> Self {
        let mut out = self.clone();
        out.dims.0[i..].rotate_left(1);
        out.strides.0[i..].rotate_left(1);
        out.dims.0[MAX_RANK - 1] = 0;
        out.strides.0[MAX_RANK - 1] = 0;
        out
    }

    pub fn nditer(self, start: usize) -> impl Iterator<Item = usize> {
        let rank = self.rank();
        let mut index = [0; MAX_RANK];

        let mut curr = start;
        std::iter::from_fn(move || {
            let prev = curr;

            for i in (0..rank).rev() {
                index[i] += 1;
                curr += self.get_stride(i);

                let dim = self.get_dim(i);
                if index[i] < dim {
                    break;
                } else {
                    index[i] = 0;
                    curr -= dim * self.get_stride(i);
                }
            }

            Some(prev)
        })
    }
}
