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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub dims: Dims,
    pub strides: Dims,
}

impl Shape {
    pub fn new(a: &[usize]) -> Self {
        assert!(a.len() <= MAX_RANK);

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
        assert!((i.abs() as usize) < MAX_RANK);

        if i >= 0 {
            i as usize
        } else {
            let r = self.rank() as isize;
            (r + i) as usize
        }
    }

    pub fn unravel_index(&self, i: usize) -> Dims {
        assert!(i < Shape::product(self));

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
            assert!(rank + diff <= MAX_RANK);

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

        assert!(i < MAX_RANK);
        self.dims.0[i]
    }

    pub fn get_stride<I: ShapeIndex>(&self, i: I) -> usize {
        let i = i.to_isize();
        let i = self.real_index(i);

        assert!(i < MAX_RANK);
        self.strides.0[i]
    }

    pub fn squeeze<I: ShapeIndex>(&self, i: I) -> Self {
        let i = i.to_isize();
        let i = self.real_index(i);

        assert!(i < MAX_RANK);
        assert_eq!(self.dims.0[i], 1);
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
            let a = self_iter.next().copied().unwrap_or(0);
            let b = other_iter.next().copied().unwrap_or(0);
            assert!(a == b || (a == 1 || b == 1) || (a == 0 || b == 0));

            out_dims.0[i] = a.max(b);
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

    pub fn span(&self) -> usize {
        let dim_iter = self.dims.0.iter();
        let stride_iter = self.strides.0.iter();

        let max_offset = dim_iter
            .zip(stride_iter)
            .take_while(|&(&dim, _)| dim != 0)
            .map(|(&dim, &stride)| (dim - 1) * stride)
            .sum::<usize>();

        // Zero based offset is off by one
        max_offset + 1
    }

    pub fn product(&self) -> usize {
        Dims::product(&self.dims.0)
    }

    pub fn drop(&self, i: usize) -> Self {
        assert!(i < MAX_RANK);

        let mut out = self.clone();
        out.dims.0[i..].rotate_left(1);
        out.strides.0[i..].rotate_left(1);
        out.dims.0[MAX_RANK - 1] = 0;
        out.strides.0[MAX_RANK - 1] = 0;
        out
    }

    pub fn nditer(self, start: usize) -> impl Iterator<Item = usize> {
        let rank = self.rank();
        let len = Shape::product(&self);
        let mut index = [0; MAX_RANK];

        let mut n = 0;
        let mut curr = start;
        std::iter::from_fn(move || {
            if n == len {
                return None;
            }

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

            n += 1;
            Some(prev)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        let rank_0 = Shape::new(&[0, 0, 0, 0]);
        let rank_1 = Shape::new(&[7, 0, 0, 0]);
        let rank_2 = Shape::new(&[8, 1, 0, 0]);
        let rank_4 = Shape::new(&[9, 5, 4, 2]);

        assert_eq!(rank_0.rank(), 0);
        assert_eq!(rank_1.rank(), 1);
        assert_eq!(rank_2.rank(), 2);
        assert_eq!(rank_4.rank(), 4);
    }

    #[test]
    fn test_real_index() {
        let shape = Shape::new(&[9, 5, 4, 2]);
        let neg = shape.real_index(-3);
        let pos = shape.real_index(2);

        assert_eq!(neg, 1);
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_unravel_index() {
        let shape = Shape::new(&[4, 2, 2, 1]);
        let index = shape.unravel_index(13);

        assert_eq!(index.0, [3, 0, 1, 0])
    }

    #[test]
    fn test_expand() {
        let shape = Shape::new(&[2, 8, 0, 0]);
        let expanded = shape.expand(3);

        assert_eq!(expanded.dims.0, [1, 2, 8, 0]);
        assert_eq!(expanded.strides.0, [0, 8, 1, 0]);
    }

    #[test]
    fn test_get_dim() {
        let shape = Shape::new(&[2, 8, 7, 0]);

        assert_eq!(shape.get_dim(0), 2);
        assert_eq!(shape.get_dim(2), 7);
    }

    #[test]
    fn test_get_stride() {
        let shape = Shape::new(&[2, 8, 7, 0]);

        assert_eq!(shape.get_stride(0), 56);
        assert_eq!(shape.get_stride(2), 1);
    }

    #[test]
    fn test_squeeze() {
        let shape = Shape::new(&[3, 1, 7, 0]);
        let squeezed = shape.squeeze(1);

        assert_eq!(squeezed.dims.0, [3, 7, 0, 0]);
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(&[3, 1, 7, 0]);
        let b = Shape::new(&[2, 7, 0, 0]);
        let broadcast = a.broadcast(b);

        assert_eq!(broadcast.0, [3, 2, 7, 0]);
    }

    #[test]
    fn test_batches() {
        let a = Shape::new(&[3, 1, 7, 8]);
        let b = Shape::new(&[1, 7, 8, 0]);
        let c = Shape::new(&[7, 8, 0, 0]);

        assert_eq!(a.batches().dims.0, [3, 1, 0, 0]);
        assert_eq!(b.batches().dims.0, [1, 0, 0, 0]);
        assert_eq!(c.batches().dims.0, [0, 0, 0, 0]);
    }

    #[test]
    fn test_drop() {
        let shape = Shape::new(&[3, 7, 8, 1]);

        assert_eq!(shape.drop(1).dims.0, [3, 8, 1, 0]);
    }

    #[test]
    fn test_nditer() {
        let mut shape = Shape::new(&[3, 2, 0, 0]);
        shape.strides.0[1] = 0; // Make it non-contiguous
        let mut iter = shape.nditer(0);

        assert_eq!(iter.next().unwrap(), 0);
        assert_eq!(iter.next().unwrap(), 0);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 4);
    }
}
