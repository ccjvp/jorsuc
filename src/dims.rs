pub const MAX_RANK: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims(pub [usize; MAX_RANK]);

impl Dims {
    // Shape strides should be used if there is not a need to recompute
    pub fn strides(self) -> Self {
        let mut strides = Self::default();
        let n = self.0.iter().take_while(|&d| *d != 0).count();
        for i in 0..n {
            strides.0[i] = Self::product(&self.0[i + 1..]);
        }
        strides
    }

    pub fn dot(self, b: Dims) -> usize {
        let mut dot = 0;
        for (i, v) in self.0.iter().enumerate() {
            dot += v * b.0[i];
        }
        dot
    }

    pub fn add(self, new_dims: &[usize]) -> Self {
        let mut out = self.clone();
        let mut new_iter = new_dims.iter();

        while let Some(pos) = out.0.iter().position(|&d| d == 0) {
            match new_iter.next() {
                Some(d) => out.0[pos] = *d,
                None => break,
            }
        }

        out
    }

    pub fn insert(self, dim: usize, i: usize) -> Dims {
        assert!(dim < self.rank());

        let mut out = self.clone();
        for j in (dim..self.0.len() - 1).rev() {
            out.0[j + 1] = out.0[j];
        }
        out.0[dim] = i;
        out
    }

    pub fn product(dims: &[usize]) -> usize {
        dims.iter().take_while(|d| **d != 0).product()
    }

    pub fn rank(self) -> usize {
        self.0.iter().take_while(|&d| *d != 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        let rank_0 = Dims([0, 0, 0, 0]);
        let rank_1 = Dims([7, 0, 0, 0]);
        let rank_2 = Dims([8, 1, 0, 0]);
        let rank_4 = Dims([9, 5, 4, 2]);

        assert_eq!(rank_0.rank(), 0);
        assert_eq!(rank_1.rank(), 1);
        assert_eq!(rank_2.rank(), 2);
        assert_eq!(rank_4.rank(), 4);
    }

    #[test]
    fn test_product() {
        let identity = Dims([0, 0, 0, 0]);
        let six_seven = Dims([6, 7, 0, 0]);

        assert_eq!(Dims::product(&identity.0), 1);
        assert_eq!(Dims::product(&six_seven.0), 42);
    }

    #[test]
    fn test_insert() {
        let a = Dims([8, 0, 0, 0]);

        let b = a.insert(0, 4);
        assert_eq!(b.0, [4, 8, 0, 0]);

        let c = b.insert(1, 2);
        assert_eq!(c.0, [4, 2, 8, 0]);
    }

    #[test]
    fn test_add() {
        let a = Dims([8, 0, 0, 0]);
        let b = a.add(&[2, 1]);

        assert_eq!(b.0, [8, 2, 1, 0]);
    }

    #[test]
    fn test_dot() {
        let a = Dims([8, 4, 2, 0]);
        let b = Dims([1, 9, 4, 0]);

        assert_eq!(a.dot(b), 52);
    }

    #[test]
    fn test_strides() {
        let dims = Dims([2, 3, 7, 2]);
        let strides = dims.strides();

        assert_eq!(strides.0, [42, 14, 2, 1]);
    }
}
