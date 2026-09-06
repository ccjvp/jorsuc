pub const MAX_RANK: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
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
