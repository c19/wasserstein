use linfa::dataset::Float;
use linfa_nn::distance::Distance;
use ndarray::{Dimension, ArrayView};

use crate::wasserstein::wasserstein_1d;

/// Earth mover's distance or Wasserstein metric. https://en.wikipedia.org/wiki/Earth_mover%27s_distance
#[derive(Debug, Clone, PartialEq)]
pub struct EMD<F: Float>(pub F);
impl<F: Float> EMD<F> {
    pub fn new(p: F) -> Self {
        EMD(p)
    }
}
impl<F: Float> Distance<F> for EMD<F> {
    #[inline]
    fn distance<D: Dimension>(&self, a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
        let u64_a: Vec<u64> = a.iter().map(|one| (*one * Float::cast(1_0000_0000.0)).as_()).map(|one| one as u64).collect();
        let u64_b: Vec<u64> = b.iter().map(|one| (*one * Float::cast(1_0000_0000.0)).as_()).map(|one| one as u64).collect();
        let u64_dist = wasserstein_1d(u64_a, u64_b).unwrap();
        Float::cast((u64_dist as f64 )/ 1_0000_0000.0)
    }
}

pub fn distance<D: Dimension, F: Float>(a: ArrayView<F, D>, b: ArrayView<F, D>) -> F {
    let u64_a: Vec<u64> = a.iter().map(|one| (*one * Float::cast(1_0000_0000.0)).as_()).map(|one| one as u64).collect();
    let u64_b: Vec<u64> = b.iter().map(|one| (*one * Float::cast(1_0000_0000.0)).as_()).map(|one| one as u64).collect();
    let u64_dist = wasserstein_1d(u64_a, u64_b).unwrap();
    Float::cast((u64_dist as f64 )/ 1_0000_0000.0)
}