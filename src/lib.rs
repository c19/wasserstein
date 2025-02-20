//! This crate provides an implementation of the Wasserstein Distance metric on multi-dimensional probability distributions.
//! This implementation follows [this paper](https://arxiv.org/pdf/1805.07416v1.pdf)
//! It uses the Network Simplex algorithm from the [LEMON](http://lemon.cs.elte.hu/trac/lemon) library,
//! although we plan on removing our reliance on LEMON in favor of a Rust implementation
//! of Network Simplex with optimizations specific to our use case in Wasserstein distance.
//!
//! # Example
//! TODO
//!

pub mod graph;
pub mod wasserstein;
pub mod linfa;

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;

    use crate::{wasserstein::{wasserstein_1d, wasserstein_2d}, linfa::distance};


    #[test]
    fn test_distance_1d_sum_equal() {
        let left = array![0.0, 0.8, 0.0, 0.0, 0.0, 0.1, 0.1];
        let right = array![0.1, 0.1, 0.0, 0.7, 0.0, 0.0, 0.1];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 1.5);
    }

    #[test]
    fn test_distance_1d_0() {
        let left = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let right = array![0.0, 5.0, 3.0, 0.0, 2.0, 0.0, 0.0];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 22.0);
    }

    #[test]
    fn test_distance_1d_1() {
        let left = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let right = array![0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 0.0];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 3.0);
    }

    #[test]
    fn test_distance_1d_2() {
        let left = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let right = array![0.0, 2.0, 1.0, 0.0, 0.0, 3.0, 0.0];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 6.0);
    }

    #[test]
    fn test_distance_1d_3() {
        let left = array![2.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let right = array![0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 12.0);
    }


    #[test]
    fn test_distance_1d_4() {
        let left = array![0.2, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0];
        let right = array![0.0, 0.2, 0.1, 0.0, 0.0, 0.3, 0.0];
        let dist = distance(left.view(), right.view());
        assert_eq!(dist, 0.6);
    }

    #[test]
    fn test_wasserstein_1d_0() {
        let left = vec![2, 1, 0, 0, 3, 0, 4];
        let right = vec![0, 5, 3, 0, 2, 0, 0];
        let dist = wasserstein_1d(left, right).unwrap();
        assert_eq!(dist, 22);
    }

    #[test]
    fn test_wasserstein_1d_1() {
        let left = vec![2, 1, 0, 0, 3, 0, 0];
        let right = vec![0, 2, 1, 0, 3, 0, 0];
        let dist = wasserstein_1d(left, right).unwrap();
        assert_eq!(dist, 3);
    }

    #[test]
    fn test_wasserstein_1d_2() {
        let left = vec![2, 1, 0, 0, 3, 0, 0];
        let right = vec![0, 2, 1, 0, 0, 3, 0];
        let dist = wasserstein_1d(left, right).unwrap();
        assert_eq!(dist, 6);
    }

    #[test]
    fn test_wasserstein_1d_3() {
        let left = vec![2, 1, 0, 0, 3, 0, 0];
        let right = vec![0, 0, 2, 1, 0, 0, 3];
        let dist = wasserstein_1d(left, right).unwrap();
        assert_eq!(dist, 12);
    }

    #[test]
    fn test_wasserstein_2d() {
        let left = array![
            [2, 1, 0, 0, 3, 0, 4],
            [2, 1, 0, 0, 3, 0, 4],
            [2, 1, 0, 0, 3, 0, 4],
            [2, 1, 0, 0, 3, 0, 4],
        ];
        let right = array![
            [0, 5, 3, 0, 2, 0, 0],
            [0, 5, 3, 0, 2, 0, 0],
            [0, 5, 3, 0, 2, 0, 0],
            [0, 5, 3, 0, 2, 0, 0],
        ];
        let dist = wasserstein_2d(left, right).unwrap();
        assert_eq!(dist, 88);

        let left = array![
            [3, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ];
        let right = array![
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 3],
        ];
        let dist = wasserstein_2d(left, right).unwrap();
        assert_eq!(dist, 3 * 4);
    }
}
