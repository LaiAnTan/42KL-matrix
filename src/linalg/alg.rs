
use std::ops::{Add, Mul};

use crate::linalg::{Vector, errors::VectorError, utils::MulAdd};


// --- ex01: Linear Combination ---

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Result<Vector<K>, VectorError>
where
    K: Add<Output = K> + Mul<Output = K> + Clone + Copy + Default,
{
    if u.len() != coefs.len() && !u.iter().all(|vec| vec.size == u[0].size)
    {
        return Err(VectorError)
    }

    let mut result: Vector<K> = Vector {size: u[0].size,  store: vec![Default::default(); u[0].size]};

    for (i, vec) in u.iter().enumerate()
    {
        for (j, &val) in vec.store.iter().enumerate()
        {
            result[j] = val.mul_add(coefs[i], result[j]);
        }
    }

    Ok(result)
}

// --- ex02: Linear Interpolation ---

// fn lerp<V>(u: V, v: V, t: f32) -> V;
// {

// }

// unit tests
#[cfg(test)]
mod tests
{
    use super::{Vector, VectorError, linear_combination};

    #[test]
    fn test_linear_combination() -> Result<(), VectorError>
    {

        let e1 = Vector::from([1., 0., 0.]);
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);

        assert_eq!(linear_combination(&[e1, e2, e3], &[10., -2., 0.5])?.store, [10., -2., 0.5]);

        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([0., 10., -100.]);

        assert_eq!(linear_combination(&[v1, v2], &[10., -2.])?.store, [10., 0., 230.]);

        Ok(())
    }

}