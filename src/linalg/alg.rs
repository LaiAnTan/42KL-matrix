
use std::ops::{Add, Mul, Sub};
use num_traits::Float;

use crate::linalg::{Vector, Matrix, errors, utils::MulAdd};

// --- ex01: Linear Combination ---

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Result<Vector<K>, errors::VectorError>
where
    K: Add<Output = K> + Mul<Output = K> + Clone + Copy + Default,
{
    if u.len() != coefs.len() && !u.iter().all(|vec| vec.size == u[0].size)
    {
        return Err(errors::VectorError)
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

pub fn lerp<V, T>(u: V, v: V, t: T) -> Result<V, ()>
where
    V: Sub<V> + Add<Output = V> + Mul<T, Output = V> + Clone,
    <V as Sub>::Output: Mul<T, Output = V>,
    T: Float,
{
    Ok((v - u.clone()) * t + u)
}

// --- ex03: Dot Product ---

// --- ex04: Norm ---

// --- ex05: Cosine ---

// --- ex06: Cross Product ---

// --- ex07: Linear Map, Matrix Multiplication ---

// --- ex09: Trace  ---

// --- ex09: Transpose ---

// --- ex10: Row - Echelon Form---

// --- ex11: Determinant ---

// --- ex12: Inverse ---

// --- ex13: Rank ---

// --- ex14: Projection Matrix ---

// --- ex15: Complex Vector Spaces ---

// unit tests
#[cfg(test)]
mod tests
{
    use super::{Vector, Matrix, errors};
    extern crate approx; // for floating point relative assertions

    use approx::{assert_relative_eq, relative_eq};

    #[test]
    fn test_linear_combination() -> Result<(), errors::VectorError>
    {
        use super::linear_combination;

        let e1 = Vector::from([1., 0., 0.]);
        let e2 = Vector::from([0., 1., 0.]);
        let e3 = Vector::from([0., 0., 1.]);

        assert_eq!(linear_combination(&[e1, e2, e3], &[10., -2., 0.5])?.store, [10., -2., 0.5]);

        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([0., 10., -100.]);

        assert_eq!(linear_combination(&[v1, v2], &[10., -2.])?.store, [10., 0., 230.]);

        Ok(())
    }

    #[test]
    fn test_linear_interpolation() -> Result<(), ()>
    {
        use super::lerp;

        assert_relative_eq!(lerp(0., 1., 0.)?, 0.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(1., 0., 0.)?, 1.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(0., 1., 0.5)?, 0.5, max_relative = 0.00001);
        assert_relative_eq!(lerp(21., 42., 0.3)?, 27.3, max_relative = 0.00001);

        let v1 = Vector::from([2., 1.]);
        let v2 = Vector::from([4., 2.]);

        assert_eq!(lerp(v1, v2, 0.3)?.store, [2.6, 1.3]);

        let m1 = Matrix::from([[2., 1.], [3., 4.]]);
        let m2 = Matrix::from([[20.,10.], [30., 40.]]);

        assert_eq!(lerp(m1, m2, 0.5)?.store, [[11., 5.5], [16.5, 22.]]);

        Ok(())
    }

}