
use std::{iter, ops::{Mul, Sub, Neg}};
use num::traits::{Float, MulAdd, Zero, float::TotalOrder};


use crate::linalg::{Vector, Matrix, errors};

// --- ex01: Linear Combination ---

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Result<Vector<K>, errors::VectorError>
where
    K: Clone + Copy + Zero + MulAdd<K, K, Output = K>,
{
    if u.len() != coefs.len() && !u.iter().all(|vec| vec.size == u[0].size)
    {
        return Err(errors::VectorError)
    }

    let mut result: Vector<K> = Vector {size: u[0].size,  store: vec![K::zero(); u[0].size]};

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
    V: Sub<V, Output = V> + MulAdd<T, V, Output = V> + Clone,
    <V as Sub>::Output: Mul<T, Output = V>,
    T: Float,
{
    Ok((v - u.clone()).mul_add(t, u))
}

/*
Notes:

V: Sub<V, Output = V>: generic type V must be able to subtract generic type V, producing an output of generic type V
<V as Sub>::Output: Mul<T, Output = V>, The output of subtracting genereic type V must be able to multiply generic type T to produce output of generic type V
*/

// --- ex03: Dot Product ---

impl<K> Vector<K>
where
    K: MulAdd<K, K, Output = K> + Clone + Copy + Zero,
{
    pub fn dot(&self, v: Vector<K>) -> Result<K, errors::VectorError>
    {
        if self.size != v.size
        {
            return Err(errors::VectorError)
        }

        let mut result = K::zero();

        for (&i, &j) in self.store.iter().zip(v.store.iter())
        {
            result = i.mul_add(j, result);
        }

        Ok(result)
    }
}

// --- ex04: Norm ---

// fast inverse square root (might make this generic later)
pub fn fi_sqrt(number: &f32) -> f32
{
    let i = number.to_bits();

    let i = 0x5f3759df - (i >> 1);

    let mut y = f32::from_bits(i);

    y = y * (1.5  - 0.5 * number * y * y);
    y * (1.5  - 0.5 * number * y * y)
}

// generic abs
pub fn abs<V>(number: V) -> Result<V, ()>
where
    V: Neg<Output = V> + Zero + TotalOrder,
{
    match number
    {
        number if number.total_cmp(&V::zero()).is_lt() => Ok(-number),
        _ => Ok(number),
    }
}

impl<V> Vector<V>
where
    V: Clone + Copy + Zero + TotalOrder + Mul<f32, Output = V> + Mul<V, Output = V>+ Neg<Output = V> + iter::Sum<V>,
    f32: iter::Sum<V>,
{
    pub fn norm_1(&self) -> V
    {
        self.store.iter().map(|&x| abs(x)
            .unwrap_or_else(|()| panic!("norm_1 failed"))).sum()
    }

    pub fn norm(&self) -> V
    where
        f32: Into<V>
    {
        let magnitude_squared_sum = self.store.iter().map(|&x| x * x).sum();

        let inv_sqrt = fi_sqrt(&magnitude_squared_sum);

        (inv_sqrt * magnitude_squared_sum).into()
    }

    pub fn norm_inf(&self) -> V
    {
        self.store.iter().map(|&x| abs(x).unwrap())
            .max_by(|&a, b| a.total_cmp(b))
            .unwrap_or_else(|| panic!("norm_inf failed"))
    }
}

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

    use approx::assert_relative_eq;

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

    #[test]
    fn test_dot_product() -> Result<(), errors::VectorError>
    {
        let u = Vector::from([0., 0.,]);
        let v = Vector::from([1., 1.]);

        assert_eq!(u.dot(v)?, 0.0);

        let u = Vector::from([1., 1.]);
        let v = Vector::from([1., 1.]);

        assert_eq!(u.dot(v)?, 2.0);

        let u = Vector::from([-1., 6.]);
        let v = Vector::from([3., 2.]);

        assert_eq!(u.dot(v)?, 9.0);

        Ok(())
    }

    // utility to round to specified number of dp
    fn round_to(x: f32, dp: f32) -> f32
    {
        let p = 10f32.powf(dp);
        (x * p).round() / p
    }

    #[test]
    fn test_norms() -> Result<(), errors::VectorError>
    {
        let u = Vector::from([0., 0., 0.]);

        assert_eq!([u.norm_1(), u.norm(), u.norm_inf()], [0., 0., 0.]);

        let u = Vector::from([1., 2., 3.]);

        assert_eq!([u.norm_1(), round_to(u.norm(), 6.), u.norm_inf()], [6., 3.741_657, 3.]);

        let u = Vector::from([-1., -2.]);

        assert_eq!([u.norm_1(), round_to(u.norm(), 6.), u.norm_inf()], [3., 2.236_068, 2.]);

        Ok(())
    }

}