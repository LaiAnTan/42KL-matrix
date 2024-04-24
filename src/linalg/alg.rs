use std::{iter::Sum, ops::{Mul, Sub, Neg, Div}};
use num::traits::{Float, MulAdd, Zero, float::TotalOrder};

use super::{errors::{self, MatrixError}, Matrix, Vector};

use super::errors::VectorError;

// --- ex01: Linear Combination ---

pub fn linear_combination<K>(u: &[Vector<K>], coefs: &[K]) -> Result<Vector<K>, errors::VectorError>
where
    K: Clone 
        + Copy
        + Zero
        + MulAdd<K, K, Output = K>,
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
    V: Clone
        +Sub<V, Output = V> 
        + MulAdd<T, V, Output = V>,
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
    K: Clone + Copy + Zero
{
    pub fn dot(&self, v: &Vector<K>) -> Result<K, errors::VectorError>
    where
        K: MulAdd<K, K, Output = K>,
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
fn fi_sqrt(number: &f32) -> f32
{
    let i = number.to_bits();

    let i = 0x5f3759df - (i >> 1);

    let mut y = f32::from_bits(i);

    y = y * (1.5  - 0.5 * number * y * y);
    y * (1.5  - 0.5 * number * y * y)
}

// generic abs
fn abs<V>(number: V) -> Result<V, ()>
where
    V: Neg<Output = V> + Zero + PartialOrd,
{
    match number
    {
        number if number < V::zero() => Ok(-number),
        _ => Ok(number),
    }
}

impl<V> Vector<V>
where
    V: Clone
        + Copy
        + Zero
        + Mul<V, Output = V>
        + Neg<Output = V>
{
    pub fn norm_1(&self) -> Result<V, VectorError>
    where
        V: PartialOrd
            + Sum<V>
    {
        Ok(self.store.iter()
            .filter_map(|&x| abs(x).ok())
            .sum())
    }

    pub fn norm(&self) -> Result<V, VectorError>
    where
        V: Sum<V>,
        f32: Into<V>
            + Sum<V>,
    {
        let magnitude_squared_sum = self.store.iter().map(|&x| x * x)
                                            .sum();

        let inv_sqrt = fi_sqrt(&magnitude_squared_sum);

        Ok((inv_sqrt * magnitude_squared_sum).into())
    }

    pub fn norm_inf(&self) -> Result<V, VectorError>
    where
        V: TotalOrder 
            + PartialOrd
    {
        // calculate absolute maximum
        let result = self.store.iter()
            .filter_map(|&x| abs(x).ok())
            .max_by(|&a, b| a.total_cmp(b));

        match result
        {
            Some(x) => Ok(x),
            None => Err(VectorError), // this should never happen but just in case
        }
    }
}

// --- ex05: Cosine ---

impl<K> Vector<K>
where
    K: Clone
        + Copy
        + Zero
{
    pub fn angle_cos(&self, v: &Vector<K>) -> Result<K, VectorError>
    where
        K: MulAdd<Output = K> 
            + Mul<K, Output = K> 
            + Neg<Output = K> 
            + Div<K, Output = K>
            + Sum<K>,
        f32: Into<K> + Sum<K>,
    {
        Ok(self.dot(v)? / (self.norm()? * v.norm()?))
    }
}

// --- ex06: Cross Product ---

impl<K> Vector<K>
where
    K: Clone
        + Copy
        + Zero
{
    pub fn cross_product(&self, v: &Vector<K>) -> Result<Vector<K>, VectorError>
    where
        K: Mul<K, Output = K>
            + Sub<K, Output = K>,
    {
        if self.size != 3 && v.size() != 3
        {
            return Err(VectorError)
        }
        
        let mut store = vec![K::zero(); 3];

        store[0] = (self[1] * v[2]) - (self[2] * v[1]);
        store[1] = (self[2] * v[0]) - (self[0] * v[2]);
        store[2] = (self[0] * v[1]) - (self[1] * v[0]);

        Ok(Vector { size: self.size, store })
    }
}

// --- ex09: Trace  ---

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero
{
    pub fn trace(&self) -> Result<K, MatrixError>
    where
        K: std::iter::Sum
    {
        if !self.is_square()
        {
            return Err(MatrixError)
        }

        let trace: K = self.store.iter().enumerate()
            .map(|(i, row)| row[i])
            .sum();

        Ok(trace)
    }
}
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
    use crate::linalg::errors::{MatrixError, VectorError};

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

        assert_eq!(u.dot(&v)?, 0.0);

        let u = Vector::from([1., 1.]);
        let v = Vector::from([1., 1.]);

        assert_eq!(u.dot(&v)?, 2.0);

        let u = Vector::from([-1., 6.]);
        let v = Vector::from([3., 2.]);

        assert_eq!(u.dot(&v)?, 9.0);

        Ok(())
    }

    // utility to round to specified number of dp
    fn round_to(x: f32, dp: f32) -> f32
    {
        let p = 10f32.powf(dp);
        (x * p).round() / p
    }

    #[test]
    fn test_norms() -> Result<(), VectorError>
    {
        let u = Vector::from([0., 0., 0.]);

        assert_eq!([u.norm_1()?, u.norm()?, u.norm_inf()?], [0., 0., 0.]);

        let u = Vector::from([1., 2., 3.]);

        assert_eq!([u.norm_1()?, round_to(u.norm()?, 6.), u.norm_inf()?], [6., 3.741_657, 3.]);

        let u = Vector::from([-1., -2.]);

        assert_eq!([u.norm_1()?, round_to(u.norm()?, 6.), u.norm_inf()?], [3., 2.236_068, 2.]);

        Ok(())
    }

    #[test]
    fn test_cos_angle() -> Result<(), VectorError>
    {

        let u = Vector::from([1., 0.]);
        let v = Vector::from([1., 0.]);

        assert_eq!(round_to(u.angle_cos(&v)?, 1.), 1.);

        let u = Vector::from([1., 0.]);
        let v = Vector::from([0., 1.]);

        assert_eq!(round_to(u.angle_cos(&v)?, 1.), 0.);

        let u = Vector::from([-1., 1.]);
        let v = Vector::from([ 1., -1.]);

        assert_eq!(round_to(u.angle_cos(&v)?, 1.), -1.);

        let u = Vector::from([2., 1.]);
        let v = Vector::from([4., 2.]);

        assert_eq!(round_to(u.angle_cos(&v)?, 1.), 1.);

        let u = Vector::from([1., 2., 3.]);
        let v = Vector::from([4., 5., 6.]);

        assert_eq!(u.angle_cos(&v)?, 0.974_631_9);

        Ok(())
    }

    #[test]
    fn test_cross_product() -> Result<(), VectorError>
    {
        let u = Vector::from([0., 0., 1.]);
        let v = Vector::from([1., 0., 0.]);

        assert_eq!(u.cross_product(&v)?.store, [0., 1., 0.]);

        let u = Vector::from([1., 2., 3.]);
        let v = Vector::from([4., 5., 6.]);
        
        assert_eq!(u.cross_product(&v)?.store, [-3., 6., -3.]);

        let u = Vector::from([4., 2., -3.]);
        let v = Vector::from([-2., -5., 16.]);

        assert_eq!(u.cross_product(&v)?.store, [17., -58., -16.]);

        Ok(())
    }

    #[test]
    fn test_trace() -> Result<(), MatrixError>
    {

        let u = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);

        assert_eq!(u.trace()?, 2.0);

        let u = Matrix::from([
            [2., -5., 0.],
            [4., 3., 7.],
            [-2., 3., 4.],
            ]);

        assert_eq!(u.trace()?, 9.0);

        let u = Matrix::from([
            [-2., -8., 4.],
            [1., -23., 4.],
            [0., 6., 4.],
            ]);

        assert_eq!(u.trace()?, -21.0);

        Ok(())
    }

}
