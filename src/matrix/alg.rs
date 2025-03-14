use std::{iter::Sum, ops::{Mul, Sub, Neg, Div}, fmt::Display};
use num::{traits::{float::TotalOrder, Float, MulAdd, Zero, One, pow}, FromPrimitive};

use super::{errors::{self, MatrixError}, Matrix, Vector};

use super::errors::VectorError;

// --- ex01: Linear Combination ---

/*
Time complexity: O(n)
Space Complexity: O(n)
where n is the total number of coordinates in all vectors.
*/

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

    for (i, vec) in u.iter().enumerate() // O(n)
    {
        for (j, &val) in vec.store.iter().enumerate()
        {
            result[j] = val.mul_add(coefs[i], result[j]);
        }
    }

    Ok(result)
}

// --- ex02: Linear Interpolation ---

pub fn lerp<V, T>(u: V, v: V, t: T) -> V
where
    V: Clone
        + Sub<V, Output = V> 
        + MulAdd<T, V, Output = V>,
    <V as Sub>::Output: Mul<T, Output = V>,
    T: Float,
{
    // O(subtract) = O(n)
    (v - u.clone()).mul_add(t, u)
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

        // O(n)
        let res = self.store.iter().zip(v.store.iter())
            .fold(K::zero(), |acc, (&i, &j)| i.mul_add(j, acc));

        Ok(res)
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
    pub fn norm_1(&self) -> V
    where
        V: PartialOrd
            + Sum<V>
    {
        // O(n)
        self.store.iter().filter_map(|&x| abs(x).ok()).sum()
    }

    pub fn norm(&self) -> V
    where
        V: Sum<V>,
        f32: Into<V>
            + Sum<V>,
    {
        // O(n), does not use forbidden sqrt()

        let magnitude_squared_sum = self.store.iter().map(|&x| x * x)
                                            .sum();
        let inv_sqrt = fi_sqrt(&magnitude_squared_sum);

        (inv_sqrt * magnitude_squared_sum).into()
    }

    pub fn norm_inf(&self) -> Result<V, VectorError>
    where
        V: TotalOrder 
            + PartialOrd
    {
        // O(n)
        // calculate absolute maximum
        let result = self.store.iter()
            .filter_map(|&x| abs(x).ok())
            .max_by(|&a, b| a.total_cmp(b));

        match result
        {
            Some(x) => Ok(x),
            None => Err(VectorError)
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
        // O(n)
        Ok(self.dot(v)? / (self.norm() * v.norm()))
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

// --- ex08: Trace  ---

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

        // O(n)
        let trace: K = self.store.iter().enumerate()
            .map(|(i, row)| row[i])
            .sum();

        Ok(trace)
    }
}

// --- ex09: Transpose ---

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero
{
    pub fn transpose(&mut self) -> Matrix<K>
    where
        K: std::iter::Sum
    {
        // complexity depends on what N is, but it alwas follows subject's specs
        self.store = (0..self.columns)
            .map(|index| self.col(index))
            .collect();

        Matrix {rows: self.rows, columns: self.columns, store: self.store.clone()}
    }
}

// --- ex10: Row - Echelon Form---

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero
        + One
{

    pub fn row_echelon(&mut self) -> Matrix<K>
    where
        K: PartialEq + FromPrimitive + Div<Output = K> + Sub<Output = K>,
    {
        /* 
        Reduced row echelon form calculator using Gaussian Elimination.
        ci => column index
        pi => pivot index (row)
         */

        // find index of first non zero column
        let start_ci: usize = match (0..self.columns)
            .find(|index| self.col(*index).iter().any(|item| *item != K::zero()))
        {
            Some(x) => x,
            None => return self.clone(),
        };

        // find index of pivot of first non zero column
        let mut pi = 0;

        for ci in start_ci..self.columns
        {
            if pi == self.rows
            {
                break
            }
            
            // if pivot is zero we perform a row swap if possible
            if self.store[pi][ci] == K::zero()
            {
                match (pi + 1..self.rows).find(|&i| self.store[i][ci] != K::zero())
                {
                    Some(swap_row) => self.store.swap(pi, swap_row),
                    None => continue,
                };
            }

            // scale pivot row so that pivot == 1
            let sf = self.store[pi][ci];
            self.store[pi].iter_mut().for_each(|item| *item = *item / sf);

            /*
            make every row after pivot 0
            row[ci] => element for each row below the pivot
             */
            let pivot_clone = self.store[pi].clone();

            self.store.iter_mut()
                .skip(pi + 1)
                .filter(|row| row[ci] != K::zero())
                .for_each(|row| {
                    // scale below pivot == 0
                    // new row = row - (scalar * pivot row)
                    // eg: 5 - (5 * 1) = 0
                    *row = row.iter().enumerate()
                        
                        .map(|(j, &ej)| ej - (row[ci] * pivot_clone[j]))
                        .collect::<Vec<K>>();
                });

            pi += 1;
        }

        // -- reduced row echelon form starts here -- O(n^3) time complexity
        for ri in (1..self.rows).rev()
        {

            let pivot = match (0..self.columns).rev()
                .find(|&ci| self.store[ri][ci] == K::one())
                {
                    Some(pivot) => pivot,
                    None => continue,
                };

            if self.col(pivot)[0..ri].iter().all(|&item| item == K::zero())
            {
                continue;
            }

            let curr_row = self.store[ri].clone();

            self.store.iter_mut()
                .rev() // self.rows to 0
                .skip(self.rows - ri)
                .filter(|row| row[pivot] != K::zero())
                .for_each(|row| {
                    *row = row.iter().enumerate()
                        .map(|(ci, &elem)| elem - (row[pivot] * curr_row[ci]))
                        .collect::<Vec<K>>();
                });
        }

        self.clone()
    }
}

// --- ex11: Determinant ---

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero + One 
        + Neg<Output = K> 
        + Sub<Output = K> 
        + Mul<Output = K> 
        + From<i32>
{
    fn remove_col(&mut self, index: usize) -> Result<&mut Self, ()>
    {
        self.store.iter_mut().for_each(|row| {
            row.remove(index);
        });

        Ok(self)
    }

    
    fn det_aux(&mut self, size: usize) -> Result<K, ()>
    {
        // alternative method, laplace expansion w/ O(n!) time complexity

        /*
        when A is 2x2 matrix
        [a b]
        [c d]
        det(A) = ad - cb
         */
        if size == 2
        {
            return Ok((self.store[0][0] * self.store[1][1])
                - (self.store[0][1] * self.store[1][0]));
        }

        /*
        when A is n by n matrix we perform laplace expansion along the first row
         */

        // remove first row because we expand along first row
        let first_row = self.store[0].clone();
        self.store.remove(0);

        // Laplace expansion w/ recursion
        Ok((0..size).fold(K::zero(), |acc: K, x|{
            acc + (K::from(pow(-1, x + 1)) * first_row[x] * self.clone()
                .remove_col(x).unwrap()
                .det_aux(size - 1).unwrap())
        }))

    }

    fn det_gaussian(&self) -> Result<K, ()>
    where
        K: PartialEq + FromPrimitive + Div<Output = K> + Sub<Output = K> + std::fmt::Debug,
    {
        let mut aux = self.clone();

         // find index of first non zero column
         let start_ci: usize = match (0..aux.columns)
        .find(|index| aux.col(*index).iter().any(|item| *item != K::zero()))
        {
            Some(x) => x,
            None => return Ok(K::zero()),
        };

        let mut pi = 0;

        // O(n^3)
        for ci in start_ci..aux.columns
        {
            if pi == self.rows
            {
                break
            }
            
            // if pivot is zero we perform a row swap if possible
            if aux.store[pi][ci] == K::zero()
            {
                match (pi + 1..aux.rows).find(|&i| aux.store[i][ci] != K::zero())
                {
                    Some(swap_row) => aux.store.swap(pi, swap_row),
                    None => continue,
                };
            }

            /*
            make every row after pivot 0
            row[ci] => element for each row below the pivot
            */
            let pivot_clone = aux.store[pi].clone();

            aux.store.iter_mut()
                .skip(pi + 1)
                .filter(|row| row[ci] != K::zero())
                .for_each(|row| {
                    // scale below pivot == 0
                    // new row = row - (scalar * pivot row)
                    // eg: 5 - (5 * 1) = 0
                    *row = row.iter().enumerate()
                        
                        .map(|(j, &ej)| ej - (row[ci] * pivot_clone[j]))
                        .collect::<Vec<K>>();
                });

            pi += 1;
        }

        let det = (0..aux.rows).fold(K::one(), |total, r| total * aux.store[r][r]);

        Ok(det)
    }

    pub fn determinant(&self) -> Result<K, MatrixError>
    where
        K: PartialEq + FromPrimitive + Div<Output = K> + Sub<Output = K> + std::fmt::Debug,
    {
        if !self.is_square()
        {
            return Err(MatrixError);
        }

        Ok(self.clone().det_gaussian().unwrap())
        // Ok(self.clone().det_aux(self.rows).unwrap())
    }
}

// --- ex12: Inverse --- 

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero
        + One + Display + std::fmt::Debug
{
    pub fn inverse(&mut self) -> Result<Matrix<K>, MatrixError>
    where
        K: PartialEq + FromPrimitive + Div<Output = K> + Sub<Output = K>,
    {
        if !self.is_square()
        {
            return Err(MatrixError);
        }

        /* 
        Inverse matrix function using Gauss-Jordan Elimination.
        ci => column index
        pi => pivot index (row)
        */

        let mut inv = Matrix::new_identity(self.rows);

        // find index of first non zero column
        let start_ci: usize = match (0..self.columns)
            .find(|index| self.col(*index).iter().any(|item| *item != K::zero()))
        {
            Some(x) => x,
            None => return Ok(inv),
        };

        let mut pi = 0;

        for ci in start_ci..self.columns
        {
            if pi == self.rows
            {
                break
            }
            
            // if pivot is zero we perform a row swap if possible
            if self.store[pi][ci] == K::zero()
            {
                match (pi + 1..self.rows).find(|&i| self.store[i][ci] != K::zero())
                {
                    Some(swap_row) => {
                        self.store.swap(pi, swap_row);
                        inv.store.swap(pi, swap_row);

                    },
                    None => continue,
                };
            }

            // scale pivot row so that pivot == 1
            let sf = self.store[pi][ci];
            self.store[pi].iter_mut().for_each(|item| *item = *item / sf);
            inv.store[pi].iter_mut().for_each(|item| *item = *item / sf);

            /*
            make every row after pivot 0
            row[ci] => element for each row below the pivot
             */
            let pivot_clone = self.store[pi].clone();
            let pivot_inv_clone = inv.store[pi].clone();

            self.store.iter_mut()
                .zip(inv.store.iter_mut())
                .skip(pi + 1)
                .filter(|(row, _)| row[ci] != K::zero())
                .for_each(|(row, inv_row)| {
                    let sf = row[ci]; // scale factor

                    row.iter_mut().zip(inv_row.iter_mut()).enumerate()
                    .for_each(|(j, (elem, inv_elem))| {
                        *elem = *elem - (sf * pivot_clone[j]);
                        *inv_elem = *inv_elem - (sf * pivot_inv_clone[j]);
                    });

                });

            pi += 1;
        }

        // -- reduced row echelon form starts here --
        for ri in (1..self.rows).rev()
        {

            let pivot = match (0..self.columns).rev()
                .find(|&ci| self.store[ri][ci] == K::one())
                {
                    Some(pivot) => pivot,
                    None => continue,
                };

            if self.col(pivot)[0..ri].iter().all(|&item| item == K::zero())
            {
                continue;
            }

            let curr_row = self.store[ri].clone();
            let curr_inv_row = inv.store[ri].clone();

            self.store.iter_mut()
                .zip(inv.store.iter_mut())
                .rev() // self.rows to 0
                .skip(self.rows - ri)
                .filter(|(row, _)| row[pivot] != K::zero())
                .for_each(|(row, inv_row)| {
                    let sf = row[pivot];
                    
                    row.iter_mut().zip(inv_row.iter_mut()).enumerate()
                    .for_each(|(ci, (elem, inv_elem))| {
                        *elem = *elem - (sf * curr_row[ci]);
                        *inv_elem = *inv_elem - (sf * curr_inv_row[ci]);
                    });

                });
        }

        // check if matrix is an identity matrix
        if self.is_identity()
        {
            return Ok(inv);
        }

        Err(MatrixError)
    }
}

// --- ex13: Rank ---

impl<K> Matrix::<K>
where
    K: Clone
        + Copy
        + Zero
        + One
{
    pub fn rank(&self) -> usize
    where
        K: PartialEq + FromPrimitive + Div<Output = K> + Sub<Output = K>,
    {
        let rank = self.clone().row_echelon().store
            .iter().fold(0, |acc , x| {
                if (*x).iter().any(|&elem| elem != K::zero())
                {
                    return acc + 1;
                }
                acc
            });
        rank
    }
}

// unit tests

#[cfg(test)]
mod tests
{
    use crate::matrix::errors::{MatrixError, VectorError};

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
        let v1 = Vector::from([1., 2., 3.]);
        let v2 = Vector::from([0., 10., -100.]);

        assert_eq!(linear_combination(&[e1, e2, e3], &[10., -2., 0.5])?.store, [10., -2., 0.5]);
        assert_eq!(linear_combination(&[v1, v2], &[10., -2.])?.store, [10., 0., 230.]);

        // eval sheet tests
        assert_eq!(linear_combination(&[Vector::from([-42., 42.])], &[-1.])?.store, [42., -42.]);
        assert_eq!(linear_combination(&[Vector::from([-42.]), Vector::from([-42.]), Vector::from([-42.])], &[-1., 1., 0.])?.store, [0.]);
        assert_eq!(linear_combination(&[Vector::from([-42., 42.]), Vector::from([1., 3.]), Vector::from([10., 20.])], &[1., -10., -1.])?.store, [-62., -8.]);
        assert_eq!(linear_combination(&[Vector::from([-42., 100., -69.5]), Vector::from([1., 3., 5.])], &[1., -10.])?.store, [-52., 70., -119.5]);
        
        Ok(())
    }

    #[test]
    fn test_linear_interpolation() -> Result<(), ()>
    {
        use super::lerp;

        let v1 = Vector::from([2., 1.]);
        let v2 = Vector::from([4., 2.]);
        let m1 = Matrix::from([[2., 1.], [3., 4.]]);
        let m2 = Matrix::from([[20.,10.], [30., 40.]]);

        assert_relative_eq!(lerp(0., 1., 0.), 0.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(1., 0., 0.), 1.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(0., 1., 0.5), 0.5, max_relative = 0.00001);
        assert_relative_eq!(lerp(21., 42., 0.3), 27.3, max_relative = 0.00001);
        assert_eq!(lerp(v1, v2, 0.3).store, [2.6, 1.3]);
        assert_eq!(lerp(m1, m2, 0.5).store, [[11., 5.5], [16.5, 22.]]);

        // eval sheet tests
        assert_relative_eq!(lerp(0., 1., 0.), 0.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(0., 1., 1.), 1.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(0., 42., 0.5), 21.0, max_relative = 0.00001);
        assert_relative_eq!(lerp(-42., 42., 0.5), 0.0, max_relative = 0.00001);
        assert_eq!(lerp(Vector::from([-42., 42.]), Vector::from([42., -42.]), 0.5).store, [0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_dot_product() -> Result<(), errors::VectorError>
    {
        let a = Vector::from([0., 0.,]);
        let b = Vector::from([1., 1.]);
        let c = Vector::from([1., 1.]);
        let d = Vector::from([1., 1.]);
        let e = Vector::from([-1., 6.]);
        let f = Vector::from([3., 2.]);

        assert_eq!(a.dot(&b)?, 0.0);
        assert_eq!(c.dot(&d)?, 2.0);
        assert_eq!(e.dot(&f)?, 9.0);

        // eval sheet tests
        assert_eq!(Vector::from([0, 0]).dot(&Vector::from([0, 0]))?, 0);
        assert_eq!(Vector::from([1, 0]).dot(&Vector::from([0, 0]))?, 0);
        assert_eq!(Vector::from([1, 0]).dot(&Vector::from([1, 0]))?, 1);
        assert_eq!(Vector::from([1, 0]).dot(&Vector::from([0, 1]))?, 0);
        assert_eq!(Vector::from([1, 1]).dot(&Vector::from([1, 1]))?, 2);
        assert_eq!(Vector::from([4, 2]).dot(&Vector::from([2, 1]))?, 10);

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
        let a = Vector::from([0., 0., 0.]);
        let b = Vector::from([1., 2., 3.]);
        let c = Vector::from([-1., -2.]);

        assert_eq!([a.norm_1(), a.norm(), a.norm_inf()?], [0., 0., 0.]);
        assert_eq!([b.norm_1(), round_to(b.norm(), 6.), b.norm_inf()?], [6., 3.741_657, 3.]);
        assert_eq!([c.norm_1(), round_to(c.norm(), 6.), c.norm_inf()?], [3., 2.236_068, 2.]);

        // eval sheet tests
        assert_eq!(Vector::from([0]).norm_1(), 0);
        assert_eq!(Vector::from([1]).norm_1(), 1);
        assert_eq!(Vector::from([0, 0]).norm_1(), 0);
        assert_eq!(Vector::from([1, 0]).norm_1(), 1);
        assert_eq!(Vector::from([2, 1]).norm_1(), 3);
        assert_eq!(Vector::from([4, 2]).norm_1(), 6);
        assert_eq!(Vector::from([-4, -2]).norm_1(), 6);
        assert_relative_eq!(Vector::from([0.]).norm(), 0., max_relative = 0.00001);
        assert_relative_eq!(Vector::from([1.]).norm(), 1., max_relative = 0.00001);
        assert_relative_eq!(Vector::from([0., 0.]).norm(), 0., max_relative = 0.00001);
        assert_relative_eq!(Vector::from([1., 0.]).norm(), 1., max_relative = 0.00001);
        assert_relative_eq!(Vector::from([2., 1.]).norm(), 2.236_068, max_relative = 0.00001);
        assert_relative_eq!(Vector::from([4., 2.]).norm(), 4.472_136, max_relative = 0.00001);
        assert_relative_eq!(Vector::from([-4., -2.]).norm(), 4.472_136, max_relative = 0.00001);
        assert_eq!(Vector::from([0., 0.]).norm_inf()?, 0.);
        assert_eq!(Vector::from([1., 0.]).norm_inf()?, 1.);
        assert_eq!(Vector::from([2., 1.]).norm_inf()?, 2.);
        assert_eq!(Vector::from([4., 2.]).norm_inf()?, 4.);
        assert_eq!(Vector::from([-4., -2.]).norm_inf()?, 4.);


        Ok(())
    }

    #[test]
    fn test_cos_angle() -> Result<(), VectorError>
    {

        let a1 = Vector::from([1., 0.]);
        let a2 = Vector::from([1., 0.]);
        let b1 = Vector::from([1., 0.]);
        let b2 = Vector::from([0., 1.]);
        let c1 = Vector::from([-1., 1.]);
        let c2 = Vector::from([ 1., -1.]);
        let d1 = Vector::from([2., 1.]);
        let d2 = Vector::from([4., 2.]);
        let e1 = Vector::from([1., 2., 3.]);
        let e2 = Vector::from([4., 5., 6.]);

        assert_eq!(round_to(a1.angle_cos(&a2)?, 1.), 1.);
        assert_eq!(round_to(b1.angle_cos(&b2)?, 1.), 0.);
        assert_eq!(round_to(c1.angle_cos(&c2)?, 1.), -1.);
        assert_eq!(round_to(d1.angle_cos(&d2)?, 1.), 1.);
        assert_eq!(e1.angle_cos(&e2)?, 0.974_631_9);

        // eval sheet tests
        assert_eq!(round_to(Vector::from([1., 0.]).angle_cos(&Vector::from([0., 1.]))?, 1.), 0.);
        assert_relative_eq!(Vector::from([8., 7.]).angle_cos(&Vector::from([3., 2.]))?, 0.991_454_3, max_relative = 0.00001);
        assert_eq!(round_to(Vector::from([1., 1.]).angle_cos(&Vector::from([1., 1.]))?, 1.), 1.);
        assert_relative_eq!(Vector::from([4., 2.]).angle_cos(&Vector::from([1., 1.]))?, 0.948_683_3, max_relative = 0.00001);
        assert_relative_eq!(Vector::from([-7., 3.]).angle_cos(&Vector::from([6., 4.]))?, -0.546_267_8, max_relative = 0.00001);

        Ok(())
    }

    #[test]
    fn test_cross_product() -> Result<(), VectorError>
    {
        let a1 = Vector::from([0., 0., 1.]);
        let a2 = Vector::from([1., 0., 0.]);
        let b1 = Vector::from([1., 2., 3.]);
        let b2 = Vector::from([4., 5., 6.]);
        let c1 = Vector::from([4., 2., -3.]);
        let c2 = Vector::from([-2., -5., 16.]);

        assert_eq!(a1.cross_product(&a2)?.store, [0., 1., 0.]);
        assert_eq!(b1.cross_product(&b2)?.store, [-3., 6., -3.]);
        assert_eq!(c1.cross_product(&c2)?.store, [17., -58., -16.]);

        // eval sheet tests
        assert_eq!(Vector::from([0, 0, 0]).cross_product(&Vector::from([0, 0, 0]))?.store, [0, 0, 0]);
        assert_eq!(Vector::from([1, 0, 0]).cross_product(&Vector::from([0, 0, 0]))?.store, [0, 0, 0]);
        assert_eq!(Vector::from([1, 0, 0]).cross_product(&Vector::from([0, 1, 0]))?.store, [0, 0, 1]);
        assert_eq!(Vector::from([8, 7, -4]).cross_product(&Vector::from([3, 2, 1]))?.store, [15, -20, -5]);
        assert_eq!(Vector::from([1, 1, 1]).cross_product(&Vector::from([0, 0, 0]))?.store, [0, 0, 0]);
        assert_eq!(Vector::from([1, 1, 1]).cross_product(&Vector::from([1, 1, 1]))?.store, [0, 0, 0]);

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

        // eval sheet tests
        assert_eq!(Matrix::from([[0, 0], [0, 0]]).trace()?, 0);
        assert_eq!(Matrix::from([[1, 0], [0, 1]]).trace()?, 2);
        assert_eq!(Matrix::from([[1, 2], [3, 4]]).trace()?, 5);
        assert_eq!(Matrix::from([[8, -7], [4, 2]]).trace()?, 10);
        assert_eq!(Matrix::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).trace()?, 3);

        Ok(())
    }

    #[test]
    fn test_transpose() -> Result<(), MatrixError>
    {
        let mut u = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);
        let u_t = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);
        let mut v: Matrix<f64> = Matrix::from([
            [2., -5., 0.],
            [4., 3., 7.],
            [-2., 3., 4.],
            ]);
        let v_t: Matrix<f64> = Matrix::from([
            [2., 4., -2.],
            [-5., 3., 3.],
            [0., 7., 4.],
            ]);
        let mut w = Matrix::from([
            [-2., -8., 4.],
            [1., -23., 4.],
            [0., 6., 4.],
            ]);
        let w_t = Matrix::from([
            [-2., 1., 0.],
            [-8., -23., 6.],
            [4., 4., 4.],
            ]);
        
        assert_eq!(u.transpose(), u_t);
        assert_eq!(v.transpose(), v_t);
        assert_eq!(w.transpose(), w_t);

        // eval sheet tests
        assert_eq!(Matrix::from([[0, 0], [0, 0]]).transpose().store, [[0, 0], [0, 0]]);
        assert_eq!(Matrix::from([[1, 0], [0, 1]]).transpose().store, [[1, 0], [0, 1]]);
        assert_eq!(Matrix::from([[1, 2], [3, 4]]).transpose().store, [[1, 3], [2, 4]]);
        assert_eq!(Matrix::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).transpose().store, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        assert_eq!(Matrix::from([[1, 2], [3, 4], [5, 6]]).transpose().store, [[1, 3, 5], [2, 4, 6]]);


        Ok(())
    }

    #[test]
    fn test_row_echelon() -> Result<(), ()>
    {
        let mut u = Matrix::from([
            [8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.],
        ]);
        let u_ref = Matrix::from([
            [1.0, 0.625, 0.0, 0.0, -12.1666667],
            [0.0, 0.0, 1.0, 0.0, -3.6666667],
            [0.0, 0.0, 0.0, 1.0, 29.5 ],
        ]);

        for (row_u, row_u_ref) in u.row_echelon().store.iter().zip(u_ref.store.iter()) {
            for (elem_u, elem_u_ref) in row_u.iter().zip(row_u_ref.iter()) {
                assert_relative_eq!(elem_u, elem_u_ref, max_relative = 0.00001);
            }
        }

        // eval sheet tests
        assert_eq!(Matrix::from([[0, 0], [0, 0]]).row_echelon().store, [[0, 0], [0, 0]]);
        assert_eq!(Matrix::from([[1, 0], [0, 1]]).row_echelon().store, [[1, 0], [0, 1]]);
        assert_eq!(Matrix::from([[0, 1], [1, 0]]).row_echelon().store, [[1, 0], [0, 1]]);
        assert_eq!(Matrix::from([[4., 2.], [2., 1.]]).row_echelon().store, [[1., 0.5], [0., 0.]]);
        assert_eq!(Matrix::from([[-7, 2], [4, 8]]).row_echelon().store, [[1, 0], [0, 1]]);
        assert_eq!(Matrix::from([[1, 2], [4, 8]]).row_echelon().store, [[1, 2], [0, 0]]);
        
        Ok(())
    }

    fn test_determinant() -> Result<(), MatrixError>
    {
        let a = Matrix::from([
            [ 1., -1.],
            [-1., 1.],
            ]);
        let b = Matrix::from([
            [2., 0., 0.],
            [0., 2., 0.],
            [0., 0., 2.],
            ]);
        let c = Matrix::from([
            [8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.],
            ]);
        let d = Matrix::from([
            [ 8., 5., -2., 4.],
            [ 4., 2.5, 20., 4.],
            [ 8., 5., 1., 4.],
            [28., -4., 17., 1.],
            ]);

        assert_eq!(a.determinant()?, 0.0);
        assert_eq!(b.determinant()?, 8.0);
        assert_eq!(c.determinant()?, -174.0);
        assert_eq!(d.determinant()?, 1032.0);
        
        Ok(())
    }

    #[test]
    fn test_inverse() -> Result<(), MatrixError>
    {
        let mut a = Matrix::from([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            ]);
        let a_inv = Matrix::from([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            ]);
        let mut b = Matrix::from([
            [2., 0., 0.],
            [0., 2., 0.],
            [0., 0., 2.],
            ]);
        let b_inv = Matrix::from([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            ]);
        let mut c = Matrix::from([
            [8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.],
            ]);
        let c_inv = Matrix::from([
            [0.649425287, 0.097701149, -0.655172414],
            [-0.781609195, -0.126436782, 0.965517241],
            [0.143678161, 0.074712644, -0.206896552],
        ]);

        for (a_row, a_inv_row) in a.inverse()?.store.iter().zip(a_inv.store.iter()) {
            for (e, ei) in a_row.iter().zip(a_inv_row.iter()) {
                assert_relative_eq!(e, ei, max_relative = 0.00001);
            }
        }

        for (b_row, b_inv_row) in b.inverse()?.store.iter().zip(b_inv.store.iter()) {
            for (e, ei) in b_row.iter().zip(b_inv_row.iter()) {
                assert_relative_eq!(e, ei, max_relative = 0.00001);
            }
        }

        for (c_row, c_inv_row) in c.inverse()?.store.iter().zip(c_inv.store.iter()) {
            for (e, ei) in c_row.iter().zip(c_inv_row.iter()) {
                assert_relative_eq!(e, ei, max_relative = 0.00001);
            }
        }

        // eval sheet tests
        assert_eq!(Matrix::from([[1, 0], [0, 1]]).inverse()?.store, [[1, 0], [0, 1]]);
        assert_eq!(Matrix::from([[2., 0.], [0., 2.]]).inverse()?.store, [[0.5, 0.], [0., 0.5]]);
        assert_eq!(Matrix::from([[0.5, 0.], [0., 0.5]]).inverse()?.store, [[2., 0.], [0., 2.]]);
        assert_eq!(Matrix::from([[0, 1], [1, 0]]).inverse()?.store, [[0, 1], [1, 0]]);
        assert_eq!(Matrix::from([[1., 2.], [3., 4.]]).inverse()?.store, [[-2., 1.], [1.5, -0.5]]);
        assert_eq!(Matrix::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).inverse()?.store, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        
        Ok(())
    }

    #[test]
    fn test_rank() -> Result<(), ()>
    {
        let a = Matrix::from([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            ]);
        let b = Matrix::from([
            [ 1., 2., 0., 0.],
            [ 2., 4., 0., 0.],
            [-1., 2., 1., 1.],
            ]);
        let c = Matrix::from([
            [ 8., 5., -2.],
            [ 4., 7., 20.],
            [ 7., 6., 1.],
            [21., 18., 7.],
            ]);

        assert_eq!(a.rank(), 3);
        assert_eq!(b.rank(), 2);
        assert_eq!(c.rank(), 3);

        // eval sheet tests
        assert_eq!(Matrix::from([[0, 0], [0, 0]]).rank(), 0);
        assert_eq!(Matrix::from([[1, 0], [0, 1]]).rank(), 2);
        assert_eq!(Matrix::from([[2, 0], [0, 2]]).rank(), 2);
        assert_eq!(Matrix::from([[1, 1], [1, 1]]).rank(), 1);
        assert_eq!(Matrix::from([[0, 1], [1, 0]]).rank(), 2);
        assert_eq!(Matrix::from([[1, 2], [3, 4]]).rank(), 2);
        assert_eq!(Matrix::from([[-7, 5], [4, 6]]).rank(), 2);
        assert_eq!(Matrix::from([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).rank(), 3);

        Ok(())
    }

}
