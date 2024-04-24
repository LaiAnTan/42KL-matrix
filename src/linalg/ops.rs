use std::ops::{Add, Sub, Mul};

use crate::linalg::{Matrix, Vector};
use super::errors::{MatrixError, VectorError};


use num::traits::{Zero, MulAdd};

// --- ex00: Add, Subtract, Scale ---

// overload unary addition: m1 + m2
impl<K> Add< Matrix<K> > for Matrix<K>
where
    K: Add<Output = K> + Clone + Copy,
{
    type Output = Matrix<K>;

    fn add(self, rhs: Self) -> Matrix<K>
    {
        assert_eq!((self.rows, self.columns), (rhs.rows, rhs.columns), "Matrix must be same shape to add");

        Self
        { 
            rows: self.rows,
            columns: self.columns,
            store: self.store.iter().zip(rhs.store.iter())
                    .map(|(row_1, row_2)| row_1.iter().zip(row_2.iter())
                        .map(|(&a, &b)| a + b).collect())
                    .collect()
        }
    }
}

// overload unary subtraction: m1 - m2
impl<K> Sub< Matrix<K> > for Matrix<K>
where
    K: Sub<Output = K> + Clone + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self
    {
        assert_eq!((self.rows, self.columns), (rhs.rows, rhs.columns), "Matrix must be same shape to subtract");

        Self
        { 
            rows: self.rows,
            columns: self.columns,
            store: self.store.iter().zip(rhs.store.iter())
                    .map(|(row_1, row_2)| row_1.iter().zip(row_2.iter())
                        .map(|(&a, &b)| a - b).collect())
                    .collect()
        }
    }
}

// overload scalar multiplication: m1 * 3 (no bidirectional scaling)
impl<K> Mul<K> for Matrix<K>
where
    K: Mul<Output = K> + Clone + Copy,
{
    type Output = Self;

    fn mul(self, rhs: K) -> Self
    {
        Self
        { 
            rows: self.rows,
            columns: self.columns,
            store: self.store.iter()
                    .map(|row| row.iter().map(|&x| x * rhs).collect())
                    .collect()
        }
    }
}

// function form
impl<K> Matrix<K>
where
    K: Clone,
{
    pub fn add(&mut self, m: &Matrix<K>) -> Result<Self, MatrixError>
    where
        K: Add<Output = K> + Clone + Copy,
    {
        if self.rows != m.rows || self.columns != m.columns
        {
            return Err(MatrixError)
        }

        self.store = self.store.iter().zip(m.store.iter())
            .map(|(row_1, row_2)| row_1.iter().zip(row_2.iter())
                .map(|(&a, &b)| a + b).collect())
            .collect();

        Ok(self.clone())
    }

    pub fn sub(&mut self, m: &Matrix<K>) -> Result<Self, MatrixError>
    where
        K: Sub<Output = K> + Clone + Copy
    {
        if self.rows != m.rows || self.columns != m.columns
        {
            return Err(MatrixError)
        }

        self.store = self.store.iter().zip(m.store.iter())
            .map(|(row_1, row_2)| row_1.iter().zip(row_2.iter())
                .map(|(&a, &b)| a - b).collect())
            .collect();

        Ok(self.clone())
    }

    pub fn scl(&mut self, a: K) -> Result<Self, ()>
    where
        K: Mul<Output = K> + Clone + Copy,
    {
        self.store = self.store.iter()
            .map(|row| row.iter().map(|&x| x * a).collect())
            .collect();

        Ok(self.clone())
    }

}

// Vector Operations

// overload unary addition: v1 + v2
impl<K> Add< Vector<K> > for Vector<K>
where
    K: Add<Output = K> + Clone + Copy,
{
    type Output = Vector<K>;

    fn add(self, rhs: Self) -> Vector<K>
    {
        assert_eq!(self.size, rhs.size, "Vectors must be same size to add");

        Self
        { 
            size: self.size,
            store: self.store.iter().zip(rhs.store.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        }
    }
}

// overload unary subtraction: v1 - v2
impl<K> Sub< Vector<K> > for Vector<K>
where
    K: Sub<Output = K> + Clone + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self
    {
        assert_eq!(self.size, rhs.size, "Vectors must be same size to subtract");

        Self
        { 
            size: self.size,
            store: self.store.iter().zip(rhs.store.iter())
                .map(|(&a, &b)| a - b)
                .collect()
        }
    }
}

// overload scalar multiplication: v1 * 3 (no bidirectional scaling)
impl<K> Mul<K> for Vector<K>
where
    K: Mul<Output = K> + Clone + Copy,
{
    type Output = Self;

    fn mul(self, rhs: K) -> Self
    {
        Self
        { 
            size: self.size,
            store: self.store.iter().map(|&x| x * rhs)
                .collect()
        }
    }
}

// function form
impl<K> Vector<K>
where
    K: Clone,
{
    // add
    pub fn add(&mut self, v: &Vector<K>) -> Result<Self, VectorError>
    where
        K: Add<Output = K> + Clone + Copy,
    {
        if self.size != v.size
        {
            return Err(VectorError);
        }
        
        self.store = self.store.iter().zip(v.store.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(self.clone())
    }

    // subtract
    pub fn sub(&mut self, v: &Vector<K>) -> Result<Self, VectorError>
    where
        K: Sub<Output = K> + Clone + Copy,
    {
        if self.size != v.size
        {
            return Err(VectorError);
        }
        
        self.store = self.store.iter().zip(v.store.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(self.clone())
    }

    // scale
    pub fn scl(&mut self, a: K)
    where
        K: Mul<Output = K> + Clone + Copy,
    {
        self.store = self.store.iter().map(|&x| x * a)
                .collect()
    }
}

// --- ex07: Linear Map, Matrix Multiplication ---

// impl<K> Mul<Vector<K>> for Matrix<K>
// where
//     K: Clone
// {
//     type Output = Vector<K>;

//     fn mul(self, rhs: Vector<K>) -> Self::Output
//     {

//     }
// }

// impl<K> Mul<Matrix<K>> for Matrix<K>
// where
//     K: Clone
// {
//     type Output = Self;

//     fn mul(self, rhs: Matrix<K>) -> Self::Output
//     {

//     }
// } 



// function form
impl<K> Matrix<K>
where
    K: Clone + Copy + Zero + MulAdd<Output = K>
{
    pub fn mul_vec(&mut self, vec: &Vector<K>) -> Result<Vector<K>, MatrixError>
    where
        K: Mul<Output = K> + Clone + Copy,
    {
        if self.columns != vec.size
        {
            return Err(MatrixError)
        }

        let store: Vec<K> = self.store.iter()
            .map(|row| {

                let row_vec: Vector<K> = Vector::from(row.clone());
                vec.dot(&row_vec).unwrap() // this wont panic, the error is already handled above
            
            })
            .collect();

        Ok(Vector {size: store.len(), store})
    }

    pub fn mul_mat(&mut self, mat: &Matrix<K>) -> Result<Matrix<K>, MatrixError>
    where
        K: Mul<Output = K> + Clone + Copy + std::fmt::Display + std::fmt::Debug,
    {
        if self.columns != mat.rows
        {
            return Err(MatrixError)
        }
        
        // bozo matrix multiplication
        self.store = self.store.iter()
            .map(|row| {
                (0..mat.columns).map(|index| {
                    Vector::from(row.clone())
                        .dot(&Vector::from(mat.col(index).clone()))
                        .unwrap() // safe unwrap, error already handled above
                })
                .collect()

            })
            .collect();

        Ok(Matrix {rows: self.rows, columns: mat.columns, store: self.store.clone()})
    }
}

// unit tests
#[cfg(test)]
mod tests
{
    use crate::linalg::errors::{self, MatrixError, VectorError};

    use super::{Vector, Matrix};
    extern crate approx; // for floating point relative assertions

    //use approx::assert_relative_eq;

    #[test]
    fn test_ops() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn test_vec_mul() -> Result<(), errors::MatrixError>
    {
        let mut u = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);
        let v = Vector::from([4., 2.]);

        assert_eq!(u.mul_vec(&v)?.store, [4., 2.]);

        let mut u = Matrix::from([
            [2., 0.],
            [0., 2.],
            ]);
        let v = Vector::from([4., 2.]);
        
        assert_eq!(u.mul_vec(&v)?.store, [8., 4.]);

        let mut u = Matrix::from([
        [2., -2.],
        [-2., 2.],
        ]);
        let v = Vector::from([4., 2.]);
        
        assert_eq!(u.mul_vec(&v)?.store, [4., -4.]);
        
        Ok(())
    }

    #[test]
    fn test_mat_mul() -> Result<(), errors::MatrixError>
    {
        let mut a = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);
        let b = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);

        assert_eq!(a.mul_mat(&b)?.store, [[1., 0.], [0., 1.]]);

        let c = Matrix::from([
            [2., 1.],
            [4., 2.],
            ]);

        assert_eq!(a.mul_mat(&c)?.store, [[2., 1.], [4., 2.]]);

        let mut d = Matrix::from([
            [3., -5.],
            [6., 8.],
            ]);
        let e = Matrix::from([
            [2., 1.],
            [4., 2.],
            ]);
        
        assert_eq!(d.mul_mat(&e)?.store, [[-14., -7.], [44., 22.]]);
        
        Ok(())
    }
}