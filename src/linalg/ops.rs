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

    pub fn scl(&mut self, a: K) -> Result<Self, MatrixError>
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
    pub fn scl(&mut self, a: K) -> Result<Self, VectorError>
    where
        K: Mul<Output = K> + Clone + Copy,
    {
        self.store = self.store.iter().map(|&x| x * a)
                .collect();

        Ok(self.clone())
    }
}

// --- ex07: Linear Map, Matrix Multiplication ---

// overload: mat * vec
impl<K> Mul<Vector<K>> for Matrix<K>
where
    K: Clone + MulAdd<Output = K> + Zero + Copy
{
    type Output = Vector<K>;

    fn mul(self, rhs: Vector<K>) -> Self::Output
    {
        assert_eq!(self.columns, rhs.size, "Number of columns of matrix must be equal to vector size to multiply");
        
        Vector::from(self.store.iter()
        .map(|row| {
            let row_vec: Vector<K> = Vector::from(row.clone());
            rhs.dot(&row_vec).unwrap() // this wont panic, the error is already handled above
        })
        .collect::<Vec<K>>())
    }
}

// overload: mat * mat
impl<K> Mul<Matrix<K>> for Matrix<K>
where
    K: Clone + MulAdd<Output = K> + Zero + Copy
{
    type Output = Matrix<K>;

    fn mul(self, rhs: Matrix<K>) -> Self::Output
    {
        assert_eq!(self.columns, rhs.rows, "Number of columns of lhs must be equal to number of rows of rhs to multiply");

        Self
        {
            columns: self.columns,
            rows: self.rows,
            store: self.store.iter()
                .map(|row| {
                    (0..rhs.columns).map(|index| {
                        Vector::from(row.clone()).dot(&Vector::from(rhs.col(index).clone())).unwrap() // safe unwrap, error already handled above
                    }).collect()
                }).collect()
        }
    }
} 

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

    #[test]
    fn test_vec_add() -> Result<(), VectorError>
    {
        let a= Vector::from([1., 2., 3.]);
        let b = Vector::from([-1., 40., 3.]);
        let mut c = a.clone();
        let d = b.clone();

        assert_eq!((a + b).store, [0., 42., 6.]);
        assert_eq!(c.add(&d)?.store, [0., 42., 6.]);

        Ok(())
    }

    #[test]
    fn test_vec_sub() -> Result<(), VectorError>
    {
        let a= Vector::from([1., 2., 3.]);
        let b = Vector::from([-1., 40., 3.]);
        let mut c = a.clone();
        let d = b.clone();

        assert_eq!((a - b).store, [2.0, -38.0, 0.0]);
        assert_eq!(c.sub(&d)?.store, [2.0, -38.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_vec_scl() -> Result<(), VectorError>
    {
        let a= Vector::from([0., 2., 3.]);
        let mut b = Vector::from([0., 2., 3.]);

        assert_eq!((a * 3.).store, [0., 6.0, 9.0]);
        assert_eq!(b.scl(3.)?.store, [0., 6.0, 9.0]);

        Ok(())
    }

    #[test]
    fn test_mat_add() -> Result<(), MatrixError>
    {
        let a = Matrix::from([
            [1., 2.],
            [0., -3.],
            ]);
        let b = Matrix::from([
            [1., 40.],
            [-2., 3.],
            ]);
            
        assert_eq!(a.clone().add(&b.clone())?.store, [
            [2., 42.],
            [-2., 0.],
        ]);
        assert_eq!((a + b).store, [
            [2., 42.],
            [-2., 0.],
        ]);

        Ok(())
    }

    #[test]
    fn test_mat_sub() -> Result<(), MatrixError>
    {
        let a = Matrix::from([
            [1., 2.],
            [0., -3.],
            ]);
        let b = Matrix::from([
            [1., 40.],
            [-2., 3.],
            ]);
            
        assert_eq!(a.clone().sub(&b.clone())?.store, [
            [0., -38.],
            [2., -6.],
        ]);
        assert_eq!((a - b).store, [
            [0., -38.],
            [2., -6.],
        ]);

        Ok(())
    }

    #[test]
    fn test_mat_scl() -> Result<(), MatrixError>
    {
        let mut a = Matrix::from([
            [1., 2.],
            [0., -3.],
            ]);

        assert_eq!((a.clone() * 3.).store, [
            [3., 6.],
            [0., -9.],
        ]);
        assert_eq!(a.scl(3.)?.store, [
            [3., 6.],
            [0., -9.],
        ]);

        Ok(())
    }

    #[test]
    fn test_vec_mul() -> Result<(), errors::MatrixError>
    {
        let mut a = Matrix::from([
            [1., 0.],
            [0., 1.],
            ]);
        let b = Vector::from([4., 2.]);
        let mut c = Matrix::from([
            [2., 0.],
            [0., 2.],
            ]);
        let d = Vector::from([4., 2.]);
        let mut e = Matrix::from([
        [2., -2.],
        [-2., 2.],
        ]);
        let f = Vector::from([4., 2.]);

        assert_eq!(a.mul_vec(&b)?.store, [4., 2.]);
        assert_eq!(c.mul_vec(&d)?.store, [8., 4.]);
        assert_eq!((e.clone() * f.clone()).store, [4., -4.]);
        assert_eq!(e.mul_vec(&f)?.store, [4., -4.]);

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
        let c = Matrix::from([
            [2., 1.],
            [4., 2.],
            ]);
        let mut d = Matrix::from([
            [3., -5.],
            [6., 8.],
            ]);
        let e = Matrix::from([
            [2., 1.],
            [4., 2.],
            ]);
        let mut f = Matrix::from([
            [1., 2., 3.],
            [4., 5., 6.],
        ]);
        let g = Matrix::from([
            [10., 11.],
            [20., 21.],
            [30., 31.],
        ]);

        assert_eq!(a.mul_mat(&b)?.store, [[1., 0.], [0., 1.]]);
        assert_eq!(a.mul_mat(&c)?.store, [[2., 1.], [4., 2.]]);
        assert_eq!(d.mul_mat(&e)?.store, [[-14., -7.], [44., 22.]]);
        assert_eq!((f.clone() * g.clone()).store, [[140., 146.], [320., 335.]]);
        assert_eq!(f.mul_mat(&g)?.store, [[140., 146.], [320., 335.]]);
        
        Ok(())
    }
}