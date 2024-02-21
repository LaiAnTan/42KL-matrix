use std::ops::{Add, Sub, Mul};

use crate::linalg::{Matrix, Vector};
use crate::linalg::errors::{MatrixError, VectorError};

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

// unit tests
#[cfg(test)]
mod tests
{
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}