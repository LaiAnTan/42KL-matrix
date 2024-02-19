use std::fmt::Display;
use std::fmt::Result;
use core::ops::{Mul, Add, Index, IndexMut};

use crate::linalg::{Matrix, Vector};

impl<K> Display for Matrix<K>
where
    K: Display + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result
    {
        for row in &self.store
        {
            write!(f, "[")?;
            for value in row[..row.len() - 1].iter()
            {
                write!(f, "{}, ", &value)?;
            }
            writeln!(f, "{}]", row.last().unwrap())?;
        }
        Ok(())
    }
}

impl<K, const N: usize, const M: usize> From<[[K; N]; M]> for Matrix<K>
where
    K: Clone,
{
    fn from(value: [[K; N]; M]) -> Self {

        Matrix::<K>
        {
            rows: M,
            columns: N,
            store: value.iter().map(|row| row.to_vec()).collect()
        }
    }
}

// index assignment operator
impl<K> Index<usize> for Matrix<K>
where
    K: Clone,
{
    type Output = Vec<K>;

    fn index(&self, index: usize) -> &Vec<K> {
        &self.store[index]
    }
}


impl<K: Clone> Matrix<K>
where
    K: Clone,
{
    pub fn shape(&self) -> (usize, usize)
    {
        (self.rows, self.columns)
    }

    pub fn to_vector(&self) -> Vector<K>
    {
        let mut store: Vec<K> = Vec::new();

        if self.columns > 1
        {
            panic!("Matrix converting to Vector must have only 1 column")
        }

        for row in &self.store
        {
            store.push(row[0].clone())
        }

        Vector { size: self.rows, store}
    }
}

// Vector Utilities

// display trait for printing
impl<K> Display for Vector<K>
where
    K: Display + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        write!(f, "[")?;
        for value in self.store[..self.store.len() - 1].iter()
        {
            write!(f, "{}, ", &value)?;
        }

        write!(f, "{}]", self.store.last().unwrap())?;
        Ok(())
    }
}

// from trait for array to vector 
impl<K, const N: usize> From<[K; N]> for Vector<K>
where
    K: Clone,
{
    fn from(value: [K; N]) -> Self {
        Vector::<K> {size: value.len(), store: value.to_vec()}
    }
}

// index overload
impl<K> Index<usize> for Vector<K>
where
    K: Clone,
{
    type Output = K;

    fn index(&self, i: usize) -> &K {
        &self.store[i]
    }
}

impl<K> IndexMut<usize> for Vector<K>
where
    K: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.store[index]
    }
}

impl<K> Vector<K>
where
    K: Clone
{
    // vector size
    pub fn size(&self) -> usize
    {
        self.size
    }

    pub fn len(&self) -> f64
    where
        K: Mul<Output = K> + Add<Output = K> + std::convert::Into<f64> + std::iter::Sum + Clone
    {
        (self.store.iter()
            .map(|x| Into::<f64>::into(x.clone()).powi(2))
            .sum::<f64>()
        ).sqrt()
    }

    // vector to matrix
    pub fn to_matrix(&self) -> Matrix<K>
    {
        let mut store: Vec< Vec<K> > = Vec::new();

        for value in &self.store
        {
            store.push(vec![value.clone()])
        }

        Matrix { rows: self.size, columns: 1, store}
    }

}

// generic fused-multiply-accumulate because rust only has f32 and f64 variants
pub trait MulAdd
{
    fn mul_add(&self, a: Self, b: Self) -> Self;
}

impl<K> MulAdd for K
where
    K: Add<Output = K> + Mul<Output = K> + Clone + Copy,
{
    fn mul_add(&self, a: Self, b: Self) -> Self
    {
        *self * a + b
    }
}