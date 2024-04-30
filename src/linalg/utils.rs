use std::fmt::Display;
use std::fmt::Result;
use core::ops::{Index, IndexMut};

use num::traits::{MulAdd, Float};


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

impl<K> IndexMut<usize> for Matrix<K>
where
    K: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.store[index]
    }
}

// PartialEq trait for comparing Matrices
impl<K> PartialEq for Matrix<K>
where
    K: Clone + std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        
        if self.shape() != other.shape()
        {
            return false
        }

        self.store.iter().zip(other.store.iter())
            .all(|(row_self, row_other)| row_self == row_other)

    }
}

impl<K> Matrix<K>
where
    K: Clone,
{
    pub fn shape(&self) -> (usize, usize)
    {
        (self.rows, self.columns)
    }

    pub fn is_square(&self) -> bool
    {
        self.rows == self.columns
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

    pub fn col(&self, index: usize) -> Vec<K>
    {
        self.store.iter().map(|row| row[index].clone()).collect()
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


// from trait for Vec<K> to vector 
impl<K> From<Vec<K>> for Vector<K>
where
    K: Clone,
{
    fn from(value: Vec<K>) -> Self {
        Vector::<K> {size: value.len(), store: value}
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

// PartialEq trait for comparing Vectors
impl<K> PartialEq for Vector<K>
where
    K: Clone + std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        
        if self.size() != other.size()
        {
            return false
        }

        self.store == other.store
    }
}

// generic fused-multiply-accumulate for Vector<K> and Matrix<K> using num::traits::MulAdd
impl<K> MulAdd<K, Vector<K>> for Vector<K>
where
    K: Float,
{
    type Output = Self;

    fn mul_add(self, a: K, b: Self) -> Self
    {
        self * a + b
    }
}

impl<K> MulAdd<K, Matrix<K>> for Matrix<K>
where
    K: Float,
{
    type Output = Self;

    fn mul_add(self, a: K, b: Self) -> Self
    {
        self * a + b
    }
}