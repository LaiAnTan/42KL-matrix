#![allow(dead_code)]

pub mod ops;
pub mod utils;
pub mod errors;
pub mod alg;


#[derive(Debug, Clone)]
pub struct Matrix<K: Clone>
{
    pub rows: usize,
    pub columns: usize,
    pub store: Vec< Vec<K> > // vector containing vectors(rows)
}

#[derive(Debug, Clone)]
pub struct Vector<K: Clone>
{
    pub size: usize,
    pub store: Vec<K>
}
