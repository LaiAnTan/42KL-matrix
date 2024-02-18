use std::fmt::Display;

use crate::matrix::Matrix;
use crate::vector::Vector;

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

impl<K> Display for Matrix<K>
where
    K: Display + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        let mut iter = self.store[..self.store.len() - 1].iter().peekable();

        write!(f, "[")?;

        while let Some(row) = iter.next() // while let does not move iter
        {
            write!(f, "[")?;
            for value in row
            {
                write!(f, "{}, ", &value)?;
            }
            write!(f, "{}]", row.last().unwrap())?;
            
            if iter.peek().is_some()
            {
                write!(f, ",")?;
            }
        }
        Ok(())
    }
}