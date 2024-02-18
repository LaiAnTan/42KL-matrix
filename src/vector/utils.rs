use std::fmt::Display;

use crate::vector::Vector;
use crate::matrix::Matrix;

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

impl<K, const N: usize> From<[K; N]> for Vector<K>
where
    K: Clone,
{
    fn from(value: [K; N]) -> Self {
        Vector::<K> {size: value.len(), store: value.to_vec()}
    }
}

impl<K> Vector<K>
where
    K: Clone
{
    pub fn size(&self) -> usize
    {
        self.size
    }

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