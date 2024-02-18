use std::ops::Add;

use crate::vector::Vector;
use crate::vector::errors::VecSizeDiffError;

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

impl<K> Vector<K>
where
    K: Clone,
{
    fn add(&mut self, v: &Vector<K>) -> Result<Self, VecSizeDiffError>
    where
        K: Add<Output = K> + Clone + Copy,
    {
        if self.size != v.size
        {
            return Err(VecSizeDiffError);
        }
        
        self.store = self.store.iter().zip(v.store.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(self.clone())
    }
}