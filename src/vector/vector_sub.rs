use std::ops::Sub;

use crate::vector::Vector;
use crate::vector::errors::VecSizeDiffError;

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

impl<K> Vector<K>
where
    K: Clone,
{
    fn sub(&mut self, v: &Vector<K>) -> Result<Self, VecSizeDiffError>
    where
        K: Sub<Output = K> + Clone + Copy,
    {
        if self.size != v.size
        {
            return Err(VecSizeDiffError);
        }
        
        self.store = self.store.iter().zip(v.store.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(self.clone())
    }
}