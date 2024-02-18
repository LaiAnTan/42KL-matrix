use std::ops::Mul;

use crate::vector::Vector;

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

impl<K> Vector<K>
where
    K: Clone,
{
    fn scl(&mut self, a: K)
    where
        K: Mul<Output = K> + Clone + Copy,
    {
        self.store = self.store.iter().map(|&x| x * a)
                .collect()
    }
}