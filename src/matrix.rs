pub mod utils;

pub struct Matrix<K>
{
    pub rows: usize,
    pub columns: usize,
    pub store: Vec< Vec<K> >
}
