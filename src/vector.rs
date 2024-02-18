pub mod vector_add;
pub mod vector_scl;
pub mod vector_sub;
pub mod utils;
pub mod errors;

// main struct
#[derive(Clone)]
pub struct Vector<K: Clone>
{
    pub size: usize,
    pub store: Vec<K>
}
