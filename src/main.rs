
mod linalg;

// need to implement add/sub/mul assign
use crate::linalg::{Matrix, errors};

fn main() -> Result<(), errors::MatrixError> // main must return result for error propogation to be legal
{
    let mut u = Matrix::from([
        [8., 5., -2., 4., 28.],
        [4., 2.5, 20., 4., -4.],
        [8., 5., 1., 4., 17.],
        ]);

    println!("{:?}", u.row_echelon());
    println!("Rank: {:?}", u.rank());
    
    Ok(())
}