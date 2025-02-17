
mod matrix;

// need to implement add/sub/mul assign
use crate::matrix::{Matrix, errors};

fn main() -> Result<(), errors::MatrixError> // main must return result for error propogation to be legal
{
    let mut u = Matrix::from([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
        ]);
    println!("{}", u.inverse()?);
    // [0.649425287, 0.097701149, -0.655172414]
    // [-0.781609195, -0.126436782, 0.965517241]
    // [0.143678161, 0.074712644, -0.206896552]

    Ok(())
}