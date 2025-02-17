
mod linalg;

// need to implement add/sub/mul assign
use crate::linalg::{Matrix, errors, alg::projection};

fn main() -> Result<(), errors::MatrixError> // main must return result for error propogation to be legal
{
    let mut u = Matrix::from([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        ]);
    println!("{}", u.inverse()?);
    // [1.0, 0.0, 0.0]
    // [0.0, 1.0, 0.0]
    // [0.0, 0.0, 1.0]
    let mut u = Matrix::from([
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.],
        ]);
    println!("{}", u.inverse()?);
    // [0.5, 0.0, 0.0]
    // [0.0, 0.5, 0.0]
    // [0.0, 0.0, 0.5]
    let mut u = Matrix::from([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
        ]);
    println!("{}", u.inverse()?);
    // [0.649425287, 0.097701149, -0.655172414]
    // [-0.781609195, -0.126436782, 0.965517241]
    // [0.143678161, 0.074712644, -0.206896552]
    
    let proj: Matrix<f32> = projection(90f32, 4f32/3f32, 5f32, 100f32).unwrap();

    println!("{}", proj);

    Ok(())
}