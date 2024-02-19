
mod linalg;

// need to implement add/sub/mul assign
use crate::linalg::Vector;
use crate::linalg::Matrix;

fn main() -> Result<(), linalg::errors::VectorError> // main must return result for error propogation to be legal
{
    let m = Vector::<i32> {size: 3, store: vec![2, -2, 3]};



    let test = Vector::<i32>::from([2, 3, 4]);

    println!("Vector: {}, Length: {}", m, m.len());
    println!("Test: {}, Size: {}", test, test.size());
    println!("Out: {}", m + test);

    let mut u = Matrix::<f32>::from([
        [1., 2.],
        [3., 4.],
        ]);
    let v = Matrix::<f32>::from([
        [7., 4.],
        [-2., 2.]
        ]);
    
    println!("u: {:?}", u);
    println!("v: {:?}", v);

    let _ = u.add(&v);
    println!("{}", u);
    // [-6.0, -2.0]
    // [5.0, 2.0]

    println!("Accessing element 1, 1: {}", u[1][1]);
    
    Ok(())
}