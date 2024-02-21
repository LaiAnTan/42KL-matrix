
mod linalg;

// need to implement add/sub/mul assign
use crate::linalg::{Matrix, Vector, errors, alg};

fn main() -> Result<(), errors::VectorError> // main must return result for error propogation to be legal
{
    let m = Vector::<i32> {size: 3, store: vec![2, -2, 3]};



    let test = Vector::<i32>::from([2, 3, 4]);

    println!("Vector: {}", m);
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

    let v1 = Vector::from([2., 1.]);
    let v2 = Vector::from([4., 2.]);

    let res = alg::lerp(v1, v2, 0.3);

    println!("{}", res.unwrap());

    let u = Vector::from([1., 2., 3.]);

    println!("{} {} {}", u.norm()?, u.norm_1()?, u.norm_inf()?);
    
    Ok(())
}