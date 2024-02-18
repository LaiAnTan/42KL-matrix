mod vector;
mod matrix;

// need to implement add/sub/mul assign
use crate::vector::Vector;

fn main() -> Result<(), vector::errors::VecSizeDiffError> // main must return result for error propogation to be legal
{
    let m = Vector::<i32> {size: 3, store: vec![1, 2, 3]};

    let test = Vector::<i32>::from([2, 3, 4]);

    println!("Vector: {}, Size: {}", m, m.size());
    println!("Test: {}, Size: {}", test, test.size());
    println!("Out: {}", m * 3);
    Ok(())
}