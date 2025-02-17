# 42KL-matrix

A linear algebra module written in Rust.

Note: This module is not meant to be performant, and are just an exercise to discover linear algebra & the algorithms computers use to perform linear algebra calculations.

## Dependencies

All dependencies listed [here](/Cargo.toml).

- approx: for floating point approx assertion macros (relative_eq!)
- num: for generic number trait bounds (Float),generic MulAdd trait, complex number type support

## Installation

Add to cargo project

```bash
cargo add matrix42
```

## Functions

Supported Functions:

- Addition, Subtraction, Scaling for Vector and Matrices
- Vector Matrix multiplication, Matrix multiplication
- Linear Combination
- Linear Interpolation (lerp)
- Dot Product
- Manhattan Norm (L-1), Euclidean Norm (L-2), Infinity Norm (L-inf)
- Cosine
- Cross Product
- Matrix Trace
- Matrix Transpose
- Matrix Row - Echelon Form
- Matrix Inverse
- Matrix Determinant
- Matrix Rank
