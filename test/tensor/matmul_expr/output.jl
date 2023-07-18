using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "sample_algs.jl" )

# 2x2 output matrix
A = [1 2;
     3 4]
B = A

# This algorithm is equivalent to an elementwise product
eval( FMM._generate_fastmatmul_expr( Test2x2x2, Float64 ) )

# Verify partial products are mapped to correct elements of the output
@test C[1,1] ≈ m1
@test C[1,2] ≈ m2
@test C[2,1] ≈ m3
@test C[2,2] ≈ m4

# This algorithm is equivalent to an elementwise product, but flipped
eval( FMM._generate_fastmatmul_expr( Test2x2x2mix, Float64 ) )

# Verify partial products are using the correct columns of the tensors
@test C[1,1] ≈ m4
@test C[1,2] ≈ m3
@test C[2,1] ≈ m2
@test C[2,2] ≈ m1
