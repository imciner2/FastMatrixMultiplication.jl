using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "../helpers/sample_algs.jl" )

# 2x2 output matrix
A = [1 2;
     3 4]
B = A

# This algorithm is equivalent to an elementwise product
T = Float64
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )

# Should generate 4 partial products
@test @isdefined m1
@test @isdefined m2
@test @isdefined m3
@test @isdefined m4

# Verify partial products are using the correct columns of the tensors
res = A.*A
@test m1 ≈ res[1, 1]
@test m2 ≈ res[1, 2]
@test m3 ≈ res[2, 1]
@test m4 ≈ res[2, 2]


# 3x2 output matrix
A = fill( 1, (3,2) )
B = fill( 2, (2,2) )

eval( FMM._generate_fastmatmul_expr( Test3x2x2 ) )

# Should generate 8 partial products
@test @isdefined m1
@test @isdefined m2
@test @isdefined m3
@test @isdefined m4
@test @isdefined m5
@test @isdefined m6
@test @isdefined m7
@test @isdefined m8
