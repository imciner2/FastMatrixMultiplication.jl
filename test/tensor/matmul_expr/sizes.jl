using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "sample_algs.jl" )

# 2x2 output matrix
A = fill( 1, (2,2) )
B = fill( 2, (2,2) )

eval( FMM._generate_fastmatmul_expr( Test2x2x2, Float64 ) )
@test size(C) == (2,2)


# 3x2 output matrix
A = fill( 1, (3,2) )
B = fill( 2, (2,2) )

eval( FMM._generate_fastmatmul_expr( Test3x2x2, Float64 ) )
@test size(C) == (3,2)


# 3x3 output matrix
A = fill( 1, (3,2) )
B = fill( 2, (2,3) )

eval( FMM._generate_fastmatmul_expr( Test3x2x3, Float64 ) )
@test size(C) == (3,3)


# 2x3 output matrix
A = fill( 1, (2,2) )
B = fill( 2, (2,3) )

eval( FMM._generate_fastmatmul_expr( Test2x2x3, Float64 ) )
@test size(C) == (2,3)
