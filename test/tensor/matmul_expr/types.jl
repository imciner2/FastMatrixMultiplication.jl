using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "../helpers/sample_algs.jl" )

A = fill( 1, (2,2) )
B = fill( 2, (2,2) )

######################################################
# Test element types
######################################################

# Default is Float64 elements
T = Float64
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )
@test eltype(C) == Float64

T = Float32
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )
@test eltype(C) == Float32

T = Int
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )
@test eltype(C) == Int

T = Int64
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )
@test eltype(C) == Int64
