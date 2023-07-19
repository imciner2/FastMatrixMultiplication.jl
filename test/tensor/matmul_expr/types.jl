using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "sample_algs.jl" )

A = fill( 1, (2,2) )
B = fill( 2, (2,2) )

######################################################
# Test element types
######################################################

# Default is Float64 elements
eval( FMM._generate_fastmatmul_expr( Test2x2x2 ) )
@test eltype(C) == Float64

eval( FMM._generate_fastmatmul_expr( Test2x2x2, etype = Float64 ) )
@test eltype(C) == Float64

eval( FMM._generate_fastmatmul_expr( Test2x2x2, etype = Float32 ) )
@test eltype(C) == Float32

eval( FMM._generate_fastmatmul_expr( Test2x2x2, etype = Int ) )
@test eltype(C) == Int

eval( FMM._generate_fastmatmul_expr( Test2x2x2, etype = Int64 ) )
@test eltype(C) == Int64
