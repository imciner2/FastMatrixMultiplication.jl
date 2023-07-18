using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "sample_algs.jl" )

A = fill( 1, (2,2) )
B = fill( 2, (2,2) )

eval( FMM._generate_fastmatmul_expr( Test2x2x2, Float64 ) )
@test eltype(C) == Float64

eval( FMM._generate_fastmatmul_expr( Test2x2x2, Float32 ) )
@test eltype(C) == Float32

eval( FMM._generate_fastmatmul_expr( Test2x2x2, Int ) )
@test eltype(C) == Int

eval( FMM._generate_fastmatmul_expr( Test2x2x2, Int64 ) )
@test eltype(C) == Int64
