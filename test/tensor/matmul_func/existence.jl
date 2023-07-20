using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "../helpers/sample_algs.jl" )

# Test generating a single method with Float64 matrix type
FMM.@generate_fastmatmul( Test2x2x2, "testmatmul1", Float64 )

m1 = methods( testmatmul1 )
@test length(m1) >= 1

m2 = methods( testmatmul1, (Matrix{Float64}, Matrix{Float64}) )
@test length(m2) == 1

m3 = methods( testmatmul1, (Matrix{Float32}, Matrix{Float32}) )
@test length(m3) == 0

# Test generating a Float32 function as well now (should not replace the old one)
FMM.@generate_fastmatmul( Test2x2x2, "testmatmul1", Float32 )

m4 = methods( testmatmul1 )
@test length(m4) >= 1

m5 = methods( testmatmul1, (Matrix{Float64}, Matrix{Float64}) )
@test length(m5) == 1

m6 = methods( testmatmul1, (Matrix{Float32}, Matrix{Float32}) )
@test length(m6) == 1
