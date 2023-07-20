using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

include( "../helpers/sample_algs.jl" )

# 2x2 output matrix
Ai = [1 2;
     3 4]

Af64 = Float64.(Ai)
Af32 = Float32.(Ai)


# Test function is just a pairwise product
FMM.@generate_fastmatmul( Test2x2x2, "macromatmulnum" )
FMM.@generate_fastmatmul( Test2x2x2, "macromatmulabf", AbstractFloat )
FMM.@generate_fastmatmul( Test2x2x2, "macromatmulf64", Float64 )
FMM.@generate_fastmatmul( Test2x2x2, "macromatmulf32", Float32 )
FMM.@generate_fastmatmul( Test2x2x2, "macromatmuli", Int )

# These combinations shouldn't work
@test_throws MethodError macromatmulabf(Ai, Ai)
@test_throws MethodError macromatmulf64(Ai, Ai)
@test_throws MethodError macromatmulf32(Af64, Af64)
@test_throws MethodError macromatmuli(Af64, Af64)

# These should work (default to Number type)
@test macromatmulnum(Ai, Ai) == Ai.*Ai
@test macromatmulnum(Af64, Af64) ≈ Af64.*Af64
@test macromatmulnum(Af32, Af32) ≈ Af32.*Af32

# These should work (AbstractFloat type)
@test macromatmulabf(Af64, Af64) ≈ Af64.*Af64
@test macromatmulabf(Af32, Af32) ≈ Af32.*Af32

# These should work (highly specialized version)
@test macromatmuli(Ai, Ai) == Ai.*Ai
@test macromatmulf64(Af64, Af64) ≈ Af64.*Af64
@test macromatmulf32(Af32, Af32) ≈ Af32.*Af32
