using FastMatrixMultiplication
using SafeTestsets
using Test

@testset "FastMatrixMultiplication.jl" begin
    @testset "" begin
        @safetestset "Adder chain" begin include( "tensor/adder_chain.jl" ) end
    end
end
