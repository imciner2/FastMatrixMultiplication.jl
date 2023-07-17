using FastMatrixMultiplication
using SafeTestsets
using Test

@testset "FastMatrixMultiplication.jl" begin
    @testset "Function formation" begin
        @safetestset "Adder chain" begin include( "tensor/adder_chain.jl" ) end
        @safetestset "Partial product" begin include( "tensor/partial_product.jl" ) end
    end
end
