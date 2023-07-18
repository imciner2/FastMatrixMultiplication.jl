using FastMatrixMultiplication
using SafeTestsets
using Test

@testset "FastMatrixMultiplication.jl" begin
    @testset "Function formation" begin
        @safetestset "Adder chain" begin include( "tensor/adder_chain.jl" ) end
        @safetestset "Partial product" begin include( "tensor/partial_product.jl" ) end

        @testset "Matmul Expression" begin
            @safetestset "Sizes" begin include( "tensor/matmul_expr/sizes.jl" ) end
            @safetestset "Output types" begin include( "tensor/matmul_expr/types.jl" ) end
            @safetestset "Partial products" begin include( "tensor/matmul_expr/partial_products.jl" ) end
            @safetestset "Output results" begin include( "tensor/matmul_expr/output.jl" ) end
        end
    end
end
