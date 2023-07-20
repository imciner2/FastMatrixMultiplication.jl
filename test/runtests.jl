using FastMatrixMultiplication
using SafeTestsets
using Test

@testset "FastMatrixMultiplication.jl" begin

    @testset "Tensor algorithm type" begin
        @safetestset "Type creation" begin include( "tensor/type/creation.jl" ) end
    end

    @testset "Function formation" begin
        @safetestset "Adder chain" begin include( "tensor/adder_chain.jl" ) end
        @safetestset "Partial product" begin include( "tensor/partial_product.jl" ) end

        @testset "Matmul Expression" begin
            @safetestset "Sizes" begin include( "tensor/matmul_expr/sizes.jl" ) end
            @safetestset "Output types" begin include( "tensor/matmul_expr/types.jl" ) end
            @safetestset "Partial products" begin include( "tensor/matmul_expr/partial_products.jl" ) end
            @safetestset "Output results" begin include( "tensor/matmul_expr/output.jl" ) end
        end

        @testset "Function wrapping" begin
            @safetestset "Existence" begin include( "tensor/matmul_func/existence.jl" ) end
            @safetestset "Usage" begin include( "tensor/matmul_func/usage.jl" ) end
        end
    end
end
