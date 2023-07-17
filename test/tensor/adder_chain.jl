using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

##############################################
# Test building adder chains for array values
##############################################
x = [1 2 3 4 5]

@testset "Empty values" begin
    # Test addition with positives
    u = [0 0 0 0 0]
    exp = FMM._adder_chain( u, "x", true )
    @test exp == :()

    exp = FMM._adder_chain( u, "x", false )
    @test exp == :()
end

@testset "Array adder chain - one" begin
    # Test addition with positives
    u = [1  1  1  1  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0  1  0  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with negatives
    u = [-1 -1 -1 -1 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0 -1  0 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with both positives and negatives
    u = [-1  1 -1  1 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [1 -1  1 -1  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  1 -1 -1  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0 -1 -1  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  1 -1  0  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0 -1  1 -1  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )
end

@testset "Array adder chain - other" begin
    # Test addition with positives
    u = [2  4  6  8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0  6  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  1  6  0  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with negatives
    u = [-2 -4 -6 -8 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-2  0 -6  0 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-2  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0 -6  0 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with both positives and negatives
    u = [-2  4 -6  8 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2 -4  6 -8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  4 -6 -8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0 -6 -8  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  4 -6  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0 -4  6 -8  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  4 -6  1 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )
end

#######################################################
# Test building adder chains for individual variables
#######################################################
# Define these at global scope because eval only operates at global scope
x1 = 1
x2 = 2
x3 = 3
x4 = 4
x5 = 5

@testset "Variable adder chain - one" begin
    # Test addition with positives
    u = [1  1  1  1  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0  1  0  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0  0  0  0]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with negatives
    u = [-1 -1 -1 -1 -1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0 -1  0 -1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0  0  0  0]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0 -1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with both positives and negatives
    u = [-1  1 -1  1 -1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [1 -1  1 -1  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  1 -1 -1  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [1  0 -1 -1  0]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  1 -1  0  1]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )

    u = [0 -1  1 -1  0]
    exp = FMM._adder_chain( u, "x", false )
    @test eval( exp ) ≈ dot( u, x )
end


@testset "Variable adder chain - other" begin
    # Test addition with positives
    u = [2  4  6  8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0  6  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  1  6  0  1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with negatives
    u = [-2 -4 -6 -8 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-2  0 -6  0 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-2  0  0  0  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  0  0  0 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  0 -6  0 -1]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    # Test addition with both positives and negatives
    u = [-2  4 -6  8 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2 -4  6 -8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  4 -6 -8  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [2  0 -6 -8  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0  4 -6  0  9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [0 -4  6 -8  0]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )

    u = [-1  4 -6  1 -9]
    exp = FMM._adder_chain( u, "x", true )
    @test eval( exp ) ≈ dot( u, x )
end
