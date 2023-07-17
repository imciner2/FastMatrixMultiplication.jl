using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

x = [1 2 3 4 5]
y = reverse(x)

w1 = 1
w2 = 2
w3 = 3
w4 = 4
w5 = 5

z1 = 5
z2 = 4
z3 = 3
z4 = 2
z5 = 1

@testset "Empty partial product" begin
    u = [1  1  1  1  1]
    v = mod.(u.+1, 2)
    Achain = FMM._adder_chain( u, "x", true )
    Bchain = FMM._adder_chain( v, "y", true )

    eval( FMM._partial_product( Achain, Bchain, 1 ) )
    @test iszero( m1 )

    eval( FMM._partial_product( Bchain, Achain, 2) )
    @test iszero( m2 )
end

@testset "Both array" begin
    # Test addition with positives
    u = [1  0  0  1  1]
    v = mod.(u.+1, 2)
    Achain = FMM._adder_chain( u, "x", true )
    Bchain = FMM._adder_chain( v, "y", true )

    eval( FMM._partial_product( Achain, Bchain, 1 ) )
    @test m1 ≈ ( dot(u, x) * dot(v, y) )

    eval( FMM._partial_product( Bchain, Achain, 2 ) )
    @test m2 ≈ ( dot(v, y) * dot(u, x) )
    @test m1 ≈ m2
end

@testset "Both individuals" begin
    # Test addition with positives
    u = [1  0  0  1  1]
    v = mod.(u.+1, 2)
    Achain = FMM._adder_chain( u, "w", false )
    Bchain = FMM._adder_chain( v, "z", false )

    eval( FMM._partial_product( Achain, Bchain, 1 ) )
    @test m1 ≈ ( dot(u, [w1 w2 w3 w4 w5]) * dot(v, [z1 z2 z3 z4 z5]) )

    eval( FMM._partial_product( Bchain, Achain, 2 ) )
    @test m2 ≈ ( dot(v, y) * dot(u, x) )
    @test m1 ≈ m2
end

@testset "One array, one individuals" begin
    # Test addition with positives
    u = [1  0  0  1  1]
    v = mod.(u.+1, 2)
    Achain = FMM._adder_chain( u, "w", false )
    Bchain = FMM._adder_chain( v, "y", true )

    eval( FMM._partial_product( Achain, Bchain, 1 ) )
    @test m1 ≈ ( dot(u, [w1 w2 w3 w4 w5]) * dot(v, y) )

    eval( FMM._partial_product( Bchain, Achain, 2 ) )
    @test m2 ≈ ( dot(v, y) * dot(u, [w1 w2 w3 w4 w5]) )
    @test m1 ≈ m2
end
