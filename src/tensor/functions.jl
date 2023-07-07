# SPDX-License-Identifier: MIT

"""
    TensorFMMAlgorithm

Tensor representation of a fast matrix multiplication algorithm.
"""
struct TensorFMMAlgorithm
    # Tensor slice representing the values of A to select for each multiplication
    U::Matrix{Float64}

    # Tensor slice representing the values of B to select for each multiplication
    V::Matrix{Float64}

    # Tensor slice representing the partial products to add together to form the final result
    W::Matrix{Float64}

    # Matrix dimensions
    # Multiply an nxm matrix with an mxp matrix to produce an nxp matrix
    n::Int
    m::Int
    p::Int
end

"""
    TensorFMMAlgorithm( U, W, V )
"""
function TensorFMMAlgorithm( U, W, V )
    # Determine the size of the matrices being multiplied
    n = size( U, 1 )
    m = size( V, 1 )
    p = size( W, 1 )

    return TensorFMMAlgorithm( U, W, V, n, m, p )
end

import LinearAlgebra: rank

"""
    rank( tfmm::TensorFMMAlgorithm )
"""
rank( tfmm::TensorFMMAlgorithm ) = size( tfmm.W, 2 )


"""
    tensor_fmma( tfmm::TensorFMMAlgorithm )
"""
tensor_fmma( tfmm::TensorFMMAlgorithm ) = tensor_fmma( tfmm.U, tfmm.V, tfmm.W )


"""
    tensor_fmma( U, V, W )
"""
function tensor_fmma( U, V, W )
    # Initial part of the function is transforming to row-linear ordering
    init = quote
                Ap = PermutedDimsArray( A, (2, 1) )
                Bp = PermutedDimsArray( B, (2, 1) )
           end

    # Form the partial products
    partial = Vector{Expr}()
    for (i, u) in enumerate( eachcol( U ) )
        Aadder = _adder_chain( u, :Ap )
        Badder = _adder_chain( V[:,i], :Bp )

        mname = Symbol( "m$i" )
        push!( partial, :($mname = $Aadder * $Badder) )
    end

    partials = Expr( :block, partial...)

    Expr( :block, init, partials )
end

function _adder_chain( u, varname::Symbol )
    varScaled = Vector{Expr}()

    for (ind, val) in enumerate( u )
        if !iszero( val )
            if isone( abs( val ) )
                push!( varScaled, :($varname[$ind]) )
            else
                push!( varScaled, :($val * $varname[$ind]) )
            end
        end
    end

    adder = varScaled[1]

    while length( varScaled ) >= 1
        val = popfirst!( varScaled )
        adder = :($adder + $val)
    end

    return adder
end
