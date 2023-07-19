# SPDX-License-Identifier: MIT

export TensorFMMAlgorithm

"""
    TensorFMMAlgorithm

Tensor representation of a fast matrix multiplication algorithm.
"""
struct TensorFMMAlgorithm{UT, VT, WT}
    # Tensor slice representing the values of A to select for each multiplication
    U::UT

    # Tensor slice representing the values of B to select for each multiplication
    V::VT

    # Tensor slice representing the partial products to add together to form the final result
    W::WT

    # Matrix dimensions
    # Multiply an nxm matrix with an mxp matrix to produce an nxp matrix
    n::Int
    m::Int
    p::Int

    function TensorFMMAlgorithm(U, V, W, n::Int, m::Int, p::Int)
        (Aelem, Arank) = size(U)
        (Belem, Brank) = size(V)
        (Celem, Crank) = size(W)

        if Aelem != n*m
            throw( DimensionMismatch( "U must have nm rows" ) )
        end

        if Belem != m*p
            throw( DimensionMismatch( "V must have mp rows" ) )
        end

        if Celem != n*p
            throw( DimensionMismatch( "W must have np rows" ) )
        end

        # All matrices must have the same number of columns to ensure they all operate on the
        # same number of partial products
        if ( Arank != Brank ) || ( Arank != Crank ) || ( Brank != Crank )
            throw( DimensionMismatch( "U, V, W must have the same number of columns" ) )
        end

        new{typeof(U), typeof(V), typeof(W)}(U, V, W, n, m, p)
    end
end
