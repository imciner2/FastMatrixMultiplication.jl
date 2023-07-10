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
