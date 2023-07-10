import LinearAlgebra: rank

"""
    rank( tfmm::TensorFMMAlgorithm )
"""
rank( tfmm::TensorFMMAlgorithm ) = size( tfmm.W, 2 )


"""
    _add_count( tfmm::TensorFMMAlgorithm )


"""
function _add_count( tfmm::TensorFMMAlgorithm )
    # The number of additions in each product term is the number of 1s in each column
    # minus 1.
    Uadds = sum( max.(sum(abs, tfmm.U, dims=1 ) .- 1, 0) )
    Vadds = sum( max.(sum(abs, tfmm.V, dims=1 ) .- 1, 0) )

    # The number of additions in the final
    Wadds = sum( max.(sum(abs, tfmm.W, dims=1 ) .- 1, 0) )

    return Uadds + Vadds + Wadds
end


"""
    opcount( tfmm::TensorFMMAlgorithm )


"""
function opcount( tfmm::TensorFMMAlgorithm )
    return ( mul = rank( tfmm ), add = _add_count( tfmm ) )
end
