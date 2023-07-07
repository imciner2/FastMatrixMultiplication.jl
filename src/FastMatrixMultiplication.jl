# SPDX-License-Identifier: MIT

module FastMatrixMultiplication

include( "tensor/types.jl" )

# Generate Julia functions to perform fast matrix multiplication
include( "tensor/functions.jl")

# Strassen's methods
include( "tensor/algorithms/strassen.jl" )

end
