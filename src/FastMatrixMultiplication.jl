# SPDX-License-Identifier: MIT

module FastMatrixMultiplication

include( "tensor/types.jl" )

# Functions to analyze the algorithm
include( "tensor/analysis.jl" )

# Generate Julia functions to perform fast matrix multiplication
include( "tensor/functions.jl" )

# Strassen's methods
include( "tensor/algorithms/strassen.jl" )

end
