using FastMatrixMultiplication
using LinearAlgebra
using Test

const FMM = FastMatrixMultiplication

U1 = [1  0  0  0;
      0  1  0  0;
      0  0  1  0;
      0  0  0  1]

U2 = [1  0  1  0  1 -1  0  0;
      0  0  0  0  1  0  1  1;
      0  1  0  0  0  1  0  0;
      1  1  0  1  0  0 -1  0;
      0  0  1  0  1  0  1  0;
      1 -1  0  1  0  0  0  0]

U3 = [1  0  0  1 -1  0  1;
      0  0  1  0  1  0  0;
      0  1  0  1  0  0  0;
      1 -1  1  0  0  1  0;
      0  0  1  0  1  0  1;
      1 -1  0  1  0  0  0;
      1  0  0  1 -1  0  1;
      0  0  1  0  1  0  0;
      0  1  0  1  0  0  0]

##########################################################
# Test that passing the wrong n, m, p value flags an error
##########################################################
# Test that U checking works
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1, U1, U1, 3, 2, 2 ) # Incorrect n
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1, U1, U1, 2, 3, 2 ) # Incorrect m

# Test that V checking works
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1, U1, U1, 2, 2, 3 ) # Incorrect p
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U2, U1, U1, 2, 3, 2 ) # Incorrect m

# Test that W checking works
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1, U2, U1, 2, 2, 3 ) # Incorrect p
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U2, U1, U1, 2, 2, 3 ) # Incorrect n


##########################################################
# Test that passing tensor elements with different lengths flags an error
##########################################################
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1[:,1:3], U1,        U1,        2, 2, 2 )
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1,        U1[:,1:3], U1,        2, 2, 2 )
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1,        U1,        U1[:,1:3], 2, 2, 2 )
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1,        U1[:,1:3], U1[:,1:3], 2, 2, 2 )
@test_throws DimensionMismatch FMM.TensorFMMAlgorithm( U1[:,1:3], U1[:,1:3], U1,        2, 2, 2 )


##########################################################
# Test that one gets created
##########################################################
V = 2*U1
W = 3*U1
alg1 = FMM.TensorFMMAlgorithm( U1, V, W, 2, 2, 2 )
@test alg1.U == U1
@test alg1.V == V
@test alg1.W == W
@test alg1.n == 2
@test alg1.m == 2
@test alg1.p == 2
