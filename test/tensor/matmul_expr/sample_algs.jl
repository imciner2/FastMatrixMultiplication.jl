# Algorithms for matrices of different sizes
# Testnxmxp is multiplying an nxm matrix by an mxp matrix to get an nxp matrix

# Behaves like an element-wise product
Test2x2x2 = FMM.TensorFMMAlgorithm(
    # U
    [1  0  0  0;
     0  1  0  0;
     0  0  1  0;
     0  0  0  1],
    # V
    [1  0  0  0;
     0  1  0  0;
     0  0  1  0;
     0  0  0  1],
    # W
    [1  0  0  0;
     0  1  0  0;
     0  0  1  0;
     0  0  0  1],
    # Matrix dimensions
    2, 2, 2)

# Behaves like an element-wise product, but reversed
Test2x2x2mix = FMM.TensorFMMAlgorithm(
    # U
    [1  0  0  0;
     0  1  0  0;
     0  0  1  0;
     0  0  0  1],
    # V
    [1  0  0  0;
     0  1  0  0;
     0  0  1  0;
     0  0  0  1],
    # W
    [0  0  0  1;
     0  0  1  0;
     0  1  0  0;
     1  0  0  0],
    # Matrix dimensions
    2, 2, 2)

# Not actually a valid algorithm
Test3x2x2 = FMM.TensorFMMAlgorithm(
    # U
    [1  0  1  0  1 -1  0  0;
     0  0  0  0  1  0  1  1;
     0  1  0  0  0  1  0  0;
     1  1  0  1  0  0 -1  0;
     0  0  1  0  1  0  1  0;
     1 -1  0  1  0  0  0  0],
    # V
    [1  1  0 -1  0  1  0  0;
     0  0  1  0  0  1  0 -1;
     0  0  0  1  0  0  1  0;
     1  0 -1  0  1  0  1  1],
    # W
    [1  0  0  1 -1  0  1  1;
     0  0  1  0  1  0  0  1;
     0  1  0  1  0  0  0  1;
     1 -1  1  0  0  1  0  1;
     0  1  0  1  0  0  0  1;
     1 -1  1  0  0  1  0  1],
    # Matrix dimensions
    3, 2, 2)

# Not actually a valid algorithm
Test3x2x3 = FMM.TensorFMMAlgorithm(
    # U
    [1  0  1  0  1 -1  0;
     0  0  0  0  1  0  1;
     0  1  0  0  0  1  0;
     1  1  0  1  0  0 -1;
     0  0  1  0  1  0  1;
     1 -1  0  1  0  0  0],
    # V
    [1  1  0 -1  0  1  0;
     0  0  1  0  0  1  0;
     0  0  0  1  0  0  1;
     1  0 -1  0  1  0  1;
     0  0  0  1  0  0  1;
     1  0 -1  0  1  0  1],
     # W
    [1  0  0  1 -1  0  1;
     0  0  1  0  1  0  0;
     0  1  0  1  0  0  0;
     1 -1  1  0  0  1  0;
     0  0  1  0  1  0  1;
     1 -1  0  1  0  0  0;
     1  0  0  1 -1  0  1;
     0  0  1  0  1  0  0;
     0  1  0  1  0  0  0],
    # Matrix dimensions
    3, 2, 3)

# Not actually a valid algorithm
Test2x2x3 = FMM.TensorFMMAlgorithm(
    # U
    [1  0  1  0  1 -1  0;
     0  0  0  0  1  0  1;
     0  1  0  0  0  1  0;
     1  1  0  1  0  0 -1],
    # V
    [1  1  0 -1  0  1  0;
     0  0  1  0  0  1  0;
     0  0  0  1  0  0  1;
     1  0 -1  0  1  0  1;
     0  0  0  1  0  0  1;
     1  0 -1  0  1  0  1],
     # W
    [1  0  0  1 -1  0  1;
     0  0  1  0  1  0  0;
     0  1  0  1  0  0  0;
     1 -1  1  0  0  1  0;
     0  0  1  0  1  0  1;
     1 -1  0  1  0  0  0],
    # Matrix dimensions
    2, 2, 3)
