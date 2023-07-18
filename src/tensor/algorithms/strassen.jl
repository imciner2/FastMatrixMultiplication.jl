# SPDX-License-Identifier: MIT

Strassen2x2x2 = TensorFMMAlgorithm(
    # U
    [1  0  1  0  1 -1  0;
     0  0  0  0  1  0  1;
     0  1  0  0  0  1  0;
     1  1  0  1  0  0 -1],
    # V
    [1  1  0 -1  0  1  0;
     0  0  1  0  0  1  0;
     0  0  0  1  0  0  1;
     1  0 -1  0  1  0  1],
     # W
    [1  0  0  1 -1  0  1;
     0  0  1  0  1  0  0;
     0  1  0  1  0  0  0;
     1 -1  1  0  0  1  0],
    # Matrix dimensions
    2, 2, 2)
