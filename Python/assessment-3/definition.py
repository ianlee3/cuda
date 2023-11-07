import numpy as np
from numba import cuda, types
@cuda.jit
def mm_shared(a, b, c):
    x, y = cuda.grid(2)
    sum = 0

    # `a_cache` and `b_cache` are already correctly defined
    # block (32 x 32)
    a_cache = cuda.shared.array(block_size, types.int32)
    b_cache = cuda.shared.array(block_size, types.int32)

    # NxN threads per block 
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    # M/N blocks per grid.
    gsize = cuda.gridDim.x
    
    # Use each thread to populate one element each a_cache and b_cache
    for i in range(gsize):
        # Preload data into shared memory
        a_cache[tx][ty] = a[x,ty + i*block_size[0]]
        b_cache[tx][ty] = b[tx + i*block_size[0],y]
        # Wait until all threads finish preloading
        cuda.syncthreads()
        for j in range(block_size[0]):
            # Calculate the `sum` value correctly using values from the cache 
            sum += a_cache[tx][j] * b_cache[j][ty]
        # Wait until all threads finish computing
        cuda.syncthreads()
    c[x][y] = sum