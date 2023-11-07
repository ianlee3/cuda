# Remember that we can't use numpy math function on the GPU...
import numpy
import math
from numba import cuda
# Consider modifying the 3 values in this cell to optimize host <-> device memory movement
normalized = cuda.device_array(shape=(n,),dtype='float32')
weighted = cuda.device_array(shape=(n,),dtype='float32')
activated = cuda.device_array(shape=(n,),dtype='float32')

# Modify these 3 function calls to run on the GPU
@vectorize(['float32(float32,)'], target='cuda')
def normalize(grayscales):
    return grayscales / 255

@vectorize(['float32(float32,float32)'], target='cuda')
def weigh(values, weights):
    return values * weights

@vectorize(['float32(float32,)'], target='cuda')
def activate(values):
    return ( math.exp(values) - math.exp(-values) ) / ( math.exp(values) + math.exp(-values) )