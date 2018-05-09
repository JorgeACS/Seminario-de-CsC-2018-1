import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>
__global__ void matrixTranspose (float *a, float *b) {
  int i = blockIdx.x * blockDim.x; 
  int j =  threadIdx.x;

  int i2 = blockIdx.x;
  int j2 = threadIdx.x*blockDim.x;
  b[i+j] = a[j2+i2];
 
}

""")

matrixTranspose = mod.get_function("matrixTranspose")
x = 5
y = 5
a = numpy.random.randint(10,size=(x,y)).astype(numpy.float32)

b = numpy.zeros_like(a).astype(numpy.float32)

blockSize = x
if x < y:
	blockSize = y
matrixTranspose(
        drv.In(a), drv.Out(b),
        block=(x,1,1), grid=(y,1,1))

print(numpy.transpose(a))
print('')
print(b)
print(numpy.equal(numpy.transpose(a),b))



