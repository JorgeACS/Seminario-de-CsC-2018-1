import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
	#include <stdio.h>
__global__ void matrixSum (float *a, float *b, float *dest) {
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  printf(" %d ",i+j);
  dest[i+j] = a[i+j] + b[i+j];

}

""")

matrixSum = mod.get_function("matrixSum")
x = 5
y = 5
a = numpy.random.randint(10,size=(x,y)).astype(numpy.float32)
b = numpy.random.randint(10,size=(x,y)).astype(numpy.float32)

dest = numpy.zeros_like(a).astype(numpy.float32)
matrixSum(
        drv.In(a), drv.In(b),drv.Out(dest),
        block=(x,1,1), grid=(y,1,1))
drv.Context.synchronize()

print(a)
print(b)
print(numpy.add(a,b))
print(dest)
print(numpy.equal(dest,numpy.add(a,b)))