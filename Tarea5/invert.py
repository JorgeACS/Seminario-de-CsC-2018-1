import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
	#include <stdio.h>
__global__ void invert (float *a, float *b) {
  int length = blockDim.x;
  int j = threadIdx.x;
  int bValue = length-j-1;
  if(bValue < 0){
  	bValue = -bValue;
  }
  b[bValue] = a[j];

}

""")

invert = mod.get_function("invert")
x = 20
y = 20
a = numpy.random.randint(10,size=x*y).astype(numpy.float32)
b = numpy.random.randint(10,size=x*y).astype(numpy.float32)

dest = numpy.zeros_like(a).astype(numpy.float32)
invert(
        drv.In(a), drv.Out(b),
        block=(x*y,1,1), grid=(1,1,1))
drv.Context.synchronize()

print(a)
print(b)
print(numpy.flip(a,0))
print(numpy.equal(numpy.flip(a,0),b))