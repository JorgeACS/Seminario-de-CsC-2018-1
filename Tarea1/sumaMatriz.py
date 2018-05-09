import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void vectorSum(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] + b[i];
}
""")

vectorSum = mod.get_function("vectorSum")

a = numpy.random.randn(500).astype(numpy.float32)
b = numpy.random.randn(500).astype(numpy.float32)

dest = numpy.zeros_like(a)
vectorSum(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(500,1,1), grid=(1,1))

print(dest)
print(a+b)
print(numpy.equal(dest,a+b))