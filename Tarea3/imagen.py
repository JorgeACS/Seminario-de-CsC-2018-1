import PIL
from PIL import Image
import time

import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void blackAndWhite (float *input, float *output) {
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  int index = i+j;
  int gray = input[index*3]*0.21f + input[index*3+1]*0.71f + input[index*3+2]*0.07f;
 
 
  output[index*3] = gray;
  output[index*3+1] = gray;
  output[index*3+2] = gray;


}

__global__ void blur (float *input, float *output,int limit) {
  int i = blockIdx.x*blockDim.x;
  int j = threadIdx.x;
  int index = i+j;
  // up-left, up, up-right, left, right, down-left, down, down-right
  int neighbouringIndeces[8] = {
  	  (index-blockDim.x-1)*3, // up-left
  	  (index-blockDim.x)*3, // up
	  (index-blockDim.x+1)*3, // up-right
	  (index-1)*3, // left
	  (index+1)*3, // right
	  (index+blockDim.x-1)*3, //down-left
	  (index+blockDim.x)*3, //down
	  (index+blockDim.x+1)*3 //down-right
  };

  int pixelsFound = 0;
  int averageR = 0;
  int averageG = 0;
  int averageB = 0;
  for( int i = 0; i < 8; i++){
  	int neighbourIndex = neighbouringIndeces[i];
  	if(neighbourIndex < limit*3 && neighbourIndex > 0){
  		averageR+=input[neighbourIndex];
  		averageG+=input[neighbourIndex+1];
  		averageB+=input[neighbourIndex+2];
  		pixelsFound+=1;
  	}
  } 
  output[index*3] = averageR/pixelsFound;
  output[index*3+1] = averageG/pixelsFound;
  output[index*3+2] = averageB/pixelsFound;

  
}

""")

blackAndWhite = mod.get_function("blackAndWhite")
blur = mod.get_function("blur")
inputImage = Image.open("./input-bnw.jpg")
print(inputImage.mode)
inputAsArray = numpy.array(inputImage)
inputAsArray = inputAsArray.astype(numpy.float32)
outputAsArray = numpy.copy(inputAsArray)
# Black and White
blackAndWhite(
        drv.In(inputAsArray),drv.Out(outputAsArray),
        block=(inputImage.size[0],1,1), grid=(inputImage.size[1],1,1))
drv.Context.synchronize()
outputAsArray = (numpy.uint8(outputAsArray))
outputImage = Image.fromarray(outputAsArray,mode ="RGB")
outputImage.save("./output-bnw.jpg")
# Blur
inputImage = Image.open("./input-bnw.jpg")
print(inputImage.mode)
inputAsArray = numpy.array(inputImage)
inputAsArray = inputAsArray.astype(numpy.float32)
outputAsArray = numpy.copy(inputAsArray)
limit = numpy.uint32(inputImage.size[0]*inputImage.size[1])
blur(
        drv.In(inputAsArray),drv.Out(outputAsArray),limit,
        block=(inputImage.size[0],1,1), grid=(inputImage.size[1],1,1))
drv.Context.synchronize()
outputAsArray = (numpy.uint8(outputAsArray))
outputImage = Image.fromarray(outputAsArray,mode ="RGB")
outputImage.save("./output-blur.jpg")