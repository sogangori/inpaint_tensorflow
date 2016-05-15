import numpy

rgb = numpy.array([
[5, 10, 15],
[20, 25, 30],
[35, 40, 45]
])

a = numpy.array([
[1,0, 0],
[0,1,0],
[0,1,1]
])

#print(rgb)
#print(a)

IMAGE_SIZE=3
#data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
rgb1 = rgb.reshape(IMAGE_SIZE*IMAGE_SIZE)
a1=a.reshape(IMAGE_SIZE*IMAGE_SIZE)

#numpy.zeros(shape=(num_images,), dtype=numpy.int64)
rgba = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*2))
rgba[0:IMAGE_SIZE*IMAGE_SIZE]=rgb1
rgba[IMAGE_SIZE*IMAGE_SIZE:]=a1
print( rgba)

