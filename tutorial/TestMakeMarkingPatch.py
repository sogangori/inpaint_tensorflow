import numpy

IMAGE_SIZE=2
CHANNEL=3
#rgba = numpy.random.rand(IMAGE_SIZE*IMAGE_SIZE*(CHANNEL+1))
rgba = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*(CHANNEL+1)), dtype=numpy.ubyte)
markingPatch = numpy.ones(shape=(IMAGE_SIZE,IMAGE_SIZE,(CHANNEL)), dtype=numpy.ubyte)
alpha = numpy.ones(shape=(IMAGE_SIZE,IMAGE_SIZE,(1)), dtype=numpy.ubyte)
print(markingPatch)
print(alpha)

print("---")


rgb = markingPatch.reshape( numpy.size(markingPatch))
a = alpha.reshape( numpy.size(alpha))
print("rgba",rgba)
print("rgb",rgb)
print("a",a)

rgba[0:numpy.size(rgb) ] = rgb
rgba[numpy.size(rgb) : ] = a
print("rgba",rgba)

# rgb1 = rgb.reshape(IMAGE_SIZE*IMAGE_SIZE)
# a1 = a.reshape(IMAGE_SIZE)
# print rgb1
# 
# print len(rgb1)
# for i in range(0,len(rgb1)):
#     rgb1[i]=rgb1[i]*a1[i%3]
#     
# print rgb1

#numpy.zeros(shape=(num_images,), dtype=numpy.int64)
#rgba = numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*2))
#rgba[0:IMAGE_SIZE*IMAGE_SIZE]=rgb1
#rgba[IMAGE_SIZE*IMAGE_SIZE:]=a1
#print( rgba)

