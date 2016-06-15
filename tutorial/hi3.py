import numpy
import random

def randomUnknownArray(arrayLength, unknownRatio ):

    unknownCount= numpy.int( arrayLength*unknownRatio)
    
    x  =  random.randint(0,arrayLength-unknownCount)
    arr=numpy.ones(arrayLength, dtype="float")    
    for x in range(x, x+unknownCount):
        arr[x] = 0
    return arr

for i in range(0 ,  10):
    a=randomUnknownArray(9,i*0.1)
    print a        


c=range(0,5)
d=c*2
e=numpy.reshape(numpy.asarray(c, dtype="uint8"),[5]);
print  c
print  d
print  e
print  e*-1+255

m = numpy.array([[1,2],[3,4]], int)
print m
print numpy.rot90(m) 
print numpy.rot90(m, 2)

f = numpy.array([10,20,30,40], int)
g=numpy.reshape(numpy.asarray(f, dtype="uint8"),[2,2]);
print f
print numpy.reshape(numpy.asarray(numpy.rot90(g), dtype="uint8"),[len(f)]) 
print numpy.reshape(numpy.asarray(numpy.rot90(g,1), dtype="uint8"),[len(f)])
print numpy.reshape(numpy.asarray(numpy.rot90(g,2), dtype="uint8"),[len(f)])
print len(f), len(g)