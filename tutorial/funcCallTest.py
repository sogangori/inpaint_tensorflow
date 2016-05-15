import functions
import numpy
def hi():
    
    print "hi"
    

b=hi()

a = numpy.zeros(shape=(1,5), dtype=numpy.ubyte)
        
for y in range(0,1):
    for x in range(0,5):
        a[y][x]=x
        print x
        
print a
print a[0][2:]