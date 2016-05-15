import Image
import numpy
from inpaint.MarkingPatchMaker import MarkingPatchMaker

def save_image_rgba( npdata, outfilename ) :   
    
    img = Image.fromarray( npdata, "RGBA")
    img.save( outfilename )
    print ("marking patch saved ",outfilename)
    
def save_image( npdata, outfilename ) :   
    
    img = Image.fromarray( npdata, "RGB")
    img.save( outfilename )
    print ("marking patch saved ",outfilename)

markingPatchFolder = "./markingPatch/"
dstImageName = "marking"+"Patch12_233.png"  

markerMaker = MarkingPatchMaker()
markingPatch=markerMaker.makeRandomMarkingPatch()
#print ("markingPatch",markingPatch)
print ('markingPatch size',numpy.size(markingPatch,0),numpy.size(markingPatch,1),numpy.size(markingPatch,2) )

save_image(markingPatch, markingPatchFolder+dstImageName)

#1 dim array
rgbaPatch=markerMaker.makeRandomMarkingPatchRGBA()
#print ("markingPatch",markingPatch)
print ('rgbaPatch size',numpy.size(rgbaPatch))
print 9*9*3,  9*9*4

trainingSet=markerMaker.makeRandomMarkingPatchRGBA_count(2)
print ("trainingSet",trainingSet)
print ('rgbaPatch size',numpy.size(trainingSet))
print 9*9*3,  9*9*4  