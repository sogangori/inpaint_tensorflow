import Image
import numpy
from inpaint.MarkerMaker import MarkerMaker
from inpaint.PatchMaker import PatchMaker

def save_image( npdata, outfilename ) :   
    
    img = Image.fromarray( npdata, "RGB")
    img.save( outfilename )
    print ("marking patch saved ",outfilename)

markingPatchFolder = "./markingPatch/"
dstImageName = "marking"+"Patch12_233.png"  

patchMaker = PatchMaker()
imgPath = "image/golf.png"
patchMaker.setImagePath(imgPath)
patch = patchMaker.makeRandomPatch()

markerMaker = MarkerMaker()
maker = markerMaker.makeRandomMarker()  

IMAGE_SIZE=9
CHANNEL=3
markingPatch = numpy.zeros(shape=(IMAGE_SIZE,IMAGE_SIZE,(CHANNEL)), dtype=numpy.ubyte)

for y in range(0,IMAGE_SIZE):
    for x in range(0,IMAGE_SIZE):
        markingPatch[y][x]=patch[y][x]*(maker[y][x]/255)

print("training image is ", patchMaker.imgPath , " (x,y)",patchMaker.x1, patchMaker.y1)
#print ("patch",patch)
save_image(markingPatch, markingPatchFolder+dstImageName)
  