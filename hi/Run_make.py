import Image
import numpy
import math
#import inpaint.MarkerMaker
from inpaint.MarkerMaker import MarkerMaker
from inpaint.PatchMaker import PatchMaker

marker = MarkerMaker()
marker.saveRandomMarker(1)    

patchMaker = PatchMaker()
imgPath = "image/golf.png"
patchMaker.setImagePath(imgPath)
patchMaker.saveRandomPatch(1)