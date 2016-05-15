import Image
import numpy
import math

import pygtk
pygtk.require('2.0')
import gtk

from inpaint.MarkingPatchMaker import MarkingPatchMaker
from inpaint.GTK_windows import GTK_Window

gtkWin=GTK_Window()

trainCount=3
markerMaker = MarkingPatchMaker()        
trainingSet = markerMaker.makeRandomMarkingPatchRGBA_count(trainCount)
patchSet=markerMaker.getPatchSet()  

IMAGE_SIZE=9
CHANNEL=3
imgSize=IMAGE_SIZE*IMAGE_SIZE*CHANNEL
markImgSize=IMAGE_SIZE*IMAGE_SIZE*(CHANNEL+1)

for i in range(0,trainCount):
    onePatch =  numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)    
    oneMarkingPatch =  numpy.zeros(shape=(IMAGE_SIZE*IMAGE_SIZE*CHANNEL), dtype=numpy.ubyte)
    onePatch[:]=patchSet[i*imgSize:i*imgSize+imgSize]
    oneMarkingPatch[:]=trainingSet[i*markImgSize:i*markImgSize+imgSize]    
    
    gtkWin.ShowImage(onePatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),150)
    gtkWin.ShowImage(oneMarkingPatch.reshape(IMAGE_SIZE,IMAGE_SIZE,CHANNEL),150)     

gtk.main()
print 1