import scipy.ndimage as ndi
import scipy
import numpy
import Image
import math
from trainEdge.CannyMaker import CannyMaker
from inpaint.GTK_windows import GTK_Window
import pygtk
pygtk.require('2.0')
import gtk

f = '../image/golf.png'
f = '../image/New_york_retribution.png'
folder = "canny/"
patchSize=25
showSize=90
channel=2
kind=7
count = kind*8
gtkWin=GTK_Window()
cannyMaker= CannyMaker()


set = cannyMaker.generatePatchSet(f, count, patchSize)
[set,DamagedSet] = cannyMaker.generatePatchSet2mix(f, count, patchSize)
for i in range(0,count):
               
    gtkWin.ShowGrayImage(set[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)    
    gtkWin.ShowGrayImage(DamagedSet[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)
    if i!=0 and (i+1)%kind==0 :
        gtkWin.AddOffsetX(0)
        gtkWin.AddOffsetY(showSize)
gtk.main()
