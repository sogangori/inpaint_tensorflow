import scipy.ndimage as ndi
import scipy
import numpy
from PIL import Image
import math
import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'inpaint'))
from CannyMaker import CannyMaker
from GTK_windows import GTK_Window
import pygtk
pygtk.require('2.0')
import gtk

f = '../image/golf.png'
f = '../image/New_york_retribution.png'
folder = "canny/"
patchSize=25
showSize=90
count = 40
gtkWin=GTK_Window()
cannyMaker= CannyMaker()

channel=2
set = cannyMaker.generatePatchSet(f, count, patchSize)
[set,DamagedSet] = cannyMaker.generatePatchSet3(f, count, patchSize)
for i in range(0,count):
    for c in range(0,channel):
            
        gtkWin.ShowGrayImage(set[i][c][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)    
        gtkWin.ShowGrayImage(DamagedSet[i][c][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)
        if i!=0 and i%5==0 :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showSize)
gtk.main()
