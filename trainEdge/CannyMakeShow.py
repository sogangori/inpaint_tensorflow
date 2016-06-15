import scipy.ndimage as ndi
import scipy
import numpy
from PIL import Image
import math
from CannyMaker import CannyMaker
import sys,os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'inpaint'))
from GTK_windows import GTK_Window
import pygtk
pygtk.require('2.0')
import gtk

f = '../image/golf.png'
f = '../image/New_york_retribution.png'
f = '../image/New_york1.png'
folder = "canny/"
patchSize=11
showSize=90

kind=4
count = kind*8
gtkWin=GTK_Window()
cannyMaker= CannyMaker()

def basic():
    unknownRatio = 0.4
    [set,DamagedSet] = cannyMaker.generatePatchSetWhatStudy(f, count, patchSize,unknownRatio)
    for i in range(0,count):
                   
        gtkWin.ShowGrayImage(set[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)    
        gtkWin.ShowGrayImage(DamagedSet[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize,1)
        #gtkWin.ShowGrayImage(hintSet[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)
        gtkWin.AddOffsetX(15)
        if i!=0 and (i+1)%kind==0 :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showSize+1)
        

def ch2():
    channel=3
    aug=2
    rotate=4
    count = aug*rotate*4
    unknownRatio=0.5
    [set,DamagedSet] = cannyMaker.generatePatchSetChannel(f, count, patchSize, channel,unknownRatio)    
    for i in range(0,count):
        for c in range(0,channel):                   
            gtkWin.ShowGrayImage(set[i][c][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)    
            
        gtkWin.ShowGrayImage(DamagedSet[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize,1)
        #gtkWin.ShowGrayImage(hintSet[i][:].reshape(patchSize,patchSize),patchSize,patchSize,1,showSize)
        gtkWin.AddOffsetX(15)
        if i!=0 and (i+1)%rotate==0 :
            gtkWin.AddOffsetX(0)
            gtkWin.AddOffsetY(showSize+1)
#basic()
ch2();
gtk.main()
