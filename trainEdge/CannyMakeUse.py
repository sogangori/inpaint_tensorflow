import scipy.ndimage as ndi
import scipy
import numpy
import Image
import math
from trainEdge.CannyMaker import CannyMaker

f = '../image/golf.png'
f = '../image/New_york_retribution.png'
folder = "canny/"
patchSize=200
cannyMaker= CannyMaker()
[img,width,height]=cannyMaker.readImage(f)

[rgbPatch,grayPatch]=cannyMaker.cropPatch(patchSize);
scipy.misc.imsave(folder+'rgb.jpg', rgbPatch)
scipy.misc.imsave(folder+'gray.jpg', grayPatch)

[sobeloutmag,sobeloutdir]=cannyMaker.makeSobel(grayPatch)

scipy.misc.imsave(folder+'cannynewmag.jpg', sobeloutmag)#thick
scipy.misc.imsave(folder+'cannynewdir.jpg', sobeloutdir)#looks object
print 'cannynewmag.jpg'

sobeloutdirquentize=cannyMaker.makeSobelOut()
scipy.misc.imsave(folder+'cannynewdirquantize.jpg', sobeloutdirquentize)#sobeloutdir black/white
print 'cannynewdirquantize.jpg'

mag_sup=cannyMaker.Make3();
scipy.misc.imsave(folder+'cannynewmagsup.jpg', mag_sup)#thin edge

[gnh,gnl]=cannyMaker.Make4();                
scipy.misc.imsave(folder+'cannynewgnlbeforeminus.jpg', gnl)
scipy.misc.imsave(folder+'cannynewgnlafterminus.jpg', gnl-gnh)
scipy.misc.imsave(folder+'cannynewgnh.jpg', gnh)

gnh=cannyMaker.make5();
scipy.misc.imsave('cannynewout.jpg', gnh)
print 'cannynewout.jpg'
print 'finish'

