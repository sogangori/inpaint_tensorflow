import numpy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from inpaint.PatchMaker import PatchMaker


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def getRGBImage2GRAY_Edge(image):
    image_edge = image.filter(ImageFilter.FIND_EDGES)
    return image_edge.convert('LA')


def getRGB_GRAY_Edge(imageData):
    print("getRGB_GRAY_Edge",  len(imageData))
    image = Image.fromarray( imageData, "RGB")
    print("image",  image)
    edgeGrayImg=getRGBImage2GRAY_Edge(image)
    return edgeGrayImg

def getRgbArray_GRAY_EdgeArray(imageData):
    edgeGrayImg=rgb2gray(imageData)
    dstData =numpy.asarray(edgeGrayImg, dtype="uint8")  
    return dstData

imgPath="../hi/image/golf.png"
image = Image.open(imgPath)
image_edge = image.filter(ImageFilter.FIND_EDGES)
image_edge.save('golf_RGB_find_edges.png') 

image_gray = image.convert('LA')
image_gray.save('image_gray.png')

#getRGBImage2GRAY_Edge(image).save('golf_GRAY_find_edges.png') 

pm = PatchMaker()
pm.setImagePath(imgPath)
rgbPatch = pm.makeRandomPatch()
grayEdge = getRGB_GRAY_Edge(rgbPatch)
grayEdge.save('golf_GRAY_find_edges.png')

pm = PatchMaker()
pm.setImagePath(imgPath)
rgbPatch = pm.makeRandomPatch()
grayEdgeData = getRgbArray_GRAY_EdgeArray(rgbPatch)
print("grayEdgeData",grayEdgeData)
grayEdgeDataIm = Image.fromarray(grayEdgeData)
    
grayEdgeDataIm.save('golf_GRAY_find_edges2.png')