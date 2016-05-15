import Image
import numpy

from resizeimage import resizeimage

#Read patch_train All Image files  
def load_image( infilename ) :
    img = Image.open( infilename )
    print (img)
    row,col =  img.size
    print ('row',row,'col',col)    
    data = numpy.asarray( img, dtype="uint8" )    
    return data

def save_image( npdata, outfilename ) :
    #img = Image.fromarray( numpy.asarray( numpy.clip(npdata,0,255), dtype="uint8"), "L" )
    img = Image.fromarray( npdata, "RGBA")
    img.save( outfilename )
    
PIXEL_DEPTH=255 
IMAGE_SIZE=9

image = "patch_train/patch1_7.jpg"
img=load_image(image);

imNormal = (img - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
reshapeData = imNormal.reshape([1, IMAGE_SIZE , IMAGE_SIZE , 3])
print img


