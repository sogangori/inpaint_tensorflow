import Image
import numpy

def load_image( infilename ) :
    img = Image.open( infilename )
    print (img)
    row,col =  img.size
    print ('row',row,'col',col)    
    data = numpy.asarray( img, dtype="int32" )    
    return data

def load_image_to_size( infilename ,dstW, dstH) :
    img = Image.open( infilename )
        
    print (img)
    row,col =  img.size
    print (row,' -> ',dstW,col,' -> ',dstH)    
    im = img.resize((dstW, dstH), Image.BILINEAR)
    nrow,ncol =  im.size
    print ('new row',nrow,'new col',ncol)
        
    data = numpy.asarray( im, dtype="int32" )
    img.save( "resizeBefore1.jpg" )  
    
    return data

def save_image( npdata, outfilename ) :
    #img = Image.fromarray( numpy.asarray( numpy.clip(npdata,0,255), dtype="uint8"), "L" )
    img = Image.fromarray( npdata, "RGBA")
    img.save( outfilename )
    
PIXEL_DEPTH=255 
IMAGE_SIZE=100

image = "/tmp/n5-1.png"
image = "/tmp/golf.png"
img=load_image(image);
print( "original length", len(img))  
imData = (img - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  
imData = load_image_to_size(image, IMAGE_SIZE,IMAGE_SIZE)
print( "resize length", len(imData))
save_image(imData,"resize.jpeg")            

#data = numpy.fromiter(imData, dtype=numpy.uint8).astype(numpy.float32)

imData = (imData - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
reshapeData = imData.reshape([1, IMAGE_SIZE , IMAGE_SIZE , 3])


