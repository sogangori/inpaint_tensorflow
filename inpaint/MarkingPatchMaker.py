import Image
import numpy
from inpaint.MarkerMaker import MarkerMaker
from inpaint.PatchMaker import PatchMaker
import imageProcess.ImageConverter as imageConverter


class MarkingPatchMaker():
    '''
    Make RGB Patch From Image
    '''   
    patch_size = 9
    patchFolder = "patch"
    #imgPath = "image/golf_find_edges.png"
    #imgPath = "image/ioi_zhu.jpg"
    #imgPath = "image/colors2.jpg"        
    imgPath = "../image/building.jpg"
    imgPath = "../image/New_york_retribution.png"
    imgPath = "../image/new_york1.png"
    IMAGE_SIZE=9
    CHANNEL=3
    patchMaker = PatchMaker()
    markerMaker = MarkerMaker()
    marker=0
    patch=0
    markingPatch=0
    patchSet=0
    markerSet=0
    markingPatchSet=0
    RGBASet=0
    
    def __init__(self):
        print ("PatchMaker __init__")
        self.patchMaker.setImagePath(self.imgPath)
 
    def save_image(self, npdata, outfilename ) :          
        img = Image.fromarray( npdata, "RGB")
        img.save( outfilename )
        print ("marking patch saved ",outfilename)   
        
    def makeRandomMarkingPatch(self):        
        
        patch = self.patchMaker.makeRandomPatch()
        maker = self.markerMaker.makeRandomMarker()  
        
        markingPatch = numpy.zeros(shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,self.CHANNEL), dtype=numpy.ubyte)
        
        for y in range(0,self.IMAGE_SIZE):
            for x in range(0,self.IMAGE_SIZE):
                markingPatch[y][x]=patch[y][x]*(maker[y][x]/255)
        
        #print("training image is ", patchMaker.imgPath , " (x,y)",patchMaker.x1, patchMaker.y1)
        #print ("patch",patch)
        return markingPatch
        
    def makeRandomMarkingPatchRGB(self):
        patch = self.patchMaker.makeRandomPatch()    
        rgb = patch.reshape( numpy.size(patch))        
        #print ("patch",patch)
        return rgb
    
    def makeRandomMarkingPatchRGB_count(self,count):
        RGBLength=self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL)
        trainingSet = numpy.zeros(shape=(count*RGBLength), dtype=numpy.ubyte)
        for i in range(0,count):            
            trainingSet[i*RGBLength:i*RGBLength+RGBLength]=self.makeRandomMarkingPatchRGB()
        
        return trainingSet

    def makeRandomMarkingPatchRGBA(self):          
        
        self.patch = self.patchMaker.makeRandomPatch()        
        maker = self.markerMaker.makeRandomMarker()         
      
        self.markingPatch = numpy.zeros(shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,(self.CHANNEL)), dtype=numpy.ubyte)
        
        for y in range(0,self.IMAGE_SIZE):
            for x in range(0,self.IMAGE_SIZE):
                self.markingPatch[y][x]=self.patch[y][x]*(maker[y][x]/255)
        
        print("training image is ", self.patchMaker.imgPath , " (x,y)",self.patchMaker.x1, self.patchMaker.y1)
        
        rgba = numpy.zeros(shape=(self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL+1)), dtype=numpy.ubyte)
        rgb = self.markingPatch.reshape( numpy.size(self.markingPatch))
        self.marker = maker.reshape( numpy.size(maker))      
        
        rgba[0:numpy.size(rgb) ] = rgb
        rgba[numpy.size(rgb) : ] = self.marker
        #print ("patch",patch)
        return rgba

    def makeRandomMarkingPatchRGBA_count(self,count, scope=360):          
        print("training image is ", self.patchMaker.imgPath)
        widthHeight=self.IMAGE_SIZE*self.IMAGE_SIZE
        RGBALength=self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL+1)
        RGBLength=self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL)
        trainingSet = numpy.zeros(shape=(count*RGBALength), dtype=numpy.ubyte)
        self.RGBASet = numpy.ones(shape=(count*RGBALength), dtype=numpy.ubyte)
        self.patchSet = numpy.zeros(shape=(count*RGBLength), dtype=numpy.ubyte)
        self.markerSet = numpy.zeros(shape=(count*self.IMAGE_SIZE*self.IMAGE_SIZE), dtype=numpy.ubyte)
        for i in range(0,count):   
            self.patch = self.patchMaker.makeRandomPatch()        
            maker = self.markerMaker.makeRandomMarker(scope)         
          
            self.markingPatch = numpy.zeros(shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,(self.CHANNEL)), dtype=numpy.ubyte)
            
            for y in range(0,self.IMAGE_SIZE):
                for x in range(0,self.IMAGE_SIZE):
                    self.markingPatch[y][x]=self.patch[y][x]*(maker[y][x]/255)
            
            #print("training image is ", self.patchMaker.imgPath , " (x,y)",self.patchMaker.x1, self.patchMaker.y1)
            
            rgba = numpy.zeros(shape=(self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL+1)), dtype=numpy.ubyte)
            rgb = self.markingPatch.reshape( numpy.size(self.markingPatch))
            self.marker = maker.reshape( numpy.size(maker))      
            
            rgba[0:numpy.size(rgb) ] = rgb
            rgba[numpy.size(rgb) : ] = self.marker
                     
            trainingSet[i*RGBALength:i*RGBALength+RGBALength]=rgba
            self.patchSet[i*RGBLength:i*RGBLength+RGBLength]=self.patch.reshape( numpy.size(self.patch))
            self.markerSet[i*widthHeight:i*widthHeight+widthHeight]=self.marker
            self.RGBASet[i*RGBALength:i*RGBALength+RGBLength]=self.patch.reshape( numpy.size(self.patch))
        
        return trainingSet
    
    def generateTrainSet(self,count,dstChannel=4, scope=360):          
        print("training image is ", self.patchMaker.imgPath)
        widthHeight=self.IMAGE_SIZE*self.IMAGE_SIZE
        dstLength=widthHeight*dstChannel
        channel4L=widthHeight*4
        RGBLength=widthHeight*(self.CHANNEL)
        trainingSet = numpy.zeros(shape=(count*dstLength), dtype=numpy.ubyte)
        self.RGBASet = numpy.ones(shape=(count*channel4L), dtype=numpy.ubyte)
        self.patchSet = numpy.zeros(shape=(count*RGBLength), dtype=numpy.ubyte)
        self.markerSet = numpy.zeros(shape=(count*self.IMAGE_SIZE*self.IMAGE_SIZE), dtype=numpy.ubyte)
        for i in range(0,count):   
            self.patch = self.patchMaker.makeRandomPatch()        
            maker = self.markerMaker.makeRandomMarker(scope)         
          
            self.markingPatch = numpy.zeros(shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,(self.CHANNEL)), dtype=numpy.ubyte)
            
            for y in range(0,self.IMAGE_SIZE):
                for x in range(0,self.IMAGE_SIZE):
                    self.markingPatch[y][x]=self.patch[y][x]*(maker[y][x]/255)
            
            #print("training image is ", self.patchMaker.imgPath , " (x,y)",self.patchMaker.x1, self.patchMaker.y1)
            
            rgba = numpy.zeros(shape=(self.IMAGE_SIZE*self.IMAGE_SIZE*(self.CHANNEL+1)), dtype=numpy.ubyte)
            rgb = self.markingPatch.reshape( numpy.size(self.markingPatch))
            self.marker = maker.reshape( numpy.size(maker))      
            
            rgba[0:numpy.size(rgb) ] = rgb
            rgba[numpy.size(rgb) : ] = self.marker
                     
            trainingSet[i*dstLength:i*dstLength+channel4L]=rgba
            if dstChannel>4:
                grayEdge=imageConverter.getRgbArray_GRAY_EdgeArray(self.patch)
                trainingSet[i*dstLength+channel4L:i*dstLength+widthHeight*5]=grayEdge.reshape( numpy.size(grayEdge))
            self.patchSet[i*RGBLength:i*RGBLength+RGBLength]=self.patch.reshape( numpy.size(self.patch))
            self.markerSet[i*widthHeight:i*widthHeight+widthHeight]=self.marker
            self.RGBASet[i*channel4L:i*channel4L+RGBLength]=self.patch.reshape( numpy.size(self.patch))            
        
        return trainingSet
    
    def getPatch(self):
        return self.patch
    
    def getmarker(self):
        return self.marker
    
    def getMarkingPatch(self):
        return self.markingPatch
        
    def getPatchSet(self):
        return self.patchSet
    
    def getMarkerSet(self):
        return self.markerSet
    
    def getRGBASet(self):
        return self.RGBASet
      