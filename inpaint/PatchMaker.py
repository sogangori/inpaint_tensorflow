from PIL import Image
import numpy


class PatchMaker():
    '''
    Make RGB Patch From Image
    '''   
    patch_size = 9
    patchFolder = "patch"
     
    img=0
    src=0
    x1=0
    y1=0
    
    def __init__(self):
        print ("PatchMaker __init__")

    def setImagePath(self, imagePath):
        self.imgPath = imagePath
        print ("PatchMaker imgPath=",imagePath)        
        self.img = Image.open(self.imgPath)
        print (self.img)
    
    def __del__(self):
        print ("PatchMaker __del__")

    def CropImage(self, pivotX,pivotY):
        srcW, srcH = self.img.size
        x1 = numpy.int(pivotX*srcW)
        y1 = numpy.int(pivotY*srcH)
        self.src = self.img.crop((x1, y1, x1+self.patch_size,y1+self.patch_size))   
        dstPath = self.patchFolder+'/patch'
        dstPath += str(x1)+"_"+str(y1)+ ".jpg"
        self.src.save(dstPath)
        print("PatchMaker image saved",dstPath)
    
    def makeRandomPatch(self):  
        pivotP=numpy.random.rand(2)       
        srcW, srcH = self.img.size
        self.x1 = numpy.int(pivotP[0]*srcW)
        self.y1 = numpy.int(pivotP[1]*srcH)
        self.src = self.img.crop((self.x1, self.y1, self.x1+self.patch_size,self.y1+self.patch_size))  
        return numpy.asarray(self.src, dtype="uint8")
          
            
    def saveRandomPatch(self,count): 
        pivotPosition=numpy.random.rand(count,2)
        
        for pivotP in pivotPosition: 
            cropCenterImg = self.CropImage(pivotP[0], pivotP[1])  
            
#patchMaker = PatchMaker()
