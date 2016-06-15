import scipy.ndimage as ndi
import scipy
import numpy
from PIL import Image
import math
from math import pi
from datetime import datetime
import time
import random
class CannyMaker():
    file=0;
    img=0;
    width=0;
    height=0;
    sobeloutmag=0;
    sobeloutdir=0;
    mag_sup=0;
    gnh=0;
    gnl=0;
    def __init__(self):
        print ("CannyMaker __init__")
        
    def readImage(self,file):
        self.file = file;
        self.img = Image.open(file).convert('L')
        self.width = self.img.size[1]
        self.height = self.img.size[0]
        return [self.img,self.width,self.height]; 
    
    def makeSobelMag(self,patchImg):    
        img = patchImg;
        imgdata = numpy.array(img, dtype = float)
        sigma = 2.2                                 
        G = ndi.filters.gaussian_filter(imgdata, sigma)                           #gaussian low pass filter
        
        sobelout = Image.new('L', img.size)                                       #empty image
        gradx = numpy.array(sobelout, dtype = float)                        
        grady = numpy.array(sobelout, dtype = float)
        
        sobel_x = [[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]]
        sobel_y = [[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]]
        
        self.width = img.size[1]
        self.height = img.size[0]      
        #calculate |G| and dir(G)        
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                px = (sobel_x[0][0] * G[x-1][y-1]) + (sobel_x[0][1] * G[x][y-1]) + \
                     (sobel_x[0][2] * G[x+1][y-1]) + (sobel_x[1][0] * G[x-1][y]) + \
                     (sobel_x[1][1] * G[x][y]) + (sobel_x[1][2] * G[x+1][y]) + \
                     (sobel_x[2][0] * G[x-1][y+1]) + (sobel_x[2][1] * G[x][y+1]) + \
                     (sobel_x[2][2] * G[x+1][y+1])
        
                py = (sobel_y[0][0] * G[x-1][y-1]) + (sobel_y[0][1] * G[x][y-1]) + \
                     (sobel_y[0][2] * G[x+1][y-1]) + (sobel_y[1][0] * G[x-1][y]) + \
                     (sobel_y[1][1] * G[x][y]) + (sobel_y[1][2] * G[x+1][y]) + \
                     (sobel_y[2][0] * G[x-1][y+1]) + (sobel_y[2][1] * G[x][y+1]) + \
                     (sobel_y[2][2] * G[x+1][y+1])
                gradx[x][y] = px
                grady[x][y] = py
        
        self.sobeloutmag = scipy.hypot(gradx, grady)        
        return self.sobeloutmag 
     
    def makeSobel(self,patchImg,sigma = 2.2):    
        img = patchImg;
        imgdata = numpy.array(img, dtype = float)                                         
        G = ndi.filters.gaussian_filter(imgdata, sigma)                           #gaussian low pass filter
        
        sobelout = Image.new('L', img.size)                                       #empty image
        gradx = numpy.array(sobelout, dtype = float)                        
        grady = numpy.array(sobelout, dtype = float)
        
        sobel_x = [[-1,0,1],
                   [-2,0,2],
                   [-1,0,1]]
        sobel_y = [[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]]
        
        self.width = img.size[1]
        self.height = img.size[0]      
        #calculate |G| and dir(G)        
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                px = (sobel_x[0][0] * G[x-1][y-1]) + (sobel_x[0][1] * G[x][y-1]) + \
                     (sobel_x[0][2] * G[x+1][y-1]) + (sobel_x[1][0] * G[x-1][y]) + \
                     (sobel_x[1][1] * G[x][y]) + (sobel_x[1][2] * G[x+1][y]) + \
                     (sobel_x[2][0] * G[x-1][y+1]) + (sobel_x[2][1] * G[x][y+1]) + \
                     (sobel_x[2][2] * G[x+1][y+1])
        
                py = (sobel_y[0][0] * G[x-1][y-1]) + (sobel_y[0][1] * G[x][y-1]) + \
                     (sobel_y[0][2] * G[x+1][y-1]) + (sobel_y[1][0] * G[x-1][y]) + \
                     (sobel_y[1][1] * G[x][y]) + (sobel_y[1][2] * G[x+1][y]) + \
                     (sobel_y[2][0] * G[x-1][y+1]) + (sobel_y[2][1] * G[x][y+1]) + \
                     (sobel_y[2][2] * G[x+1][y+1])
                gradx[x][y] = px
                grady[x][y] = py
        
        self.sobeloutmag = scipy.hypot(gradx, grady)
        self.sobeloutdir = scipy.arctan2(grady, gradx)
        return [self.sobeloutmag,self.sobeloutdir]
    
    def makeSobelOut(self):
        sobeloutdir= self.sobeloutdir
        for x in range(self.width):
            for y in range(self.height):
                if (sobeloutdir[x][y]<22.5 and sobeloutdir[x][y]>=0) or \
                   (sobeloutdir[x][y]>=157.5 and sobeloutdir[x][y]<202.5) or \
                   (sobeloutdir[x][y]>=337.5 and sobeloutdir[x][y]<=360):
                    sobeloutdir[x][y]=0
                elif (sobeloutdir[x][y]>=22.5 and sobeloutdir[x][y]<67.5) or \
                     (sobeloutdir[x][y]>=202.5 and sobeloutdir[x][y]<247.5):
                    sobeloutdir[x][y]=45
                elif (sobeloutdir[x][y]>=67.5 and sobeloutdir[x][y]<112.5)or \
                     (sobeloutdir[x][y]>=247.5 and sobeloutdir[x][y]<292.5):
                    sobeloutdir[x][y]=90
                else:
                    sobeloutdir[x][y]=135
        return sobeloutdir
    
    def Make3(self):
        sobeloutmag=self.sobeloutmag
        self.mag_sup = sobeloutmag.copy()
        sobeloutdir= self.sobeloutdir
        mag_sup=self.mag_sup
        for x in range(1, self.width-1):
            for y in range(1, self.height-1):
                if sobeloutdir[x][y]==0:
                    if (sobeloutmag[x][y]<=sobeloutmag[x][y+1]) or \
                       (sobeloutmag[x][y]<=sobeloutmag[x][y-1]):
                        mag_sup[x][y]=0
                elif sobeloutdir[x][y]==45:
                    if (sobeloutmag[x][y]<=sobeloutmag[x-1][y+1]) or \
                       (sobeloutmag[x][y]<=sobeloutmag[x+1][y-1]):
                        mag_sup[x][y]=0
                elif sobeloutdir[x][y]==90:
                    if (sobeloutmag[x][y]<=sobeloutmag[x+1][y]) or \
                       (sobeloutmag[x][y]<=sobeloutmag[x-1][y]):
                        mag_sup[x][y]=0
                else:
                    if (sobeloutmag[x][y]<=sobeloutmag[x+1][y+1]) or \
                       (sobeloutmag[x][y]<=sobeloutmag[x-1][y-1]):
                        mag_sup[x][y]=0
        return mag_sup
    
    def Make4(self):    
        mag_sup=self.mag_sup
        m = numpy.max(mag_sup)
        th = 0.2*m
        tl = 0.1*m    
        
        self.gnh = numpy.zeros((self.width, self.height))
        self.gnl = numpy.zeros((self.width, self.height))
        
        for x in range(self.width):
            for y in range(self.height):
                if mag_sup[x][y]>=th:
                    self.gnh[x][y]=mag_sup[x][y]
                if mag_sup[x][y]>=tl:
                    self.gnl[x][y]=mag_sup[x][y]
                    
        return [self.gnh,self.gnl];
    
    def make5(self):
        gnh = self.gnh
        gnl = self.gnl
        gnl = gnl-gnh
        def traverse(i, j):
            x = [-1, 0, 1, -1, 1, -1, 0, 1]
            y = [-1, -1, -1, 0, 0, 1, 1, 1]
            for k in range(8):
                if gnh[i+x[k]][j+y[k]]==0 and gnl[i+x[k]][j+y[k]]!=0:
                    gnh[i+x[k]][j+y[k]]=1
                    traverse(i+x[k], j+y[k])
        
        for i in range(1, self.width-1):
            for j in range(1, self.height-1):
                if gnh[i][j]:
                    gnh[i][j]=1
                    traverse(i, j)
        return gnh;
    
    def CropImage(self, pivotX,pivotY):
        srcW, srcH = self.img.size
        x1 = numpy.int(pivotX*srcW)
        y1 = numpy.int(pivotY*srcH)
        self.src = self.img.crop((x1, y1, x1+self.patch_size,y1+self.patch_size))   
        dstPath = self.patchFolder+'/patch'
        dstPath += str(x1)+"_"+str(y1)+ ".jpg"
        self.src.save(dstPath)
        print("PatchMaker image saved",dstPath)
        
    def cropPatch(self,patch_size):
        rgbImg = Image.open(self.file)  
        random.seed( time.time()*1000)
        pivotPX=random.random()  
        random.seed( time.time()*1000)     
        pivotPY=random.random()
        srcW, srcH = rgbImg.size
        x1 = numpy.int(pivotPX*srcW)
        y1 = numpy.int(pivotPY*srcH)
        if x1 + patch_size > srcW :
            x1 = srcW-patch_size
        if y1 + patch_size > srcH :
            y1 = srcH-patch_size
        rgbPatch = rgbImg.crop((x1, y1, x1+patch_size,y1+patch_size))                
        grayPatch = self.img.crop((x1, y1, x1+patch_size,y1+patch_size))
        return [rgbPatch,grayPatch]
        #return numpy.asarray(self.src, dtype="uint8")
        
    def cropPatchBalance(self,patch_size,index):
        rgbImg = Image.open(self.file)
        srcW, srcH = rgbImg.size
        indexP = (srcW-patch_size)*(srcH-patch_size)*numpy.float(index)                
        x1 = numpy.int( indexP/(srcH-patch_size));
        y1 = numpy.int(indexP%(srcH-patch_size));
        
        if index%2==0:
            x1 = srcW-x1;
            y1 = srcH-y1;
        if x1 + patch_size > srcW :
            x1 = srcW-patch_size
        if y1 + patch_size > srcH :
            y1 = srcH-patch_size                        
        grayPatch = self.img.crop((x1, y1, x1+patch_size,y1+patch_size))
        return grayPatch
        #return numpy.asarray(self.src, dtype="uint8")
        
    def generatePatchSet(self, file, count, patch_size):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        set = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        for i in range(0,count):
            [rgbPatch,grayPatch]=self.cropPatch(patch_size);                
            [sobeloutmag,sobeloutdir]=self.makeSobel(grayPatch)            
            set[i][:]= numpy.reshape(numpy.asarray(sobeloutmag, dtype="uint8"),[patch_length]);
        return set
    
    def generatePatchSet2(self, file, count, patch_size):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        set = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        setDam = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        for i in range(0,count):       
            if i%2==0:     
                grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);                             
            [sobeloutmag,sobeloutdir]=self.makeSobel(grayPatch)                
            set[i][:] = numpy.reshape(numpy.asarray(sobeloutdir, dtype="uint8"),[patch_length]);
            for k in range(0,patch_length/2):
                setDam[i][k]=set[i][k] 
            
        return [set,setDam]
    
    def reshapeImgToSingle(self,src,length):
        return numpy.reshape(numpy.asarray(src, dtype="uint8"),[length]);
    
    def generatePatchSet2mix(self, file, count, patch_size):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        set = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        setDam = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        kind = 4 
        for i in range(0,count/kind):       
            if i%2==0:     
               grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);
            for k in range(0,kind):
                dstIndex = i*kind+k;
                makeImage=0;
                if k%kind==0:
                    makeImage=grayPatch;
                if k%kind==1:
                    [makeImage,sobeloutdir]=self.makeSobel(grayPatch,sigma = 0.5)
                if k%kind==2:
                    [makeImage,sobeloutdir]=self.makeSobel(grayPatch,sigma = 1.6)
                if k%kind==3:                    
                    [makeImage,sobeloutdir]=self.makeSobel(grayPatch)
                
                set[dstIndex][:] = numpy.reshape(numpy.asarray(makeImage, dtype="uint8"),[patch_length]);
                for j in range(0,patch_length/2):
                    setDam[dstIndex][j]=set[dstIndex][j] 
            
        return [set,setDam]
    
    def randomUnknownArray(self, arrayLength, unknownRatio ):        
        unknownCount= numpy.int( arrayLength*unknownRatio)
        
        x  =  random.randint(0,arrayLength-unknownCount)
        random.seed( time.time()*x)
        arr=numpy.ones(arrayLength, dtype="float")    
        for x in range(x, x+unknownCount):
            arr[x] = 0
        return arr
    
    def randomUnknownArrayrandom(self, arrayLength, unknownRatio ):        
        
        x  =  random.randint(0,numpy.int(arrayLength-arrayLength*unknownRatio))
        random.seed( time.time()*x)
        arr=numpy.ones(arrayLength, dtype="float")
        
        for x in range(arrayLength-x, arrayLength):
            arr[x] = 0
        return arr

    def generatePatchSetWhatStudy(self, file, count, patch_size, unknownRatio=0.5):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        set = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        setDamege = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
                 
        for i in range(0,count):
            n2  = self.randomUnknownArray(patch_length, unknownRatio)       
            if i%4==0:     
                grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);
                       
            set[i][:] = numpy.reshape(numpy.asarray(grayPatch, dtype="uint8"),[patch_length]);                                
            setDamege[i][:]=set[i][:]*n2
                            
        return [set,setDamege]
         
    def generatePatchSetWidthHint(self, file, count, patch_size, unknownRatio=0.5):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        set = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        setDamege = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        setHint = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        unknownCount=numpy.int(patch_length*unknownRatio);
        n0=numpy.ones(patch_length-unknownCount, dtype="float")    
        n1=numpy.zeros(unknownCount, dtype="float")
        n2=numpy.concatenate((n0, n1), axis=0)
        kind = 1 
        for i in range(0,count/kind):       
            if i%4==0:     
                grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);
            for k in range(0,kind):
                dstIndex = i*kind+k;
                makeImage=0;
                hintImage=0;
                makeImage=grayPatch;
                [hintImage,sobeloutdir]=self.makeSobel(grayPatch,sigma = 1.0)     
                set[dstIndex][:] = numpy.reshape(numpy.asarray(makeImage, dtype="uint8"),[patch_length]);
                setHint[dstIndex][:] = numpy.reshape(numpy.asarray(hintImage, dtype="uint8"),[patch_length]);                
                setDamege[dstIndex][:]=set[dstIndex][:]*n2
                            
        return [set,setDamege,setHint]
    
    def  reversal(self,src):        
        return src*-1+255
    
    def generatePatchSetChannel(self, file, count, patch_size, channel,unknownRatio=0.5):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        inputSet = numpy.zeros(shape=(count,channel,patch_length), dtype=numpy.ubyte)
        labelSet = numpy.zeros(shape=(count,patch_length), dtype=numpy.ubyte)
        aug=2
        rotate=4
        for i in range(0,count/(aug*rotate)):
            n2  = self.randomUnknownArrayrandom(patch_length, unknownRatio)
            if i%4==3:     
                grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);
           
            [hintImage,sobeloutdir]=self.makeSobel(grayPatch,sigma = 0.6)
            hintImage = numpy.reshape(numpy.asarray(hintImage, dtype="uint8"),[patch_length]);
            hint2d   = numpy.reshape(numpy.asarray(hintImage, dtype="uint8"),[patch_size,patch_size])
            srcImage = numpy.reshape(numpy.asarray(grayPatch, dtype="uint8"),[patch_length]);
            srcImage2d  = numpy.reshape(numpy.asarray(srcImage, dtype="uint8"),[patch_size,patch_size])
            
            for  j in range(0,rotate):            
                                 
                srcRot90 = numpy.reshape(numpy.asarray(numpy.rot90(srcImage2d,j), dtype="uint8"),[patch_length])
                hintRot90 = numpy.reshape(numpy.asarray(numpy.rot90(hint2d,j), dtype="uint8"),[patch_length])
                dindex =  i*(aug*rotate)+j*(aug)
                                     
                labelSet[dindex+0][:] =srcRot90
                labelSet[dindex+1][:] =self.reversal(srcRot90)
                
                inputSet[dindex+0][0][:]=labelSet[dindex]*n2
                inputSet[dindex+1][0][:]= labelSet[dindex+1]*n2
                
                if channel>=2:
                    inputSet[dindex+0][1][:] =n2*255                
                    inputSet[dindex+1][1][:] =inputSet[dindex][1][:]                
                    
                if channel>=3:
                    inputSet[dindex+0][2][:] =hintRot90*n2               
                    inputSet[dindex+1][2][:] =hintRot90*n2
            
        return [inputSet,labelSet]
