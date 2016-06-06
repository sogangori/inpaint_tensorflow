import scipy.ndimage as ndi
import scipy
import numpy
import Image
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
     
    def makeSobel(self,patchImg):    
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
        random.seed( time.time())
        pivotPX=random.random()  
        random.seed( time.time())     
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
        kind = 5 
        for i in range(0,count):       
            if i%kind==0:     
                grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);
            else:   
                [rgbPatch,grayPatch]=self.cropPatch(patch_size);   
             
            if i%kind==0:             
                [sobeloutmag,sobeloutdir]=self.makeSobel(grayPatch)
                set[i][:] = self.reshapeImgToSingle(sobeloutdir,patch_length)
            if i%kind==1:                
                [sobeloutmag,sobeloutdir]=self.makeSobel(grayPatch)                
                set[i][:] = numpy.reshape(numpy.asarray(sobeloutmag*2, dtype="uint8"),[patch_length]);
            if i%kind==2:             
                set[i][:] = numpy.reshape(numpy.asarray(grayPatch, dtype="uint8"),[patch_length]);
            if i%kind==3:
                sobeloutdirquentize=self.makeSobelOut()
                set[i][:] = numpy.reshape(numpy.asarray(sobeloutdirquentize, dtype="uint8"),[patch_length]);  
            if i%kind==4:
                dx = ndi.sobel(grayPatch, 0)  # horizontal derivative
                dy = ndi.sobel(grayPatch, 1)  # vertical derivative
                mag = numpy.hypot(dx, dy)  # magnitude
                mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)                
                set[i][:] = numpy.reshape(numpy.asarray(mag, dtype="uint8"),[patch_length]);
            for k in range(0,patch_length/2):
                setDam[i][k]=set[i][k] 
            
        return [set,setDam]
    
    def generatePatchSet3(self, file, count, patch_size):
        self.readImage(file)        
        patch_length = patch_size*patch_size;
        channel = 2
        set = numpy.zeros(shape=(count,channel,patch_length), dtype=numpy.ubyte)
        setDam = numpy.zeros(shape=(count,channel,patch_length), dtype=numpy.ubyte)
        for i in range(0,count):            
            grayPatch=self.cropPatchBalance(patch_size,numpy.float(i)/count);                
            [sobeloutmag,sobeloutdir]=self.makeSobel(grayPatch)                           
            set[i][0][:] = numpy.reshape(numpy.asarray(sobeloutmag, dtype="uint8"),[patch_length]);
            set[i][1][:] = numpy.reshape(numpy.asarray(sobeloutdir, dtype="uint8"),[patch_length]);
            for k in range(0,patch_length/2):
                setDam[i][0][k]=set[i][0][k] 
                setDam[i][1][k]=set[i][1][k]
            
        return [set,setDam]
                    
                    
        
        
         

