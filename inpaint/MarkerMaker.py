from PIL import Image
import numpy
import math

class MarkerMaker():
    '''
    Marker maker
    '''
    patch_size = 9
    markerV = 255;
    start_angle = 0
    end_angle = 0
    src = 0

    def __init__(self):
        print ("MarkerMaker __init__")
    
    def __del__(self):
        print ("MarkerMaker __del__")
            
    def makeMarker(self, patch_size):
        self.src = numpy.zeros((patch_size, patch_size))
        tangent = math.tan(math.radians(self.start_angle))
        if tangent == 0:
           tangent += 0.01
        a = 1 / tangent
          
        self.src[patch_size // 2][patch_size // 2] = self.markerV
        for y in range(0, patch_size):
            for x in range(0, patch_size):
                if (y - patch_size // 2) < a * (x - patch_size // 2):                 
                    self.src[y][x] = self.markerV
        self.src = numpy.asarray(self.src, dtype="uint8")
        return self.src
    
    
    def makeRandomMarker(self, scope=360):      
         
         self.start_angle = numpy.random.rand() * scope  
         self.end_angle = (self.start_angle + 180) % 360        
         #print('start, end', self.start_angle, self.end_angle)
     
         self.makeMarker(self.patch_size)
         self.checkMarkerValid(self.patch_size, self.patch_size)
               
         return self.src         

    def saveRandomMarker(self,count):        
         for howMany in range(0,count,1):
             self.start_angle = numpy.random.rand() * 360 
             self.end_angle = (self.start_angle + 180) % 360        
             #print('start, end', self.start_angle, self.end_angle)
         
             marker = self.makeMarker(self.patch_size)
             if self.checkMarkerValid(self.patch_size, self.patch_size):
                 # print(marker)
                 dstFileName = "marker/marker" + str(self.start_angle) + ".png"
                 self.save_image(dstFileName)  
             else:
                 print('error')
         return self.src 
     
    def checkMarkerValid(self, w, h):
        markv = self.src[h / 2][w / 2]
        unknownRatio = numpy.sum(self.src) / (w * h * markv)    
        if unknownRatio > 0.8:
            print('[ERROR] unknownRatio', unknownRatio)
            return False
        else:        
            return True    
        
    def save_image(self, outfilename):
        data = numpy.asarray(self.src, dtype="uint8")
        img = Image.fromarray(data)
        img.save(outfilename)
        print("marker image saved",outfilename)  
    
#marker = MarkerMaker()
#marker.makeRandomMarker()
#marker.saveRandomMarker(1)
