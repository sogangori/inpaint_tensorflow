#!/usr/bin/env python

# example layout.py

import pygtk
from matplotlib.colors import Colormap
pygtk.require('2.0')
import gtk
import random
import numpy
from inpaint.MarkingPatchMaker import MarkingPatchMaker

class GTK_Window:
    paddig=2
    imageShowCount=0
    X0=0
    Y0=0
    
    def WindowDeleteEvent(self, widget, event):
        # return false so that window will be destroyed
        return False

    def WindowDestroy(self, widget, *data):
        # exit main loop
        gtk.main_quit()

    def ButtonClicked(self, button):
        # move the button
        self.layout.move(button, random.randint(0,200), random.randint(0,200))

    def __init__(self):
        # create the top level window
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title("Layout Example")
        self.window.set_default_size(1500,1000)
        self.window.set_position(5)
        self.window.connect("delete-event", self.WindowDeleteEvent)
        self.window.connect("destroy", self.WindowDestroy)
        # create the table and pack into the window
        table = gtk.Table(2, 2, False)
        self.window.add(table)
        
        # create the layout widget and pack into the table
        self.layout = gtk.Layout(None, None)
        self.layout.set_size(1200, 600)
        table.attach(self.layout, 0, 1, 0, 1, gtk.FILL|gtk.EXPAND,
                     gtk.FILL|gtk.EXPAND, 0, 0)
        # create the scrollbars and pack into the table
        vScrollbar = gtk.VScrollbar(None)
        table.attach(vScrollbar, 1, 2, 0, 1, gtk.FILL|gtk.SHRINK,
                     gtk.FILL|gtk.SHRINK, 0, 0)
        hScrollbar = gtk.HScrollbar(None)
        table.attach(hScrollbar, 0, 1, 1, 2, gtk.FILL|gtk.SHRINK,
                     gtk.FILL|gtk.SHRINK, 0, 0)    
        # tell the scrollbars to use the layout widget's adjustments
        vAdjust = self.layout.get_vadjustment()
        vScrollbar.set_adjustment(vAdjust)
        hAdjust = self.layout.get_hadjustment()
        hScrollbar.set_adjustment(hAdjust)
        # create 3 buttons and put them into the layout widget
        button = gtk.Button("Press Me")
        button.connect("clicked", self.ButtonClicked)
        self.layout.put(button, 0, 0)
        button = gtk.Button("Press Me")
        button.connect("clicked", self.ButtonClicked)
        self.layout.put(button, 100, 0)
                
        #self.layout.put(image, 10, 0)
        #self.ShowPatch()       
        # show all the widgets
        self.window.show_all()
        
    def ShowPatch(self):       
        
        markerMaker = MarkingPatchMaker()
        
        trainingSet = markerMaker.makeRandomMarkingPatchRGBA_count(1)
        marker = markerMaker.getmarker()
        patch = markerMaker.getPatch()
        markingPatch=markerMaker.getMarkingPatch()
        showWidth=100
        pixbufPatch =gtk.gdk.pixbuf_new_from_array(patch, gtk.gdk.COLORSPACE_RGB, 8)
        pixbufPatch = pixbufPatch.scale_simple(showWidth, showWidth, gtk.gdk.INTERP_BILINEAR)
        image2 = gtk.Image()
        image2.set_from_pixbuf(pixbufPatch)
        
        self.layout.put(image2, self.imageShowCount*showWidth, showWidth)
        self.imageShowCount+=1
        
        pixbufMarker =gtk.gdk.pixbuf_new_from_array(markingPatch, gtk.gdk.COLORSPACE_RGB, 8)
        pixbufMarker = pixbufMarker.scale_simple(showWidth, showWidth, gtk.gdk.INTERP_BILINEAR)
        imageMarker = gtk.Image()
        imageMarker.set_from_pixbuf(pixbufMarker)
        self.layout.put(imageMarker, self.paddig+self.imageShowCount*showWidth, showWidth)
    
    def ShowGrayImage(self,data,w,h,channel, showWidth):
        data3=data;
        if channel==1:    
            data3 = numpy.zeros(shape=(w,h,3), dtype=numpy.ubyte)
            for y in range(0,h ):
                for x in range(0,w ):
                    data3[y,x,0]=data[y,x]    
                    data3[y,x,1]=data[y,x]
                    data3[y,x,2]=0
            
        pixbufPatch =gtk.gdk.pixbuf_new_from_array(data3, gtk.gdk.COLORSPACE_RGB, 8)
        pixbufPatch = pixbufPatch.scale_simple(showWidth, showWidth, gtk.gdk.INTERP_BILINEAR)
        image2 = gtk.Image()
        image2.set_from_pixbuf(pixbufPatch)
        
        self.layout.put(image2,self.X0+ self.imageShowCount*self.paddig+ self.imageShowCount*showWidth, self.Y0)
        self.imageShowCount+=1
        self.window.show_all()
        
    def ShowImage(self,data,showWidth):    
            
        pixbufPatch =gtk.gdk.pixbuf_new_from_array(data, gtk.gdk.COLORSPACE_RGB, 8)
        pixbufPatch = pixbufPatch.scale_simple(showWidth, showWidth, gtk.gdk.INTERP_BILINEAR)
        image2 = gtk.Image()
        image2.set_from_pixbuf(pixbufPatch)
        
        self.layout.put(image2,self.X0+ self.imageShowCount*self.paddig+ self.imageShowCount*showWidth, self.Y0)
        self.imageShowCount+=1
        self.window.show_all()
        
    def AddOffsetX(self, offsetX):
        self.X0+=offsetX        
        
    def AddOffsetY(self, offsetY):
        
        self.Y0+=offsetY
        self.X0=0
        self.imageShowCount=0
        
    def AddLabel(self, text):
        label = gtk.Label(text)        
        #self.layout.put(label,self.X0+ self.imageShowCount*self.paddig+ self.imageShowCount*100, self.Y0)
        self.layout.put(label,self.X0,self.Y0)
        
def main():
    # enter the main loop
    gtk.main()
    return 0

if __name__ == "__main__":
    GTK_Window()
    main()
