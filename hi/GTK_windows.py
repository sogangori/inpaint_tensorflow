#!/usr/bin/env python

# example layout.py

import pygtk
pygtk.require('2.0')
import gtk
import random
from inpaint.MarkingPatchMaker import MarkingPatchMaker

class LayoutExample:
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
        window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        window.set_title("Layout Example")
        window.set_default_size(300, 300)
        window.connect("delete-event", self.WindowDeleteEvent)
        window.connect("destroy", self.WindowDestroy)
        # create the table and pack into the window
        table = gtk.Table(2, 2, False)
        window.add(table)
        window.set_default_size(800,500)
        # create the layout widget and pack into the table
        self.layout = gtk.Layout(None, None)
        self.layout.set_size(600, 600)
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
        button = gtk.Button("Press Me")
        button.connect("clicked", self.ButtonClicked)
        self.layout.put(button, 200, 0)
        
        #self.layout.put(image, 10, 0)
        self.ShowPatch()       
        # show all the widgets
        window.show_all()
        
    def ShowPatch(self):       
        
        paddig=2
        image = gtk.Image()
        imgpath ="../hi/image/golf.png" 
        image.set_from_file(imgpath)
        image.show()        
        
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
        imageShowCount=0
        self.layout.put(image2, imageShowCount*showWidth, showWidth)
        imageShowCount+=1
        
        pixbufMarker =gtk.gdk.pixbuf_new_from_array(markingPatch, gtk.gdk.COLORSPACE_RGB, 8)
        pixbufMarker = pixbufMarker.scale_simple(showWidth, showWidth, gtk.gdk.INTERP_BILINEAR)
        imageMarker = gtk.Image()
        imageMarker.set_from_pixbuf(pixbufMarker)
        self.layout.put(imageMarker, paddig+imageShowCount*showWidth, showWidth)
        
        
def main():
    # enter the main loop
    gtk.main()
    return 0

if __name__ == "__main__":
    LayoutExample()
    main()
