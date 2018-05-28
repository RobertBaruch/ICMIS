# ICMIS
Integrated Circuit Microscopy Image Stitcher

Stitching is hard!

This is a work in progress that sort of kind of stitches together images taken from a CNC microscope. This means that the images must all be
the result of consistent movements of the stage. The images must be pretty flat -- no perspective or barrel/pincushion distortion.

You will need to install numpy, scipy, and opencv (python). If you don't know how to install these, then this program is not for you.

Your images must be consecutively numbered as follows: IMG_xxxx.JPG.

The program takes one argument, the number of columns in each row.

The program outputs stitch.jpg.
