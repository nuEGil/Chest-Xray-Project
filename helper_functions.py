'''This file allows us to store functions that may be useful across the different jupyter
notebooks in the series. We can import functions from this file rather than write them again 
into the new notebooks. 
'''

import os # file handling library
import cv2 # image processing library - opencv-python -- cv2 reads BGR so reverse the image channels when plotting with matplotlib
import glob # Unix style path name pattern expansion -- helps to build file lists
import numpy as np # library for matrix math, arrays, etc. 
import pandas as pd # data processing library
import matplotlib.pyplot as plt # plot library

# plots an image
def quickplot(img_):
    # make a figure
    plt.figure(figsize = (15,15))
    plt.imshow(img_)
    # get the axes object
    ax = plt.gca()
    # turn off x and y axis
    ax.set_axis_off()

# gets conditions from data_ - which is a data frame made from 
# dat_entry_2017.csv and its variants.
def get_conditions(data_):
    # The first part is the same as last time, we want to get the conditions
    conditions = data_['Finding Labels'].str.split('|').str[0].values
    conditions = sorted(np.unique(conditions)) # make sure these are alphabetical
    print(conditions)
    return conditions

# maybe a subplot generator
class multifigure():
    def __init__(self, rows = 3, cols = 3, width = 10, height = 10):  

        self.fig, self.axes = plt.subplots(rows, cols)
        self.axes = self.axes.flatten()
        self.naxes = rows*cols
        self.fig.set_figwidth(width)
        self.fig.set_figheight(height)
    
    # overwrite this call function to get different plotting behavior
    def call(self, x):
        # examples to plot and s et labels
        for ii in range(self.naxes):
            self.axes[ii].imshow(x[ii], vmin = np.min(x[ii]), vmax = np.max(x[ii]), cmap='gray')

# set an array to have 0 min and 1 max.
def norm(x):
    x_ = x-np.min(x) # subtract min
    Mx = np.max(x_) # get the max
    # only divide by the max if it is not 0
    if Mx != 0:
        x_ = x_ / np.max(x_)
    return x_