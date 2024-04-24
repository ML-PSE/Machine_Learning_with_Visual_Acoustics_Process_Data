# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:23:21 2024

@author: f0ne44
"""
import numpy as np

#%% load image using Pillow
from PIL import Image
img = Image.open('digit_color.jpeg')
img.show() # displays image using your computer's default application for photos
print(img.size)

# construct NumpPy array from image object
imgData = np.asarray(img) 
print(imgData.shape)

# resize image
img_resized = img.resize((7,7))
img_resized.show()

# save resized image
img_resized.save('digit_color_resized.jpeg')

#%% load image as NumPy array directly using Matplotlib
from matplotlib import image, pyplot
imgData2 = image.imread('digit_color.jpeg') # loads image as a 3D NumPy array
pyplot.imshow(img)

