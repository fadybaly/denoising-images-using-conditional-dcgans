# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 19:25:59 2018
@author: antonia 
"""
import numpy as np
import random
from PIL import Image
import cv2

def noisy(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
        One of the following strings, selecting the type of noise to add:
    
        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        'sp'       Replaces random pixels with 0 or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """

    if noise_typ == 'gauss':
        mean = 0
        var = 0.01
        gauss = np.random.normal(mean, var**0.5, image.shape)
        noisy = image + gauss
        noisy[noisy<0] = 0
        noisy[noisy>1] = 1
        return noisy

    elif noise_typ == 'sp':
        prob = 0.05
        noisy = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    noisy[i][j] = 0
                elif rdn > thres:
                    noisy[i][j] = 255
                else:
                    noisy[i][j] = image[i][j]
        return noisy
    
    elif noise_typ == 'pix':
        backgroundColor = (0,)*3 # black
        pixelSize = 10
#        image = cv2.imread('Aaron_Eckhart_0001.jpg')
        b, g, r   = cv2.split(image)
        image = cv2.merge((r,g,b))

        image = Image.fromarray(image)
        image = image.resize((image.size[0]//pixelSize, image.size[1]//pixelSize), Image.NEAREST)
        image = image.resize((image.size[0]*pixelSize, image.size[1]*pixelSize), Image.NEAREST)
        pixel = image.load()

        for i in range(0,image.size[0],pixelSize):
          for j in range(0,image.size[1],pixelSize):
            for r in range(pixelSize):
              pixel[i+r,j] = backgroundColor
              pixel[i,j+r] = backgroundColor

        # we choose to only take the first 3 channels as the 4th is the not wanted alpha channel
        image = np.array(image)[:,:,:3]
        # we convert the numpy image from RGB to BGR which is how the cv2 library reads the image
        # can be converted to RGB depending on usage
        r, g, b   = cv2.split(image)
        image = cv2.merge((b,g,r)) 
        return image

    elif noise_typ =='speckle':
        row,col, ch = image.shape
        speckle = np.random.randn(row,col, ch)*0.15
        speckle = speckle.reshape(row, col, ch) 
        noisy = image + image * speckle
        return noisy
