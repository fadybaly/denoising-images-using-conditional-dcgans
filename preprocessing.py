# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 14:55:06 2018

@author: Antonia
"""
import os
import cv2
import numpy as np
from noise_func import noisy

def load_and_preprocess_dataset(data_dir, noise):
    if noise == 'pix':
        A = []
        B = []
        idx = 0
        flag = 0
        for path, _, files in os.walk(data_dir):
            for file in files:
                im = cv2.imread(path + '/' + file)
                A.append(im)
                if idx==4000:
                    flag = 1
                    break
                idx += 1
            if flag == 1:
                break
        A = np.asarray(A, dtype=np.uint8)
        for im in A:
            B.append(noisy('pix', im))
        B = np.asarray(B)
    #    
        A, mean_1, mean_2, mean_3, X_min, X_max = standardize_and_preprocess_orig(np.array(A, dtype = np.int32))
        B = standardize_and_preprocess_noise(np.array(B, dtype=np.int32), mean_1, mean_2, mean_3, X_min, X_max)
        return A, B
    else:
        A = []
        B = []
        idx = 0
        flag = 0
        for path, _, files in os.walk(data_dir):
            for file in files:
                im = cv2.imread(path + '/' + file)
                A.append(im)
                if idx==5000:
                    flag = 1
                    break
                idx += 1
            if flag == 1:
                break
        A = np.asarray(A, dtype=np.int32)
        A = standardize_and_preprocess(A)
    
        for im in A:
            B.append(noisy('gauss', im))
        B = np.asarray(B)
        return A, B


def standardize_and_preprocess(X):
    mean_1 = np.mean(X[:,:,:,0])
    mean_2 = np.mean(X[:,:,:,1])
    mean_3 = np.mean(X[:,:,:,2])
    X[:,:,:,0] = X[:,:,:,0] - mean_1
    X[:,:,:,1] = X[:,:,:,1] - mean_2
    X[:,:,:,2] = X[:,:,:,2] - mean_3
    X = (X - np.min(X))/(np.max(X) - np.min(X))

    return X


def standardize_and_preprocess_orig(X):
    # take out the mean per channel
    mean_1 = np.mean(X[:,:,:,0])
    mean_2 = np.mean(X[:,:,:,1])
    mean_3 = np.mean(X[:,:,:,2])
    X[:,:,:,0] = X[:,:,:,0] - mean_1
    X[:,:,:,1] = X[:,:,:,1] - mean_2
    X[:,:,:,2] = X[:,:,:,2] - mean_3
    # normalize to range [0, 1]
    X_min = np.min(X)
    X_max = np.max(X)
    X = (X - X_min)/(X_max - X_min)
    return X, mean_1, mean_2, mean_3, X_min, X_max

def standardize_and_preprocess_noise(X, mean_1, mean_2, mean_3, X_min, X_max):
    # take out the mean per channel
    X[:,:,:,0] = X[:,:,:,0] - mean_1
    X[:,:,:,1] = X[:,:,:,1] - mean_2
    X[:,:,:,2] = X[:,:,:,2] - mean_3
    # normalize to range [0, 1]
    X = (X - X_min)/(X_max - X_min)
    return X
