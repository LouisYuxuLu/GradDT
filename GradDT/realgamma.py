# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:09:01 2020

@author: Administrator
"""

import numpy as np


def AddUniformNoise(img):
    img = -np.log(img+1e-3)
    if img.shape[0] ==3:
        c,w,h = img.shape
        noise =np.zeros((c,w,h))
        sigma = np.random.randint(10,75)    #NoiseLevel between 1 and 75.
        noise = -np.log(np.random.gamma(shape=sigma,scale = 1/sigma,size =(c,w,h))+1e-3)
    else:
        c,w,h = img.shape
        noise =np.zeros((c,w,h))
        sigma = np.random.randint(10,75)    #NoiseLevel between 1 and 75.
        noise = -np.log(np.random.gamma(shape=sigma,scale = 1/sigma,size =(c,w,h))+1e-3)
        
    nimg = img + noise
    return np.exp(-nimg)

def AddNonUniformNoise(img):
    img = -np.log(img+1e-3)
    if img.shape[0] ==3:
        c,w,h = img.shape
        nimg = np.zeros(img.shape)
        sigmas = np.random.randint(10,75,2)
        minsigma = sigmas.min()
        maxsigma = sigmas.max()
        sigma = np.random.randint(minsigma,maxsigma+1)
        for i in range(w):
            for j in range(h):
                
                nimg[:,i,j] = img[:,i,j] - np.log(np.random.gamma(sigma,1/sigma,(3,)))
    else:
        c,w,h = img.shape
        nimg = np.zeros(img.shape)
        #sigmas = np.random.randint(10,75,2)
        #minsigma = sigmas.min()
        #maxsigma = sigmas.max()
        #sigma = np.random.randint(minsigma,maxsigma+1)
        sigmas = np.random.randint(10,75,2)
        minsigma = sigmas.min()
        maxsigma = sigmas.max()
        
        for i in range(w):
            for j in range(h):
                sigma = np.random.randint(minsigma,maxsigma+1)            
                nimg[:,i,j] = img[:,i,j] - np.log(np.random.gamma(sigma,1/sigma,(1,)))
        
    return np.exp(-nimg)

def AddBlockNoiseOnClear(img):
    img = -np.log(img+1e-3)
    if img.shape[0] ==3:
        c,w,h = img.shape
        Noise_block_num = np.random.randint(15)
        for i in range(Noise_block_num):
            sigma = np.random.randint(10,75)
            Noise_block_size = (np.random.randint(10,30),np.random.randint(10,30))
            Random_location = (np.random.randint(w-Noise_block_size[0]),np.random.randint(h-Noise_block_size[1]))
            img_block = img[:,Random_location[0]:Random_location[0]+Noise_block_size[0],Random_location[1]:Random_location[1]+Noise_block_size[1]]
            img[:,Random_location[0]:Random_location[0]+Noise_block_size[0],Random_location[1]:Random_location[1]+Noise_block_size[1]] = \
            img_block - np.log(np.random.gamma(sigma,1/sigma,img_block.shape))
    else:
        c,w,h = img.shape
        Noise_block_num = np.random.randint(15)
        for i in range(Noise_block_num):
            sigma = np.random.randint(10,75)
            Noise_block_size = (np.random.randint(10,30),np.random.randint(10,30))
            Random_location = (np.random.randint(w-Noise_block_size[0]),np.random.randint(h-Noise_block_size[1]))
            img_block = img[:,Random_location[0]:Random_location[0]+Noise_block_size[0],Random_location[1]:Random_location[1]+Noise_block_size[1]]
            img[:,Random_location[0]:Random_location[0]+Noise_block_size[0],Random_location[1]:Random_location[1]+Noise_block_size[1]] = \
            img_block - np.log(np.random.gamma(sigma,1/sigma,img_block.shape))
        
    return np.exp(-img)

def AddBlockNoiseOnUn(img):
    
    img = AddUniformNoise(img)
    img = AddBlockNoiseOnClear(img)
    return img

def AddBlockNoiseOnNon(img):
    
    img = AddNonUniformNoise(img)
    img = AddBlockNoiseOnClear(img)
    return img
    

def AddGammaNoise(img):
    
    return AddNonUniformNoise(img)
      
