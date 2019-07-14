# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:28:50 2019

@author: Le Tran Thinh
"""
import matplotlib.image as mpimg
import numpy as np
from skimage import measure
from skimage.transform import rotate
from scipy.signal import savgol_filter

def xuli(base, measurement, toado, goc, ghxd, ghxt, dr , ws = 1, ptd = 1):
    bimg = mpimg.imread(base)
    bxoay = rotate(bimg, goc)
    base_xoay = np.dot(bxoay[...,:3], [299, 587, 114]) #.299 + .587 + .114 
    mimg = mpimg.imread(measurement)
    mxoay = rotate(mimg, goc)
    measure_xoay = np.dot(mxoay[...,:3], [299, 587, 114])

    dd = ghxt - ghxd + 1
    x = np.zeros((dr*2,dd))
    n = np.zeros((dr*2,dd))
    for i in range(0, dr*2):
        x[i] = measure.profile_line(base_xoay, (toado - dr + i, ghxd), (toado - dr + i, ghxt))
        n[i] = measure.profile_line(measure_xoay, (toado - dr + i, ghxd), (toado - dr + i, ghxt))
    
    if ptd == 1:
        ty = np.mean(x, axis = 0)
        tm = np.mean(n, axis = 0)
    else:
        ty = np.amax(x, axis = 0)
        tm = np.amax(n, axis = 0)
    
    ty = savgol_filter(ty, ws, 2)
    tm = savgol_filter(tm, ws, 2)
    
    #np.seterr(divide='ignore', invalid='ignore')
    xm = np.divide(tm,ty)
    kenhb = ty
    kenhm = tm
    kenh = -np.log10(xm)
    return kenhb, kenhm, kenh