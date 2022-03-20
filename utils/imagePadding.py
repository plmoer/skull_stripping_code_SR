"""
The function is to pad input image, so that the dimension are 2^n
usage: img-->a 4D input image: nChannel, nRow, nCol, nSlice
       nBase --> base value, defaut value is 16

Author: Linmin Pei
Date: July 10, 2020
"""
import numpy as np


def imgPadding(img, nBase=16):
    nChannel, nX, nY, nZ = img.shape  # get image size
    if nX % nBase != 0:
        newX = (nX//nBase+1)*nBase
    else:
        newX = nX

    if nY % nBase != 0:
        newY = (nY//nBase+1)*nBase
    else:
        newY = nY

    if nZ % nBase != 0:
        newZ = (nZ//nBase+1)*nBase
    else:
        newZ = nZ
    new_img = np.zeros((nChannel, newX, newY, newZ), dtype=np.float64)
    new_img[:, :nX, :nY, :nZ] = img
    # nStartX = (newX-nX)//2
    # nStartY = (newY-nY)//2
    # nStartZ = (newZ-nZ)//2
    # nRemainderX = (newX-nX) % 2
    # nRemainderY = (newY-nY) % 2
    # nRemainderZ = (newZ-nZ) % 2
    # new_img[:, nStartX:newX-nStartX-nRemainderX, nStartY:newY -
    # nStartY-nRemainderY, nStartZ:newZ-nStartZ-nRemainderZ] = img
    return new_img


def deImgPadding(img, nRow=240, nCol=240, nSlice=155):
    new_img = img[:nSlice, :nRow, :nCol]

    # nChannel, newX, newY, newZ = img.shape  # get image size
    # nStartX = (newX-nRow)//2
    # nStartY = (newY-nCol)//2
    # nStartZ = (newZ-nSlice)//2
    # nRemainderX = (newX-nRow) % 2
    # nRemainderY = (newY-nCol) % 2
    # nRemainderZ = (newZ-nSlice) % 2
    # new_img = np.zeros((nChannel, newX, newY, newZ), dtype=np.float64)
    # new_img = img[:, nStartX:newX-nStartX-nRemainderX, nStartY:newY -
    #               nStartY-nRemainderY, nStartZ:newZ-nStartZ-nRemainderZ]
    return new_img
