import numpy     as np
import scipy.io  as sio
import random    as rd

import os
import struct

from array import array

######################################################################
# MNIST database
######################################################################
def mnist_test_lbl(fPath):
    _testLblFname = 't10k-labels-idx1-ubyte'
    _pathTestLbl = os.path.join(fPath, _testLblFname)
    _teLbl = load_mnist_lbl(_pathTestLbl)
    return _teLbl

def mnist_train_lbl(fPath):
    _trainLblFname = 'train-labels-idx1-ubyte'
    _pathTrainLbl = os.path.join(fPath, _trainLblFname)
    _trLbl = load_mnist_lbl(_pathTrainLbl)
    return _trLbl

def load_mnist_lbl(fPathLbl):
    with open(fPathLbl, 'rb') as file:
        _magic, _size = struct.unpack(">II", file.read(8))
        if _magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                                 'got %d' % _magic)

        return np.asarray(array("B", file.read()))

def mnist_test_img(fPath):
    _testImgFname = 't10k-images-idx3-ubyte'
    _pathTestImg = os.path.join(fPath, _testImgFname)
    _teImg = load_mnist_img(_pathTestImg)
    return _teImg
    
def mnist_train_img(fPath):
    _trainImgFname = 'train-images-idx3-ubyte'
    _pathTrainImg = os.path.join(fPath, _trainImgFname)
    _trImg = load_mnist_img(_pathTrainImg)
    return _trImg

def load_mnist_img(fPathImg):
    with open(fPathImg, 'rb') as file:
        _magic, _size, _rows, _cols = struct.unpack(">IIII", file.read(16))
        if _magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                                 'got %d' % _magic)

        _imgData = array("B", file.read())

        return np.asarray(_imgData).reshape(_size, _cols*_rows)/255.

######################################################################
# Matlab file
######################################################################
def matlab_file(fPath, fName, fNbPatches, fSize, fPatchSize):

    _path = os.path.join(fPath, fName + ".mat")        

    _tmp  = sio.loadmat(_path)[fName]

    return subpicture(_tmp, fNbPatches, fSize, fPatchSize)


def subpicture(fSrc, fNbPatches, fSize, fPatchSize):

    _sets   = []
    _nbImg  = len(fSrc[0][0])
    _max  = fSize - fPatchSize + 1

    for i in np.arange(0, fNbPatches):
        _sample = []
        _x  = rd.randint(0, _max - 1)
        _y  = rd.randint(0, _max - 1)
        _id = rd.randint(0, _nbImg - 1)
        
        for j in np.arange(_x, _x + fPatchSize):
            for k in np.arange(_y, _y + fPatchSize):
                _sample.append(fSrc[j][k][_id])

        _sets.append(_sample)

    return np.array(_sets)

######################################################################
# Temporary datasets
######################################################################

def load_datasets(fDataname, fType):

    _tmp = np.loadtxt("../datasets/"+fDataname+"_"+fType+"sets.txt")

    return _tmp

    
