
import numpy     as np
import scipy.io  as sio
import random    as rd
import cPickle   as cp
import display   as dy

import os
import struct

from array import array

# Global parameters
DATAPATH = "../datasets/"

#####################################################################
# MNIST database
#####################################################################
def mnist_test_lbl(fPath):

    _path = os.path.join(fPath, 't10k-labels-idx1-ubyte')
    return load_mnist_lbl(_path)

def mnist_train_lbl(fPath):

    _path = os.path.join(fPath, 'train-labels-idx1-ubyte')
    return load_mnist_lbl(_path)

def load_mnist_lbl(fPath):
    with open(fPath, 'rb') as file:
        _magic, _size = struct.unpack(">II", file.read(8))
        if _magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                                 'got %d' % _magic)

        return np.asarray(array("B", file.read()))

def mnist_test_img(fPath):

    _path = os.path.join(fPath, 't10k-images-idx3-ubyte')
    return load_mnist_img(_path)
    
def mnist_train_img(fPath):
    _trainImgFname = 'train-images-idx3-ubyte'
    _pathTrainImg = os.path.join(fPath, _trainImgFname)
    _trImg = load_mnist_img(_pathTrainImg)
    return _trImg

def load_mnist_img(fPath):
    with open(fPath, 'rb') as file:
        _magic, _size, _rows, _cols = struct.unpack(">IIII",
                                                    file.read(16))
        if _magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                                 'got %d' % _magic)

        _imgData = array("B", file.read())

        return np.asarray(_imgData).reshape(_size, _cols*_rows)/255.

def mnist_train(fPath):
    return [mnist_train_img(fPath), mnist_train_lbl(fPath)]

def mnist_test(fPath):
    return [mnist_test_img(fPath), mnist_test_lbl(fPath)]

#####################################################################
# Cifar-10 file
#####################################################################
def normalize_cifar_set(fDict):

    _imgs = np.zeros((10000*len(fDict), 1024))
    _lbls = np.zeros(10000*len(fDict))

    for i in xrange(len(fDict)):
    
        for _type, _dt in fDict[i].items():

            if _type == "data":
                _imgs[i*len(_dt):(i+1)*len(_dt),:] = (_dt[:,0:1024] / 3. + _dt[:,1024:2048] / 3. + _dt[:,2048:3072] / 3.)

            elif _type == "labels":
                _lbls[i*len(_dt):(i+1)*len(_dt)] = _dt            
                
    return _imgs / 255, _lbls
            
def cifar10_train(fPath):

    _dict = []
    
    for i in xrange(1,7):
        _file = open(fPath + "/data_batch_" + str(i), "rb")
        _dict.append(cp.load(_file))
        _file.close()        

    return normalize_cifar_set(_dict)
        
def cifar10_test(fPath):
    
    _file = open(fPath + "/test_batch", "rb")
    _dict = cp.load(_file)
    _file.close()

    return normalize_cifar_set([_dict])

    
#####################################################################
# Matlab file
#####################################################################
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

#####################################################################
# Easy loader
#####################################################################

def load_datasets(fName, fType):

    _path = "{0}{1}".format(DATAPATH, fName)        
    
    if fName == "mnist":
        if fType == "train": return mnist_train(_path)
        else:                return mnist_test (_path)

    elif fName == "cifar-10":
        if fType == "train": return cifar10_train(_path)
        else:                return cifar10_test (_path)
            
    else: return [np.loadtxt(_path + "_" + fType + "sets.txt")]
        
#####################################################################
# Input normalization
#####################################################################

# Not finished
def normalization(fName, fSets):

    print "Data preprocessing...\n"

    dy.display(fName,
               "input",
               fSets)
    
    _sets = fSets

    # Mean subtraction
    _shifted = _sets - np.mean(_sets, 0)

    # Covariance
    _cov = np.dot(_shifted.T, _shifted) / _shifted.shape[0]

    # SVD factorisation
    _U, _S, _V = np.linalg.svd(_cov)

    # Decorrelation
    _decorrelated = np.dot(_shifted, _U)

    del(_shifted)
    
    dy.display(fName,
               "decorrelated",
               np.dot(_decorrelated, _U.T))
    
    # Whitening
    _whitened = _decorrelated / np.sqrt(_S + 1e-2)

    del(_decorrelated)
    
    dy.display(fName,
               "whitened",
               np.dot(_whitened, _U.T))

    return _whitened, _U
