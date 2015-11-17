from array   import array
from numpy   import asarray, zeros
from struct  import unpack
from cPickle import load
from os.path import join

# Global parameters
DATAPATH = "../datasets/"

##################################################################
# MNIST database
##################################################################

def mnist_test_lbl(fPath):
    _path = join(fPath, 't10k-labels-idx1-ubyte')
    return load_mnist_lbl(_path)

def mnist_train_lbl(fPath):
    _path = join(fPath, 'train-labels-idx1-ubyte')
    return load_mnist_lbl(_path)

def load_mnist_lbl(fPath):
    with open(fPath, 'rb') as file:
        _magic, _size = unpack(">II", file.read(8))

        if _magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049, got %d' % _magic)

        return asarray(array("B", file.read()))

def mnist_test_img(fPath):
    _path = join(fPath, 't10k-images-idx3-ubyte')
    return load_mnist_img(_path)

def mnist_train_img(fPath):
    _path = join(fPath, 'train-images-idx3-ubyte')
    return load_mnist_img(_path)

def load_mnist_img(fPath):
    with open(fPath, 'rb') as file:
        _magic, _size, _rows, _cols = unpack(">IIII", file.read(16))

        if _magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got %d' % _magic)

        return asarray(array("B", file.read())).reshape(_size, _cols * _rows) / 255.

def mnist_train(fPath):
    return [mnist_train_img(fPath), mnist_train_lbl(fPath)]

def mnist_test(fPath):
    return [mnist_test_img(fPath), mnist_test_lbl(fPath)]

##################################################################
# Cifar-10 file
##################################################################

def normalize_cifar_set(fDict):

    _imgs = zeros((10000 * len(fDict), 1024))
    _lbls = zeros( 10000 * len(fDict))

    for i in xrange(len(fDict)):

        for _type, _dt in fDict[i].items():

            if _type == "data":
                _imgs[i*len(_dt):(i+1)*len(_dt),:] = (_dt[:,0:1024] / 3. + _dt[:,1024:2048] / 3. + _dt[:,2048:3072] / 3.)

            elif _type == "labels":
                _lbls[i*len(_dt):(i+1)*len(_dt)] = _dt

    return _imgs / 255., _lbls

def cifar10_train(fPath):

    _dict = []

    for i in xrange(1,7):
        _file = open(fPath + "/data_batch_" + str(i), "rb")
        _dict.append(load(_file))
        _file.close()

    return normalize_cifar_set(_dict)

def cifar10_test(fPath):
    _file = open(fPath + "/test_batch", "rb")
    _dict = load(_file)
    _file.close()

    return normalize_cifar_set([_dict])
