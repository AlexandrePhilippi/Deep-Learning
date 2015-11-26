from array   import array
from numpy   import asarray
from struct  import unpack
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
            raise ValueError('Magic number mismatch, expected 2049,''got %d' % _magic)

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
            raise ValueError('Magic number mismatch, expected 2051,''got %d' % _magic)

        _imgData = array("B", file.read())

        return asarray(_imgData).reshape(_size, _cols*_rows)/255.

def mnist_train(fPath):
    return [mnist_train_img(fPath), mnist_train_lbl(fPath)]

def mnist_test(fPath):
    return [mnist_test_img(fPath), mnist_test_lbl(fPath)]
