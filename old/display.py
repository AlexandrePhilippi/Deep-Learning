import math              as mt
import numpy             as np
import random            as rd
import matplotlib.cm     as cm
import matplotlib.pyplot as plt

# Global parameters
IMGPATH = "../img/"

def display(fName, fType, *fSrc):

    _nbImgs = fSrc[0].shape[0] * len(fSrc)
    _length = fSrc[0].shape[1]

    # Are images squares ?
    if int(mt.sqrt(_length) != mt.sqrt(_length)):
        print "These images are not squares.\n"; return

    else:
        _size = [int(mt.sqrt(_length))] * 2

    # Limit of 400 images printed at the same time
    if int(mt.sqrt(_nbImgs)) > 20:
        _nbImgsR = _nbImgsC = 20

    else:
        _nbImgsR = _nbImgsC = int(mt.sqrt(_nbImgs))
        
    # Number of columns multiple of number of sources
    if _nbImgsC % len(fSrc) != 0:
        _nbImgsC += _nbImgsC % len(fSrc)

    # Building images
    _firstRow = True
    _pool     = [-1]
    _id       = -1
    
    for i in xrange(_nbImgsR):

        _firstCell = True

        for j in xrange(_nbImgsC / len(fSrc)):

            while _id in _pool:
                _id = rd.randint(0, len(fSrc[0])-1)

            _pool.append(_id)

            for k in xrange(len(fSrc)):

                if _firstCell:
                    _firstCell = False
                    _hl = np.reshape(fSrc[k][_id], _size)

                else:
                    _hl = np.hstack((_hl, np.reshape(fSrc[k][_id],
                                                     _size)))

                if j < _nbImgsC - 1 or k < (len(fSrc) - 1):
                    _hl = np.hstack((_hl, np.zeros((_size[0], 1))))

        if _firstRow:
            _firstRow = False
            _vl = _hl
                
        else:
            _vl = np.vstack((_vl, _hl))
                
        if i < _nbImgsR - 1:
            _vl = np.vstack((_vl, np.zeros(len(_vl[0]))))

    plt.imshow(_vl, cmap=cm.Greys_r)
        
    if fName is not None and fType is not None:
        plt.savefig("../img/" + fName + "_" + fType + ".png")
    else:
        plt.show()

    plt.close()

######################################################################
    
def plot(fAbs, fOrd, fName, fType):
    '''With a name given it saves the plot, without it just
    prints the plot.
    
    fType correspond to a specification for the file saved (output, 
    neurons visions etc.)
    
    INPUT  : Abscissas, ordinates, name and type (both not necessary)
    OUTPUT : Nothing'''
    
    plt.plot(fAbs, fOrd)
    
    if fName is not None:
        plt.savefig(IMGPATH + fName + fType)
    else:
        plt.show()
        
    plt.close()
