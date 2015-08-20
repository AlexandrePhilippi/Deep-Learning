import math              as mt
import numpy             as np
import random            as rd
import matplotlib.cm     as cm
import matplotlib.pyplot as plt

# Used to display testsets and output together
def display(fName, fType, fSrc, fNbImgR=10, fNbImgC=10):

    if int(mt.sqrt(fSrc[0].shape[1])) != mt.sqrt(fSrc[0].shape[1]):
        print "These images are not squares.\n"; return

    _dim       = [mt.sqrt(fSrc[0].shape[1])] * 2
    _firstRow  = 1

    for i in np.arange(fNbImgR):

        _firstCell = 1

        for j in np.arange(fNbImgC / len(fSrc)):
            
            _id = rd.randint(0, len(fSrc[0]) - 1)

            for k in np.arange(len(fSrc)):

                if(_firstCell == 1):
                    _firstCell = 0
                    _hl = np.reshape(fSrc[k][_id], _dim)

                else:
                    _hl = np.hstack((_hl, np.reshape(fSrc[k][_id],
                                                     _dim)))

                if j < fNbImgC - 1 or k < (len(fSrc) - 1):
                    _hl = np.hstack((_hl, np.zeros((_dim[0], 1))))

        if(_firstRow == 1):              
            _firstRow = 0
            _vl = _hl
                
        else:
            _vl = np.vstack((_vl, _hl))
                
        if i < fNbImgR - 1:
            _vl = np.vstack((_vl, np.zeros(len(_vl[0]))))
                

    plt.imshow(_vl, cmap=cm.Greys_r)

    if fName is not None and fType is not None:
        plt.savefig("../img/" + fName + "_" + fType + ".png")
    else:
        plt.show()

    plt.close()
