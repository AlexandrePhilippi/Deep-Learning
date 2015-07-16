import random            as rd
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm

# Used to display testsets and output together
def display(fName, fSrc, fNbPatches, fDim, fType=None, fNbImgR=10, fNbImgC=10):

    _firstRow  = 1
    
    for i in np.arange(fNbImgR):

        _firstCell = 1

        for j in np.arange(fNbImgC / len(fSrc)):
            
            _id = rd.randint(0, fNbPatches - 1)

            for k in np.arange(len(fSrc)):

                if(_firstCell == 1):
                    _firstCell = 0
                    _hl = np.reshape(fSrc[k][_id], fDim)

                else:
                    _hl = np.hstack((_hl, np.reshape(fSrc[k][_id], fDim)))

                if j < fNbImgC - 1 or k < (len(fSrc) - 1):
                    _hl = np.hstack((_hl, np.zeros((fDim[0], 1))))

        if(_firstRow == 1):              
            _firstRow = 0
            _vl = _hl
                
        else:
            _vl = np.vstack((_vl, _hl))
                
        if i < fNbImgR - 1:
            _vl = np.vstack((_vl, np.zeros(len(_vl[0]))))
                

    plt.imshow(_vl, cmap=cm.Greys_r)

    if fName is not None and fType is not None:
        plt.savefig("img/" + fName + "_" + fType + ".png")
    else:
        plt.show()

    plt.close()
