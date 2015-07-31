from numba import vectorize, float64, float32

import numpy as np

# @vectorize([float32(float32),
#             float64(float64)])
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

#####################################################################

def binary(fDataset, fThresholds=0.5):

    print "Binarization of the datasets..."
    
    return np.array([[1 if j > fThresholds else 0 for j in i] for i in fDataset])
