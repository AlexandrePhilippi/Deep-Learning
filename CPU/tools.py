import numpy as np

def sigmoid(fX, fCoef=1):
    return 1. / (1 + np.exp(-fCoef * fX))

#####################################################################

def binary(fDataset, fThresholds=0.5):

    print "Binarization of the datasets..."
    
    return np.array([[1 if j > fThresholds else 0 for j in i] for i in fDataset])
