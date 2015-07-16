import numpy             as np
import gnumpy            as gpu
import math              as mt
import matplotlib.pyplot as plt

class NEURAL_NETWORK(object):

    def __init__(self, fNbLayers, fNeurons):

        # Learning rate
        self.mEpsilon   = 0.0001

        # Regularization coefficient
        self.mTeta      = 0.005

        # Sparsity parameters
        self.mRho       = 0.1

        # Sparsity coefficient
        self.mBeta      = 3   

        # Momentum
        self.mMomentum  = 0.001
        
        # Numbers of layers
        self.mNbLayers  = fNbLayers

        # List of number of neurons per layer
        self.mNeurons   = fNeurons

        # Cross validation
        self.mIndex     = 0
        self.mSlice     = 10000
        self.mCycle     = 6

        # Random initialization to avoid symetrical evolution
        self.mWeights = []
        self.mDWgrad  = []
        
        for i in xrange(self.mNbLayers - 1):
            _nIn  = fNeurons[i]
            _nOut = fNeurons[i+1]
            
            _min = -4 * mt.sqrt(6. / (_nIn + _nOut + 1))
            _max = -_min
            
            _tmp = gpu.garray(np.random.uniform(_min, _max, (_nOut, _nIn)))

            self.mWeights.append(_tmp)
            self.mDWgrad.append(gpu.zeros((_nOut, _nIn)))

        # Initialization to zeros' vector
        self.mBiases = []

        for i in xrange(1, self.mNbLayers):
            self.mBiases.append(gpu.zeros((fNeurons[i],1)))
    
#####################################################################

    # Compute an approximation of neurons vision from first hidden
    # layer
    def neurons_visions(self):

        print "Building an approximate neurons vision..."
        
        _img = []
            
        for i in xrange(len(self.mWeights[0])):            

            _row = []
            _sws = 0
            
            for j in xrange(len(self.mWeights[0][i])):
                _sws += self.mWeights[0][i,j]**2
            
            for j in xrange(len(self.mWeights[0][i])):
                _row.append(self.mWeights[0][i,j] / np.sqrt(_sws))

            _img.append(_row)

        return _img

#####################################################################
# FOLLOWING METHODS USED FOR COST COMPUTATION    
#####################################################################
    
    # Compute the average cost obtained with a set of train inputs
    def computation_cost(self, fOutput, fInput):

        return gpu.sum((fOutput - fInput)**2) / 2.

#####################################################################
    
    # Compute the regularization cost to avoid weights too big
    def regularization(self):

        _tmp = 0
        
        for w in self.mWeights:
            _tmp = _tmp + gpu.sum(w**2)

        return self.mTeta * _tmp / 2

#####################################################################

    def init_average_list(self):

        return [gpu.zeros((self.mNeurons[i], 1)) for i in xrange(1, self.mNbLayers-1)]

#####################################################################
    
    # Compute the average activation of a neurons
    def average_activation(self, fHid, fSize):

        _avg = []

        for i in xrange(1, self.mNbLayers-1):

            _tmp = fHid[i].mean(1)
            _avg.append(_tmp.reshape(len(_tmp), 1))
        
        return _avg

#####################################################################
    
    # Compute the sparsity cost
    def sparsity(self, fAvg):
        
        _rho = self.mRho
        _tmp = 0

        for rows in fAvg:
            for j in rows:
                _tmp += _rho * mt.log(_rho / j[0])
                _tmp += (1 - _rho) * mt.log((1 - _rho) / (1 - j[0])) 

        return self.mBeta * _tmp

#####################################################################

    # Sum of all costs computed
    def global_cost(self, fCost, fAvg=[]):

        return fCost + self.regularization() + self.sparsity(fAvg)

#####################################################################
# FOLLOWING METHODS USED FOR BACKUP AND MULTILAYERS TRAINING
#####################################################################

    # Save the weights and biases computed in a textfile
    def save_state(self, fName):

        for i in np.arange(len(self.mWeights)):
            _str = "../states/" + fName + "_W" + str(i) + ".txt"
            np.savetxt(_str, self.mWeights[i])

        for i in np.arange(len(self.mBiases)):
            _str = "../states/" + fName + "_B" + str(i) + ".txt"
            np.savetxt(_str, self.mBiases[i])

#####################################################################

    # Load the weights and biases computed from a textfile
    def load_state(self, fName):

        for i in np.arange(len(self.mWeights)):
            _str = "../states/" + fName + "_W" + str(i) + ".txt"
            try:
                self.mWeights[i] = np.loadtxt(_str)
            except IOError:
                print "Random initialization for W{0}".format(i)

        for i in np.arange(len(self.mBiases)):
            _str = "../states/" + fName + "_B" + str(i) + ".txt"
            try:
                self.mBiases[i] = np.expand_dims(np.loadtxt(_str), 1)
            except IOError:
                print "Zero initialization for B{0}".format(i)

#####################################################################
    
    # Use to create a new datasets for next layers
    def save_output(self, fName, fType, fOutput):

        _data = np.array(fOutput)        
        _str  = "../datasets/" + fName + "_" + fType + "sets.txt"

        np.savetxt(_str, fOutput)

#####################################################################
# FOLLOWING METHODS USED FOR CODE SIMPLIFICATIONS
#####################################################################

    # Cross validation set
    def cross_validation(self, fSets):

        _slice = self.mSlice
        k      = self.mIndex % self.mCycle

        _trainsets = fSets[0:k*_slice, :]

        if k == 0:
            _trainsets = fSets[(k+1)*_slice:len(fSets), :]
            
        else:
            np.vstack((_trainsets,
                       fSets[(k+1)*_slice:len(fSets), :]))
    
        _testsets = fSets[k*_slice:(k+1)*_slice, :]

        self.mIndex = k + 1
        
        return gpu.garray(_trainsets), gpu.garray(_testsets)

#####################################################################

    def plot(self, fAbs, fOrd, fName):

        plt.plot(fAbs, fOrd)

        if fName is not None:
            plt.savefig("../img/" + fName)
        else:
            plt.show()

        plt.close()

#####################################################################

    def build_batch(self, fSets, fSize, fIndex=None):

        if fIndex is None:
            fIndex = np.random.randint(len(fSets), size=fSize)

        return fSets[fIndex,:].T
