import numpy             as np
import math              as mt
import time              as tm
import matplotlib.pyplot as plt

class NEURAL_NETWORK(object):

    def __init__(self, fNbLayers, fNeurons):

        # Numbers of layers
        self.mNbLayers  = fNbLayers

        # List of number of neurons per layer
        self.mNeurons   = fNeurons

        # Learning rate
        self.mEpsilon   = 0.01

        # Sparsity
        self.mBeta     = 3.0
        self.mSparsity = 0.05
        
        # Momentum
        self.mMomentum  = 0.5
        self.mLambda    = 0.2
        
        # Neural network inner parameters initialization
        self.mWeights  = []
        self.mBiases   = []
        self.mWvar     = []

        # Cross-validation, batch building
        self.mCycle    = 6
        self.mSlice    = 10000
        self.mPool     = np.arange(50000)
        
        for i in xrange(self.mNbLayers-1):

            _nIn  = fNeurons[i]
            _nOut = fNeurons[i+1]
            
            _min = -mt.sqrt(6. / (_nIn + _nOut + 1))
            _max = -_min

            # Weights random initialization
            _rand = np.random.RandomState(int(tm.time()))
            _size = (_nOut, _nIn)

            self.mWeights.append(np.asarray(_rand.uniform(_min,
                                                          _max,
                                                          _size)))

            # Biases initialization to zeros' vector
            self.mBiases.append(np.zeros((_nOut, 1)))

            # Weights variations
            self.mWvar.append(np.zeros(self.mWeights[i].shape))

#####################################################################
# SETS PREPARATION AND RESULTS VISUALIZATION
#####################################################################

    # Cross validation set
    def cross_validation(self, fIdx, fSets, fLbls=None):

        _slice = self.mSlice

        _trainsets = fSets[0:fIdx * _slice, :]

        _trainsets = np.vstack((_trainsets,
                                fSets[(fIdx+1)*_slice:len(fSets),:]))

        _testsets = fSets[fIdx * _slice:(fIdx + 1) * _slice, :]

        #########################################################
        # Only for decision making
        if fLbls is not None:
            _trainlbls = fLbls[0:fIdx * _slice]
            
            if fIdx == 0:
                _trainlbls = fLbls[(fIdx + 1) * _slice:len(fLbls)]

            else:
                np.hstack((_trainlbls,
                           fLbls[(fIdx + 1) * _slice:len(fLbls)]))

            _testlbls = fLbls[fIdx * _slice:(fIdx + 1) * _slice]

            return (_trainsets, _trainlbls), (_testsets , _testlbls)
        ###########################################################
                
        return _trainsets, _testsets

#####################################################################

    def build_batch(self, fSize, fIdx, fSets, fLbls=None):

        if fIdx == 0:
            np.random.shuffle(self.mPool)
        
        _index = self.mPool[fIdx * fSize:(fIdx + 1) * fSize]

        _sets = fSets[_index,:].T

        ###############################################
        # Only for decision making
        if fLbls is not None:
            _lbls = np.zeros((self.mNeurons[-1], fSize))

            for l in xrange(fSize):
                _lbls[fLbls[_index[l]]][l] = 1

            return _sets, _lbls
        ###############################################
                
        return _sets
    
#####################################################################

    def plot(self, fAbs, fOrd, fName, fType):

        plt.plot(fAbs, fOrd)

        if fName is not None:
            plt.savefig("../img/" + fName + fType)
        else:
            plt.show()

        plt.close()

#####################################################################
            
    # Neurons vision from first hidden layer
    def neurons_visions(self):

        print "Building an approximate neurons vision...\n"
        
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
# COST PROPAGATION    
#####################################################################
    
    # Compute the average cost obtained with a set of train inputs
    def error(self, fOut, fIn):

        return np.sum((fOut - fIn)**2) / 2.

#####################################################################
# ADAPTIVE LEARNING RATE
#####################################################################
        
    def grad_dir_angle(self, fWvar, fWgrad):

        return np.sum(-fWgrad * fWvar) / (np.linalg.norm(fWgrad) * np.linalg.norm(fWvar))

#####################################################################
    
    def angle_driven_approach(self, fWgrad):

        # Learning rate update
        self.mEpsilon = self.mEpsilon * (1 + 0.5 * self.grad_dir_angle(self.mWvar[-1], fWgrad[-1]))

        # Momentum update
        self.mMomentum = self.mLambda * self.mEpsilon * np.linalg.norm(fWgrad[-1]) / np.linalg.norm(self.mWvar[-1])
    
#####################################################################
# BACKUP
#####################################################################

    # Save the weights and biases computed in a textfile
    def save_state(self, fName):

        if fName is None:
            return
        
        for i in np.arange(len(self.mWeights)):
            _str = "../states/" + fName + "_W" + str(i) + ".txt"
            np.savetxt(_str, self.mWeights[i])

        for i in np.arange(len(self.mBiases)):
            _str = "../states/" + fName + "_B" + str(i) + ".txt"
            np.savetxt(_str, self.mBiases[i])

#####################################################################

    # Load the weights and biases computed from a textfile
    def load_state(self, fName):

        if fName is None:
            return
        
        for i in xrange(len(self.mWeights)):
            _str = "../states/" + fName + "_W" + str(i) + ".txt"
            try:
                self.mWeights[i] = np.loadtxt(_str)
            except IOError:
                print "Random initialization for W{0}".format(i)

        for i in xrange(len(self.mBiases)):
            _str = "../states/" + fName + "_B" + str(i) + ".txt"
            try:
                self.mBiases[i] = np.expand_dims(np.loadtxt(_str), 1)
            except IOError:
                print "Zero initialization for B{0}".format(i)

#####################################################################
    
    # Use to create a new datasets for next layers
    def save_output(self, fName, fType, fImgs):

        _out = np.empty((len(fImgs),self.mNeurons[1]))
        
        for i in xrange(len(fImgs)):
            _out[[i],:] = self.propagation(fImgs[[i],:].T)[1].T

        np.savetxt("../datasets/"+fName+"_"+fType+"sets.txt", _out)

#####################################################################
# VERIFICATIONS
#####################################################################
    
    # Compute numerical gradient value in order to check results
    def numerical_gradient(self, fInput, fRef, fBatch):

        _epsilon  = 0.00001
        
        _numWgrad = []
        _numBgrad = []

        # Numerical gradient according to W
        print "\t Numerical gradient according to Weights."
        for i in xrange(len(self.mWeights)):

            print "\t \t -> Layer", i + 1
            _m = np.zeros(self.mWeights[i].shape)
            
            for j in np.arange(len(self.mWeights[i])):
                for k in np.arange(len(self.mWeights[i][j])):
                    self.mWeights[i][j,k] += _epsilon
                    _left = self.output_and_cost(fInput, fRef)

                    self.mWeights[i][j,k] -= 2. * _epsilon
                    _right = self.output_and_cost(fInput, fRef)

                    _res = (_left[1] - _right[1]) / (2. * _epsilon)
                    _m[j][k] = _res / fBatch
                    
                    self.mWeights[i][j,k] += _epsilon

            _numWgrad.append(_m)

        # Numerical gradient according to b
        print "\t Numerical gradient according to Biases."    
        for i in np.arange(len(self.mBiases)):

            print "\t \t -> Layer", i + 1
            _v = np.zeros(self.mBiases[i].shape)
            
            for j in np.arange(len(self.mBiases[i])):
            
                self.mBiases[i][j] += _epsilon
                _left = self.output_and_cost(fInput, fRef)

                self.mBiases[i][j] -= 2. * _epsilon
                _right = self.output_and_cost(fInput, fRef)

                _res  = (_left[1] - _right[1]) / (2. * _epsilon)
                _v[j] = _res / fBatch
                
                self.mBiases[i][j] += _epsilon

            _numBgrad.append(_v)
                      
        return _numWgrad, _numBgrad

#####################################################################
    
    # Check gradient results
    def gradient_checking(self, _nWgrad, _nBgrad, _wGrad, _bGrad):

        _wError = np.zeros(len(_nWgrad))
        _bError = np.zeros(len(_nBgrad))
        
        for i in xrange(len(_nWgrad)):
            _wError[i]  = np.linalg.norm(_nWgrad[i] - _wGrad[i]) / np.linalg.norm(_nWgrad[i] + _wGrad[i])

        for i in xrange(len(_nBgrad)):
            _bError[i]  = np.linalg.norm(_nBgrad[i] - _bGrad[i]) / np.linalg.norm(_nBgrad[i] + _bGrad[i])

        print _wError
        print _bError

#####################################################################
    
    # One step of the train algorithm to get output and cost
    def output_and_cost(self, fIn, fRef):

        # All the output generated according to the batch
        _out  = self.propagation(fIn)
        
        # Cost linked to the batch passed in argument
        _cost = self.error(_out[-1], fRef)

        return _out, _cost
