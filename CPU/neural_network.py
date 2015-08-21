import math              as mt
import time              as tm
import numpy             as np
import scipy.special     as ss
import matplotlib.pyplot as plt

class NEURAL_NETWORK(object):

    def __init__(self, fNbLayers, fNeurons, fBatchSize):

        # Numbers of layers
        self.mNbLayers       = fNbLayers

        # List of number of neurons per layer
        self.mNeurons        = fNeurons

        # Batch size
        self.mBatchSize      = fBatchSize

        # Regularization terms
        self.mRegu           = 0.001
        
        # Learning rate
        self.mEpsilon        = [0.01] * 2
        self.mLeakControl    = 0.2

        # Sparsity
        self.mBeta           = 3.0
        self.mSparsity       = 0.05

        # Momentum
        self.mMomentum       = [0.5] * 2
        self.mLambda         = 0.2

        # Dropout scaling
        self.mDropoutScaling = 0.5

        # Cross-validation, batch building
        self.mCycle = 6
        self.mSlice = 10000
        self.mPool  = np.arange((self.mCycle-1) * self.mSlice)

        # Weights, biases initialization
        self.mWeights    = []
        self.mBiases     = []
        self.mVariations = []
        
        for i in xrange(self.mNbLayers - 1):

            _mean = 0.0
            _std  = 1. / mt.sqrt(fNeurons[i])
            _size = (fNeurons[i+1], fNeurons[i])

            self.mWeights.append(np.random.normal(_mean,_std,_size))

            self.mBiases.append(np.zeros((fNeurons[i+1], 1)))

            self.mVariations.append(np.zeros(_size))
        
#####################################################################
# SETS PREPARATION AND RESULTS VISUALIZATION
#####################################################################

    # Cross validation set
    def cross_validation(self, fIdx, fImgs, fLbls=None):

        _slice = self.mSlice

        _trainsets = fImgs[0:fIdx * _slice, :]

        _trainsets = np.vstack((_trainsets,
                                fImgs[(fIdx+1)*_slice:len(fImgs),:]))

        _testsets = fImgs[fIdx * _slice:(fIdx + 1) * _slice, :]

        if fLbls is None:
            return _trainsets, _testsets

        #########################################################
        # Only for decision making
            
        _trainlbls = fLbls[0:fIdx * _slice]

        _trainlbls = np.hstack((_trainlbls,
                                fLbls[(fIdx+1)*_slice:len(fLbls)]))
        
        _testlbls = fLbls[fIdx * _slice:(fIdx + 1) * _slice]
        
        return (_trainsets, _trainlbls), (_testsets , _testlbls)
        ###########################################################
        
#####################################################################

    def build_batch(self, fIdx, fImgs, fLbls=None):

        _size = self.mBatchSize
            
        if fIdx == 0:
            np.random.shuffle(self.mPool)
        
        _index = self.mPool[fIdx * _size:(fIdx + 1) * _size]

        _sets = fImgs[_index,:].T

        ###############################################
        # Only for decision making
        if fLbls is not None:
            _lbls = np.zeros((self.mNeurons[-1], _size))

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
        
        _img = np.zeros((self.mNeurons[1], self.mNeurons[0]))
        
        for i in xrange(len(self.mWeights[0])):            

            _row = []
            _sws = 0
            
            for j in xrange(len(self.mWeights[0][i])):
                _sws += self.mWeights[0][i,j]**2
            
            for j in xrange(len(self.mWeights[0][i])):
                _row.append(self.mWeights[0][i,j] / np.sqrt(_sws))

            _img[i,:] = _row

        return _img

#####################################################################
# LOGISTIC FUNCTION
#####################################################################

    def sigmoid_exp(self, fX, fA):
        '''Compute the sigmoid function : 1 / (1 + e^(-x)) + ax
        With a small linear coefficient to avoid flat spot.

        INPUT  : A single value, vector, matrix and small coefficient
        OUTPUT : Sigmoid-value of the given input'''
        
        return ss.expit(fX) + fA * fX

#####################################################################
    
    def dsigmoid_exp(self, fX, fA):
        '''Compute the derived sigmoid function following the 
        derived formula : f'(x) = f(x) (1 - f(x)) + a.

        INPUT  : Sigmoid output and small coefficient
        OUTPUT : The derived value''' 
            
        return fA + fX * (1 - fX)

#####################################################################

    def sigmoid_tanh(self, fX, fA):
        '''Compute the sigmoid function : 1.7159*tanh(2/3 * x) + ax
        With a small linear coefficient to avoid flat spot.

        INPUT  : A single value, vector, matrix and small coefficient
        OUTPUT : Sigmoid-value of the given input'''

        return 1.7159 * np.tanh(2. * fX / 3.) + fA * fX

#####################################################################

    def dsigmoid_tanh(self, fX, fA):
        '''Compute the derived sigmoid function following the 
        derived formula : f'(x) = a + f(x)^2

        INPUT  : Sigmoid output, small coefficient
        OUTPUT : The derived value''' 

        return fA + 1.7159 * 2./3. + 2./3. * fX**2

#####################################################################

    def sigmoid(self, fX):
        '''Compute a sigmoid function with a small linear 
        from developers implementation (see neural_network.py)
        coefficient to avoid flat spot. Sigmoid function depend

        INPUT  : A single value, vector, matrix
        OUTPUT : Sigmoid-value of the given input'''

        return self.sigmoid_exp(fX, 0.001)

#####################################################################

    def dsigmoid(self, fX):
        '''Compute a derived sigmoid function. The derived formula
        depend from developers implementation (see neural_network.py)

        INPUT  : Sigmoid output, small coefficient
        OUTPUT : The derived value''' 

        return self.dsigmoid_exp(fX, 0.001)
    
#####################################################################
# COST PROPAGATION    
#####################################################################
    
    # Compute the average cost obtained with a set of train inputs
    def error(self, fOut, fIn):

        return np.sum(np.square(fOut - fIn)) / 2.

#####################################################################
# ADAPTIVE LEARNING RATE
#####################################################################

    def angle_driven_approach(self, fWgrad):

        for i in xrange(self.mNbLayers - 1):

            _grad = fWgrad[i]
            _var  = self.mVariations[i]
        
            # Angle between previous update and current gradient
            _teta  = np.sum(-_grad * _var)
            _teta /= (np.linalg.norm(_grad) * np.linalg.norm(_var))
        
            # Learning rate update
            self.mEpsilon[i] *= (1 + 0.5 * _teta)

            # Momentum update
            self.mMomentum[i]  = self.mLambda * self.mEpsilon[i]
            self.mMomentum[i] *= np.linalg.norm(_grad)
            self.mMomentum[i] /= np.linalg.norm(_var)

#####################################################################

    def average_gradient_approach(self, fWgrad):

        aga = self.average_gradient_approach.__func__
        if not hasattr(aga, "_avg"):
            aga._avg = [np.zeros(self.mWeights[i].shape)
                        for i in xrange(self.mNbLayers - 1)]

        for i in xrange(self.mNbLayers - 1):

            aga._avg[i] *= (1 - self.mLeakControl)
            aga._avg[i] += self.mLeakControl * fWgrad[i]

            self.mEpsilon[i] = self.mEpsilon[i] + 0.5 * self.mEpsilon[i] * (0.01 * np.linalg.norm(aga._avg[i]) - self.mEpsilon[i])
        
    
#####################################################################
# BACKUP
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
            
            for j in xrange(len(self.mWeights[i])):
                for k in xrange(len(self.mWeights[i][j])):
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
        for i in xrange(len(self.mBiases)):

            print "\t \t -> Layer", i + 1
            _v = np.zeros(self.mBiases[i].shape)
            
            for j in xrange(len(self.mBiases[i])):
                
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
    def gradient_checking(self, fIn, fRef, fWgrad, fBgrad, fBatch):

        self.mBeta = 0
        
        _nGrad = self.numerical_gradient(fIn, fRef, fBatch)
        
        _wError = np.zeros(len(_nGrad[0]))
        _bError = np.zeros(len(_nGrad[1]))
        
        for i in xrange(len(_nGrad[0])):
            _wError[i]  = np.linalg.norm(_nGrad[0][i] - fWgrad[i])
            _wError[i] /= np.linalg.norm(_nGrad[0][i] + fWgrad[i])

        for i in xrange(len(_nGrad[1])):
            _bError[i]  = np.linalg.norm(_nGrad[1][i] - fBgrad[i])
            _bError[i] /= np.linalg.norm(_nGrad[1][i] + fBgrad[i])

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
