import time   as tm
import math   as mt
import numpy  as np

from scipy.special import expit

class Autoencoders(object):

    def __init__(self,
                 fLayers,
                 fLearningRate,
                 fMomentum,
                 fSparsityTarget,
                 fSparsityCoefficient):

        # Number of layers
        self.mNbLayers = len(fLayers)

        # List of number of neurons per layer
        self.mLayers = fLayers

        # Learning rate
        self.mLearningRate = np.ones(self.mNbLayers - 1) * fLearningRate

        # Momentum
        self.mMomentum = np.ones(self.mNbLayers - 1) * fMomentum

        # Sparisty parameters
        self.mSparsityTarget      = fSparsityTarget
        self.mSparsityCoefficient = fSparsityCoefficient

        # List of weight matrix 
        self.mWeights = []

        # List of bias vector
        self.mBiases = []

        # List of variation matrix
        self.mVariations = []

        # Initialization of both previous list
        for i in xrange(self.mNbLayers - 1):

            _mean = 0.0
            _std  = 1. / mt.sqrt(fLayers[i])
            _size = (fLayers[i+1], fLayers[i])

            self.mBiases.append(np.zeros((fLayers[i+1], 1)))
            self.mWeights.append(np.random.normal(_mean, _std, _size))
            self.mVariations.append(np.zeros(_size))

########################################################################

    def propagation(self, fInput):

        _out = [fInput]

        for i in xrange(self.mNbLayers - 1):
            _out.append(expit(np.dot(self.mWeights[i], _out[-1]) + self.mBiases[i]))

        return _out

########################################################################

    def average_activation(self, fActivation):

        return np.mean(fActivation, axis=1, keepdims=True)

########################################################################

    def sparsity(self, fActivation):

        _average  = self.average_activation(fActivation)

        return self.mSparsityCoefficient * (-(self.mSparsityTarget /_average) + ((1 - self.mSparsityTarget) / (1 - _average)))

########################################################################

    def backpropagation(self, fOut):

        _err = [-(fOut[0] - fOut[-1]) * fOut[-1] * (1 - fOut[-1])]

        for i in xrange(self.mNbLayers - 2):

            _propagation = np.dot(self.mWeights[-i-1].transpose(), _err[-1])
            _derivative  = fOut[-i-2] * (1 - fOut[-i-2])
            _sparsity    = self.sparsity(fOut[-i-2])

            _err.append((_propagation + _sparsity) * _derivative)

        _err.reverse()

        return _err

########################################################################

    def partial_derivatives(self, fErr, fOut, fBatchsize):

        _dWeights = []
        _dBiases  = []

        for i in xrange(self.mNbLayers - 1):
            _dWeights.append(np.dot(fErr[i], fOut[i].transpose()) / fBatchsize)
            _dBiases.append(fErr[i].mean(1, keepdims=True))

        return _dWeights, _dBiases

########################################################################

    def update(self, fDWeights, fDBiases):
        
        for i in xrange(self.mNbLayers - 1):

            self.mVariations[i] *= self.mMomentum[i]
            self.mVariations[i] -= self.mLearningRate[i] * fDWeights[i]

            self.mWeights[i] += self.mVariations[i]
            self.mBiases[i]  -= self.mLearningRate[i] * fDBiases[i]

########################################################################
    
    def error(self, fOut, fIn, fBatchsize):

        return np.sum(np.square(fOut - fIn)) / (2 * fBatchsize)

########################################################################
    
    def angle_driven_approach(self, fDWeights):

        for i in xrange(self.mNbLayers - 1):

            # Angle between vector of previous direction update and current direction update
            _teta = np.sum(-self.mVariations[i] * fDWeights[i]) / (np.linalg.norm(fDWeights[i]) * np.linalg.norm(self.mVariations[i]))

            # Learning rate update
            self.mLearningRate[i] *= (1 + 0.5 * _teta)

            # Momentum coefficient update
            self.mMomentum[i] = 0.5 * self.mLearningRate[i] * np.linalg.norm(fDWeights[i]) / np.linalg.norm(self.mVariations[i])

########################################################################
            
    def cross_validation(self, fData):

        np.random.shuffle(fData)

        _refsize = 3*round(len(fData) * 20 / 100)
        
        _trainset = fData[0:_refsize, :]
        _crossset = fData[_refsize:len(fData), :]

        return _trainset, _crossset
        
########################################################################

    def train(self, fEpochs, fData, fBatchsize):

        print "Training the network...\n"

        _traincost = []; _crosscost = []; _time = []
        
        for i in xrange(fEpochs):

            _traincost.append(0); _crosscost.append(0); _time.append(tm.time())

            _trainset, _crossset = self.cross_validation(fData)
            
            for j in xrange(len(_trainset) / fBatchsize):

                _in  = _trainset[j*fBatchsize:(j+1)*fBatchsize, :].transpose()

                _out = self.propagation(_in)

                _err = self.backpropagation(_out)

                _dWeights, _dBiases = self.partial_derivatives(_err, _out, fBatchsize)

                if(i > 0 or j > 0):
                    self.angle_driven_approach(_dWeights)
                    
                self.update(_dWeights, _dBiases)

                _traincost[i] += self.error(_out[-1], _in, len(_trainset))

            _crosscost[i]  = self.evaluate(_crossset)
            _time[i]       = tm.time() - _time[i]
            
            print "Epochs:", i, "Time:", _time[i], "Train Cost:", _traincost[i], "Cross Cost:", _crosscost[i]
            print "Learning Rate:", self.mLearningRate, "Momentum:", self.mMomentum

########################################################################
            
    def evaluate(self, fData):

        _in  = fData.transpose()
        _out = self.propagation(_in)
        
        return self.error(_out[-1], _in, len(fData))

########################################################################

    def test(self, fData):

        print "Testing the network...\n"

        _out  = [];
        _cost = 0;

        for i in xrange(len(fData)):

            _in = fData[i:i+1,:].transpose()

            _out.append(self.propagation(_in)[-1])

            _cost += self.error(_out[-1], _in, 1)

        print "Final cost :", _cost / len(fData)

        return _out

########################################################################

    def neurons_vision(self):

        print "Building an approximate first hidden layer neurons vision...\n"

        _img = np.zeros((self.mLayers[1], self.mLayers[0]))

        for i in xrange(len(self.mWeights[0])):

            _row = []
            _sws = 0

            for j in xrange(len(self.mWeights[0][i])):
                _sws += self.mWeights[0][i,j]**2

            for j in xrange(len(self.mWeights[0][i])):
                _row.append(self.mWeights[0][i,j] / np.sqrt(_sws))

            _img[i,:] = _row

        return _img
