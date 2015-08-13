from neural_network import NEURAL_NETWORK

import sys
import numpy         as np
import time          as tm
import display       as dy
import math          as mt
import scipy.special as ss

class AUTOENCODERS(NEURAL_NETWORK):
    
    def __init__(self, fNeurons):

        # Mother class initialization
        NEURAL_NETWORK.__init__(self, len(fNeurons), fNeurons)

#####################################################################

    # Compute all the layer output for a given input (can be a batch)
    def propagation(self, fInput):

        _out = [fInput]

        for w,b in zip(self.mWeights, self.mBiases):
            _out.append(ss.expit(np.dot(w, _out[-1]) + b))

        return _out

#####################################################################

    def sparsity(self, fOut):

        _avg = fOut.mean(1, keepdims=True)

        return self.mBeta * (-self.mSparsity /_avg + (1. -self.mSparsity) / (1. - _avg))
    
#####################################################################
    
    # Compute all the locals error in order to compute gradient
    def compute_layer_error(self, fOut, fIn):

        # Last layer local error
        _err = [-(fIn - fOut[-1]) * fOut[-1] * (1 - fOut[-1])]

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers-1):

            _backprop  = np.dot(self.mWeights[-i].T, _err[i-1])

            _sparsity  = self.sparsity(fOut[-i-1])
            
            _dsigmoid  = fOut[-i-1] * (1 - fOut[-i-1])
            
            _err.append((_backprop + _sparsity) * _dsigmoid) 

        _err.reverse()

        return _err

#####################################################################

    # Compute the gradient according to W and b in order to
    # realize the mini-batch gradient descent    
    def gradient(self, fSize, fErr, fOut):

        _wGrad = []
        _bGrad = []

        for err, out in zip(fErr, fOut):
            _wGrad.append(np.dot(err, out.T) / fSize)
            _bGrad.append(err.mean(1, keepdims=True))
            
        return _wGrad, _bGrad
    
#####################################################################

    def weight_variation(self, fWvar, fWgrad):

        return [-self.mEpsilon * fWgrad[i] + self.mMomentum * fWvar[i] for i in xrange(self.mNbLayers-1)]

#####################################################################
    
    # Update weights and biases parameters with gradient descent
    def update(self, fWvar, fBgrad):

        for i in xrange(self.mNbLayers-1):
            
            # Update weights
            self.mWeights[i] += fWvar[i]

            # Update biases
            self.mBiases[i]  += self.mEpsilon * fBgrad[i]

#####################################################################
    
    # One step of the training algorithm
    def train_one_step(self, fSize, fIn, fRef):

        # Activation propagation
        _out  = self.propagation(fIn)

        # Local error for each layer
        _err = self.compute_layer_error(_out, fRef)
        
        # Gradient for stochastic gradient descent    
        _wGrad, _bGrad = self.gradient(fSize, _err, _out)

        return (_out, _wGrad, _bGrad)
    
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName):

        print "Training...\n"

        _Wvar  = [np.zeros(_w.shape) for _w in self.mWeights]
        _sets  = fSets[0]
        _gcost = []
        _gtime = []
        
        _done  = fIter

        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)

            for j in xrange(self.mCycle):

                _trn, _tst = self.cross_validation(j, _sets)

                for k in xrange(len(_trn) / fSize):
                    
                    # Inputs and labels batch
                    _in = self.build_batch(fSize, k, _trn)
                    
                    # One training step
                    _ret = self.train_one_step(fSize, _in, _in)
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # _grad = self.numerical_gradient(_in,_in,fSize)

                    # self.gradient_checking(_grad[0], _grad[1],
                    #                        _ret[1] , _ret[2])
                    
                    # Adapt learning rate
                    # if(i > 0 or j > 0 or k > 0):
                    #     self.angle_driven_approach(_Wvar, _ret[1])

                    # Weight variation computation
                    _Wvar = self.weight_variation(_Wvar, _ret[1])
                        
                    # Update weights and biases
                    self.update(_Wvar, _ret[2])

                _gcost[i] += self.evaluate(_tst)

            # Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])    

            # Global cost for one cycle
            _gcost[i] /= self.mCycle
            print "Cost of iteration : {0}".format(_gcost[i])

            # Parameters
            print "Epsilon {0} Momentum {1}\n".format(self.mEpsilon,
                                                      self.mMomentum)

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i]) < 0.001):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")
        
        return self.create_datasets(_sets)
        
#####################################################################

    def evaluate(self, fTests):

        _cost = 0

        for i in xrange(len(fTests)):
            _in    = fTests[[i],:].T
            _out   = self.propagation(_in)
            _cost += self.propagation_cost(_out[-1], _in)

        return _cost / len(fTests)
        
#####################################################################

    # Test the neural network over a test set
    def test(self, fSets, fName):

        print "Testing the neural networks..."

        _sets = np.empty((len(fSets), self.mNeurons[1]))
        _out  = []
        
        _cost = 0
        
        for i in xrange(len(fSets)):

            _in    = fSets[[i],:].T
            _tmp   = self.propagation(_in)
            _cost += self.propagation_cost(_tmp[-1], _in)

            _out.append(_tmp[-1])
            _sets[[i],:] = _tmp[1].T

        _cost = _cost / len(fSets)
        print "Cost {0}\n".format(_cost)

        # Save output in order to have a testset for next layers
        self.save_output(fName, "test", _sets)
        
        # Check if it's possible to print the image
        _psize = [int(mt.sqrt(self.mNeurons[0])) for i in xrange(2)]

        if self.mNeurons[0] != (_psize[0] * _psize[1]):
            return
        
        # Displaying the results
        dy.display(fName, [fSets, _out], len(fSets), _psize, "out")

        # Approximated vision of first hidden layer neurons
        _res = self.neurons_visions()
        dy.display(fName, [_res], self.mNeurons[1], _psize,
                   "neurons", 5, 5)

#####################################################################
# BACKUP AND DEEP NETWORK PRE-TRAINING
#####################################################################
        
    def create_datasets(self, fSets):

        _out = np.empty((len(fSets),self.mNeurons[1]))
        
        for i in xrange(len(fSets)):
            _out[[i],:] = self.propagation(fSets[[i],:].T)[1].T

        return _out

#####################################################################
# ADAPTIVE LEARNING RATE
#####################################################################
        
    def grad_dir_angle(self, fWvar, fWgrad):

        return np.sum(-fWgrad * fWvar) / (np.linalg.norm(fWgrad) * np.linalg.norm(fWvar))

#####################################################################
    
    def angle_driven_approach(self, fWvar, fWgrad):

        # Learning rate update
        self.mEpsilon = self.mEpsilon * (1 + 0.5 * self.grad_dir_angle(fWvar[-1], fWgrad[-1]))

        # Momentum update
        self.mMomentum = self.mLambda * self.mEpsilon * np.linalg.norm(fWgrad[-1]) / np.linalg.norm(fWvar[-1])

#####################################################################
# VERIFICATIONS
#####################################################################
    
    # Compute numerical gradient value in order to check results
    def numerical_gradient(self, fInput, fRef, fSize):

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
                    _m[j][k] = _res / fSize
                    
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
                _v[j] = _res / fSize
                
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
        _cost = self.propagation_cost(_out[-1], fRef)

        return _out, _cost
