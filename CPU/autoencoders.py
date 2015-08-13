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
    def gradient(self, fBatch, fErr, fOut):

        _wGrad = []
        _bGrad = []

        for err, out in zip(fErr, fOut):
            _wGrad.append(np.dot(err, out.T) / fBatch)
            _bGrad.append(err.mean(1, keepdims=True))
            
        return _wGrad, _bGrad
    
#####################################################################
    
    # Update weights and biases parameters with gradient descent
    def update(self, fWgrad, fBgrad):

        for i in xrange(self.mNbLayers-1):

            # Weight variation
            self.mWvar[i]    *= self.mMomentum
            self.mWvar[i]    -= self.mEpsilon * fWgrad[i]
            
            # Update weights
            self.mWeights[i] += self.mWvar[i]

            # Update biases
            self.mBiases[i]  += self.mEpsilon * fBgrad[i]

#####################################################################
    
    # One step of the training algorithm
    def train_one_step(self, fBatch, fIn, fRef):

        # Activation propagation
        _out  = self.propagation(fIn)

        # Local error for each layer
        _err = self.compute_layer_error(_out, fRef)
        
        # Gradient for stochastic gradient descent    
        _wGrad, _bGrad = self.gradient(fBatch, _err, _out)

        return (_out, _wGrad, _bGrad)
    
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fImgs, fLbls, fIter, fBatch, fName):

        print "Training...\n"

        _gcost = []
        _gtime = []
        
        _done  = fIter

        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)

            for j in xrange(self.mCycle):

                _trn, _tst = self.cross_validation(j, fImgs)

                for k in xrange(len(_trn) / fBatch):
                    
                    # Inputs and labels batch
                    _in = self.build_batch(fBatch, k, _trn)
                    
                    # One training step
                    _ret = self.train_one_step(fBatch, _in, _in)
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # _grad = self.numerical_gradient(_in,_in,fBatch)

                    # self.gradient_checking(_grad[0], _grad[1],
                    #                        _ret[1] , _ret[2])
                    
                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_ret[1])

                    # Update weights and biases
                    self.update(_ret[1], _ret[2])

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
        
        return self.save_output(fName, "train", fImgs)
        
#####################################################################

    def evaluate(self, fTests):

        _cost = 0

        for i in xrange(len(fTests)):
            _in    = fTests[[i],:].T
            _out   = self.propagation(_in)
            _cost += self.error(_out[-1], _in)

        return _cost / len(fTests)
        
#####################################################################

    # Test the neural network over a test set
    def test(self, fImgs, fLbls, fName):

        print "Testing the neural networks..."

        _res     = []
        _cost    = 0
        
        for i in xrange(len(fImgs)):

            _in    = fImgs[[i],:].T
            _out   = self.propagation(_in)
            _cost += self.error(_out[-1], _in)

            _res.append(_tmp[-1])

        print "Cost {0}\n".format(_cost / len(fImgs))

        # Save output in order to have a testset for next layers
        self.save_output(fName, "test", fImgs)
        
        # Check if it's possible to print the image
        _psize = [int(mt.sqrt(self.mNeurons[0])) for i in xrange(2)]

        if self.mNeurons[0] != (_psize[0] * _psize[1]):
            return
        
        # Displaying the results
        dy.display(fName, [fImgs, _res], len(fImgs), _psize, "out")

        # Approximated vision of first hidden layer neurons
        _res = self.neurons_visions()
        dy.display(fName, [_res], self.mNeurons[1], _psize,
                   "neurons", 5, 5)
