from neural_network import NEURAL_NETWORK

import sys
import time    as tm
import math    as mt
import numpy   as np
import loader  as ld
import display as dy

class AUTOENCODERS(NEURAL_NETWORK):
    
    def __init__(self, fNeurons):

        # Mother class initialization
        NEURAL_NETWORK.__init__(self, len(fNeurons), fNeurons)

#####################################################################

    def dropout_propagation(self, fInput):
        '''Propagation of the input (can be a minibatch) throughout
        the neural network. A dropout system avoid the overfitting.
        The dropout scaling parameters can be modified in neural 
        network parameters.

        INPUT : Vector or matrix
        OUPUT : Neurons activation'''
        
        _out = [fInput]
        _p   = self.mDropoutScaling
        
        for w,b in zip(self.mWeights, self.mBiases):

            _activation  = self.sigmoid(np.dot(w, _out[-1]) + b)
            _drop = (np.random.rand(*_activation.shape) < _p) / _p
            _activation *= _drop

            _out.append(_activation)

        return _out
        
#####################################################################

    def propagation(self, fInput):
        '''Propagation of the input (can be a minibatch) throughout
        the neural network.
 
        INPUT : Vector or matrix
        OUPUT : Neurons activation'''

        _out = [fInput]
        
        for w,b in zip(self.mWeights, self.mBiases):
            _out.append(self.sigmoid(np.dot(w, _out[-1]) + b))

        return _out
    
#####################################################################
    
    # Compute all the locals error in order to compute gradient
    def compute_layer_error(self, fOut, fIn):

        # Last layer local error
        _err = [-(fIn - fOut[-1]) * self.dsigmoid(fOut[-1])]

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers-1):

            _backprop  = np.dot(self.mWeights[-i].T, _err[i-1])

            _dsigmoid  = self.dsigmoid(fOut[-i-1])

            _err.append(_backprop * _dsigmoid)

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
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fImgs, fLbls, fIter, fBatch, fName):

        print "Training...\n"

        # fImgs = ld.normalization(fImgs)

        _gcost = []
        _gtime = []
        
        _done  = fIter

        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)

            for j in xrange(self.mCycle):
                
                _trn, _tst = self.cross_validation(j, fImgs)

                for k in xrange(len(_trn) / fBatch):

                    # print self.mEpsilon, self.mMomentum
                    
                    # Inputs and labels batch
                    _in  = self.build_batch(fBatch, k, _trn)

                    # Activation propagation
                    _out = self.dropout_propagation(_in)

                    # Local error for each layer
                    _err = self.compute_layer_error(_out, _in)
        
                    # Gradient for stochastic gradient descent    
                    _wGrad,_bGrad = self.gradient(fBatch, _err, _out)
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # self.gradient_checking(_in, _in, _wGrad,
                    #                        _bGrad, fBatch)

                    # Adapt learning rate
                    if i > 0 or j > 0 or k > 0:
                        self.angle_driven_approach(_wGrad)
                    
                    # Update weights and biases
                    self.update(_wGrad, _bGrad)
                    
                # Evaluate the network    
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

        if fName is not None:
            self.save_output(fName, "train", fImgs)
        
#####################################################################

    def evaluate(self, fTests):
        
        _out  = self.propagation(fTests.T)
            
        return self.error(_out[-1], fTests.T) / len(fTests)
        
#####################################################################

    # Test the neural network over a test set
    def test(self, fImgs, fLbls, fName):

        print "Testing the neural networks..."

        _out   = self.propagation(fImgs.T)
        _cost  = self.error(_out[-1], fImgs.T) / len(fImgs)

        print "Cost {0}\n".format(_cost)

        # Save output in order to have a testset for next layers
        if fName is not None:
            self.save_output(fName, "test", fImgs)

        # Displaying the results
        dy.display(fName, "out", [fImgs, _out[-1].T])

        # Approximated vision of first hidden layer neurons
        dy.display(fName, "neurons", [self.neurons_visions()], 5, 5)
