from autoencoders import AUTOENCODERS

import numpy   as np
import time    as tm
import display as dy

# Global parameters
DROPOUT       = False
SPARSITY      = True
PREPROCESSING = False
ANGLE_DRIVEN  = True
AVG_GRADIENT  = False
GRAD_CHECK    = False
DEBUG         = False

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons, fBatchSize):

        # Mother class initialization
        AUTOENCODERS.__init__(self,
                              fNeurons,
                              fBatchSize)
                
#####################################################################

    def propagation(self, fInput, fDropout=False):
        '''Propagation of the input (can be a minibatch) throughout
        the neural network. A dropout system avoid the overfitting.
        The dropout scaling parameters can be modified in neural 
        network parameters.

        INPUT : Vector or matrix
        OUPUT : Neurons activation'''
        
        _out = [fInput]

        if fDropout:
            _p = self.mDropoutScaling

        for _w, _b in zip(self.mWeights, self.mBiases):

            _activation = self.sigmoid(np.dot(_w, _out[-1]) + _b)

            if fDropout:
                _drop  = (np.random.rand(*_activation.shape) < _p)
                _drop /= _p
                _activation *= _drop

            _out.append(_activation)

        return _out
    
#####################################################################

    def layer_error(self, fOut, fIn, fSparsity=False):
        '''Compute local error of each layer with sparsity
        term in order to get weights and biases gradients. 
        Part of backpropagation.

        INPUT  : Output of each layer, mini-batch
        OUTPUT : Error vector of each layer'''

        # Last layer local error
        try:
            _err = [-(fIn - fOut[-1]) * self.dsigmoid(fOut[-1])]
            
        except Warning:
            print sys.exc_info()[1]
            np.savetxt("log/error_dsig.log", self.dsigmoid(fOut[-1]))
            np.savetxt("log/error_diff.log", (fIn - fOut[-1]))
            sys.exit(-1)

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers-1):

            try:
                _backprop = np.dot(self.mWeights[-i].T, _err[i-1])
                
                _dsigmoid = self.dsigmoid(fOut[-i-1])

                if fSparsity:
                    _sparsity = self.sparsity(fOut[-i-1])

                else:
                    _sparsity = np.zeros(_backprop.shape)

                _err.append((_backprop + _sparsity) * _dsigmoid)

            except Warning:
                print sys.exc_info()[1]
                np.savetxt("log/error_backprop.log", _backprop)
                np.savetxt("log/error_dsigm.log", _dsigmoid)
                np.savetxt("log/error_sparsity.log", _sparsity)

        _err.reverse()

        return _err

#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fImgs, fLbls, fIterations, fName):
        '''Training algorithm. Can evolved according to your need.

        INPUT  : Images set, labels set (None for autoencoders),
                 number of iterations before stopping, name for save
        OUTPUT : Nothing'''

        if PREPROCESSING:
            fImgs, _key = ld.normalization(fName, fImgs)

        print "Training...\n"
        
        _gcost = []
        _gtime = []
        _gperf = []
        
        _done  = fIterations
        
        for i in xrange(fIterations):

            _gtime.append(tm.time())
            _gcost.append(0)
            _gperf.append(0)
            
            for j in xrange(self.mCycle):

                _trn, _tst = self.cross_validation(j, fImgs, fLbls)

                for k in xrange(len(_trn[0]) / self.mBatchSize):

                    if DEBUG:
                        print "Learning rates :", self.mEpsilon
                        print "Momentums :", self.mMomentum
                    
                    # Input and labels batch
                    _in, _lbls = self.build_batch(k,_trn[0],_trn[1])

                    # Activation propagation
                    _out = self.propagation(_in, DROPOUT)

                    # Local error for each layer
                    _err = self.layer_error(_out, _in, SPARSITY)

                    # Gradient checking
                    if GRAD_CHECK:
                        print "Gradient checking ..."
                        self.gradient_checking(_in,_in,_wGrad,_bGrad)
                    
                    # Adapt learning rate
                    if (i > 0 or j > 0 or k > 0) and ANGLE_DRIVEN:
                        self.angle_driven_approach(_wGrad)

                    # Weight variations
                    self.variations(_wGrad)
                    
                    # Update weights and biases
                    self.update(_bGrad)

                    # Adapt learning rate
                    if AVG_GRADIENT:
                        self.average_gradient_approach(_wGrad)
                        
                # Global cost and perf update in a cycle
                _cost, _perf  = self.evaluate(_tst[0], _tst[1])
                _gcost[i]    += _cost
                _gperf[i]    += _perf

                if DEBUG:
                    print "Cost :", _cost

            #Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])

            # Global cost for one cycle
            _gcost[i] /= self.mCycle
            print "Cost of iteration : {0}".format(_gcost[i])

            # Global perf for one cycle
            _gperf[i] /= self.mCycle
            print "Current performances : {0}".format(_gperf[i])

            # Parameters
            print "Epsilon {0} Momentum {1}\n".format(self.mEpsilon,
                                                      self.mMomentum)

            # Stop condition
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i])  < 0.001):
                    _done = i + 1
                    break

        dy.plot(xrange(_done), _gcost, fName, "_cost.png")
        dy.plot(xrange(_done), _gperf, fName, "_perf.png")
        dy.plot(xrange(_done), _gtime, fName, "_time.png")

#####################################################################

    def evaluate(self, fImgs, fLbls):

        _out  = []
        _cost = 0
        
        for i in xrange(len(fImgs)):
            _in   = fImgs[[i],:].T

            _lbls = np.zeros((self.mNeurons[-1], 1))
            _lbls[fLbls[i]] = 1
            
            _out.append(self.propagation(_in)[-1])
            _cost     += self.error(_out[-1], _lbls)
            
        # Benchmarking    
        _valid = 0.            
        for i in xrange(len(_out)):
            _valid += (fLbls[i] == np.argmax(_out[i]))

        _perf = (_valid * 100.) / len(fImgs)
        _cost = _cost / len(fImgs)    
            
        return _cost, _perf 

#####################################################################

    def test(self, fImgs, fLbls, fName):

        print "Testing the network ..."

        _cost, _perf = self.evaluate(fImgs, fLbls)
        
        print "Cost : {0}, Performances : {1}".format(_cost, _perf)
