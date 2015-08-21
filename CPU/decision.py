from autoencoders import AUTOENCODERS

import numpy   as np
import time    as tm
import display as dy

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons, fBatchSize):

        # Mother class initialization
        AUTOENCODERS.__init__(self, fNeurons, fBatchSize)
        
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fImgs, fLbls, fIterations, fName):

        print "Training...\n"

        _gcost = []
        _gperf = []
        _gtime = []

        _done  = fIterations
        
        for i in xrange(fIterations):

            _gtime.append(tm.time())
            _gcost.append(0)
            _gperf.append(0)
            
            for j in xrange(self.mCycle):

                _trn, _tst = self.cross_validation(j, fImgs, fLbls)

                for k in xrange(len(_trn[0]) / self.mBatchSize):

                    # Input and labels batch
                    _in, _lbls = self.build_batch(k,_trn[0],_trn[1])

                    # Activation propagation
                    _out = self.dropout_propagation(_in)

                    # Local error for each layer
                    _err = self.compute_layer_error(_out, _in)
        
                    # Gradient for stochastic gradient descent    
                    _wGrad, _bGrad = self.gradient(_err, _out)

                    # Adapt learning rate
                    # if i > 0 or j > 0 or k > 0:
                    #     self.angle_driven_approach(_wGrad)

                    # Weight variation
                    self.weight_variation(_wGrad)
                    
                    # Update weights and biases
                    self.update(_bGrad)

                    # Adapt learning rate
                    self.average_gradient_approach(_wGrad)

                # Global cost and perf update in a cycle
                _cost, _perf  = self.evaluate(_tst[0], _tst[1])
                _gcost[i]    += _cost
                _gperf[i]    += _perf

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

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gperf, fName, "_perf.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")

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
