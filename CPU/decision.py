from autoencoders import AUTOENCODERS

import numpy   as np
import time    as tm
import display as dy

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons):

        # Mother class initialization
        AUTOENCODERS.__init__(self, fNeurons)
        
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fImgs, fLbls, fIter, fBatch, fName):

        print "Training...\n"

        _gcost = []
        _gperf = []
        _gtime = []

        _done  = fIter
        
        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)
            _gperf.append(0)
            
            for j in xrange(self.mCycle):

                _trn, _tst = self.cross_validation(j, fImgs, fLbls)

                for k in xrange(len(_trn[0]) / fBatch):

                    # Input and labels batch
                    _in, _lbls = self.build_batch(fBatch , k,
                                                  _trn[0], _trn[1])
                    
                    # One training step
                    _ret = self.train_one_step(fBatch, _in, _lbls)

                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_ret[1])
                        
                    # Update weights and biases for next iteration
                    self.update(_ret[1], _ret[2])

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

            # Learning rate update
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
