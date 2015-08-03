from autoencoders import AUTOENCODERS

import numpy  as np
import time   as tm

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons):

        # Mother class initialization
        AUTOENCODERS.__init__(self, fNeurons)
        
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName, fCyc=6, fSlc=10000):

        print "Training...\n"

        _wVar  = [np.zeros(_w.shape) for _w in self.mWeights]
        _gcost = []
        _gperf = []
        _gtime = []

        _done  = fIter
        
        # Cross validation index
        _idx   = 0
        
        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)
            _gperf.append(0)
            
            for j in xrange(fCyc):

                _idx, _trn, _tst = self.cross_validation(fSets[0],
                                                         fSets[1],
                                                         _idx,
                                                         fSlc,
                                                         fCyc)
                                                         
                for k in xrange(len(_trn[0]) / fSize):

                    # Input and labels batch
                    _input, _lbls = self.build_batch(_trn[0],
                                                     _trn[1], fSize)
                    
                    # One training step
                    _ret = self.train_one_step(_input, _lbls, fSize)

                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_wVar, _ret[1])

                    # Weight variation computation
                    _wVar= self.weight_variation(_wVar, _ret[1])
                        
                    # Update weights and biases for next iteration
                    self.update(_wVar, _ret[2])

                # Global cost and perf update in a cycle
                _cost, _perf  = self.evaluate(_tst)
                _gcost[i]    += _cost
                _gperf[i]    += _perf

            #Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])

            # Global cost for one cycle
            _gcost[i] /= fCyc
            print "Cost of iteration : {0}".format(_gcost[i])

            # Global perf for one cycle
            _gperf[i] /= fCyc
            print "Current performances : {0}".format(_gperf[i])

            # Parameters
            print "Epsilon {0} Momentum {1}\n".format(self.mEpsilon,
                                                      self.mMomentum)

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i])  < 0.001 or
                       _gperf[i]   - _gperf[i-1] < 0):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gperf, fName, "_perf.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")

#####################################################################

    def evaluate(self, fTests):

        _out  = []
        _cost = 0
        
        for i in xrange(len(fTests[0])):
            _in, _lbls = self.build_batch(fTests[0],fTests[1],1,[i])
            _out.append(self.propagation(_in)[-1])
            _cost     += self.propagation_cost(_out[-1], _lbls)
            
        # Benchmarking    
        _valid = 0.            
        for i in xrange(len(_out)):
            _valid += (fTests[1][i] == np.argmax(_out[i]))

        _perf = (_valid * 100.) / len(fTests[0])
        _cost = _cost / len(fTests[0])    
            
        return _cost, _perf 

#####################################################################

    def test(self, fSets):

        print "Testing the network ..."

        _cost, _perf = self.evaluate(fSets)
        
        print "Cost : {0}, Performances : {1}".format(_cost, _perf)
