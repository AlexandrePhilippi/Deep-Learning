from autoencoders import AUTOENCODERS

import numpy  as np
import time   as tm

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons):

        # Mother class initialization
        AUTOENCODERS.__init__(self, fNeurons)
        
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName):

        _gcost = []
        _gperf  = []
        _done  = fIter
        
        # Batch-subiteration index 
        n = len(fSets[0]) / fSize
        
        for i in xrange(fIter):

            _benchmark = tm.clock()

            _gcost.append(0)
            _gperf.append(0)
            
            for j in xrange(self.mCycle):

                _train, _test = self.cross_validation(fSets[0],
                                                      fSets[1])
                
                for k in xrange(n):

                    # Input and labels batch
                    _input, _lbls = self.build_batch(fSets[0],
                                                     fSets[1], fSize)
                
                    # One training step
                    _ret = self.train_one_step(_input, _lbls, fSize)

                    # Update weights and biases for next iteration
                    self.update(_ret[2], _ret[3])

                # Global cost and perf update in a cycle
                _cost, _perf  = self.evaluate(_test)
                _gcost[i]    += _cost
                _gperf[i]    += _perf

            #Iteration information
            _benchmark = tm.clock() - _benchmark
            print "Iteration {0} in {1}s".format(i, _benchmark)
                
            # Global cost for one cycle
            _gcost[i] /= self.mCycle  
            print "Global cost of iteration :", _gcost[i]
            
            # Global perf for one cycle
            _gperf[i] /= self.mCycle
            print "Current performances :", _gperf[i]
            
            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i])  < 0.001 or
                       _gcost[i-1] - _gcost[i]   < 0     or
                       _gperf[i]   - _gperf[i-1] < 0):
                    _done = i + 1
                    break

        self.plot(np.arange(_done), _gcost, fName, "_cost.png")
        self.plot(np.arange(_done), _gperf, fName, "_perf.png")
                    
#####################################################################

    def evaluate(self, fTests):

        _out  = []
        _cost = 0
        
        for i in xrange(len(fTests[0])):
            _in, _lbls = self.build_batch(fTests, 1, [i])
            _out.append(self.propagation(_in)[-1])
            _cost     += self.propagation_cost(_out[-1], _lbls)

        # Benchmarking    
        _valid = 0.
        for i in xrange(len(_out)):
            _valid += (fTests[1][i] == np.argmax(_out[i]))

        _perf = (_valid * 100.) / len(_out)
        _cost = _cost / len(fTests)    
            
        return _cost, _perf 

#####################################################################

    def test(self, fSet):

        print "Testing the network ..."
        print "Performances :", self.evaluate(fSet)
