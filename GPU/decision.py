from autoencoders import AUTOENCODERS

import numpy  as np
import tools  as tl
import loader as ld
import time   as tm

class DECISION(AUTOENCODERS):

    def __init__(self, fNeurons):

        # Mother class initialization
        AUTOENCODERS.__init__(self, fNeurons)

#####################################################################

    # Cross validation set
    def cross_validation(self, fSets):

        _slice = self.mSlice
        k      = self.mIndex % self.mCycle

        _trainsets = fSets[0][0:k*_slice, :]
        _trainlbls = fSets[1][0:k*_slice]

        if k == 0:
            _trainsets = fSets[0][(k+1)*_slice:len(fSets[0]), :]
            _trainlbls = fSets[1][(k+1)*_slice:len(fSets[1])]
            
        else:
            np.vstack((_trainsets,
                       fSets[0][(k+1)*_slice:len(fSets[0]), :]))
            np.hstack((_trainlbls,
                       fSets[1][(k+1)*_slice:len(fSets[1])]))
    
        _testsets = fSets[0][k*_slice:(k+1)*_slice, :]
        _testlbls = fSets[1][k*_slice:(k+1)*_slice]

        self.mIndex = k + 1
        
        return (_trainsets, _trainlbls), (_testsets, _testlbls)
        
#####################################################################
        
    def build_batch(self, fSets, fSize, fIndex=None):

        if fIndex is None:
            fIndex = np.random.randint(len(fSets[0]), size=fSize)

        # Inputs
        _input = fSets[0][fIndex,:].T

        # Labels
        _lbls = np.zeros((self.mNeurons[-1], fSize))
        
        for l in xrange(fSize):
            _lbls[fSets[1][fIndex[l]]][l] = 1
                
        return _input, _lbls
        
#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName):

        _gcost = []
        _perf  = []
        _done  = fIter + 1
        
        # Batch-subiteration index 
        n = len(fSets[0]) / fSize
        
        for i in xrange(fIter):

            _benchmark = tm.clock()

            _gcost.append(0)
            _perf.append(0)
            
            for j in xrange(self.mCycle):

                _train, _test = self.cross_validation(fSets)
                _avg          = self.init_average_list()

                _lcost = 0
                
                for k in xrange(n):

                    # Input and labels batch
                    _input, _lbls = self.build_batch(fSets, fSize)
                
                    # One training step
                    _ret = self.train_one_step(_input, _lbls, fSize)

                    # Cost
                    _lcost += _ret[0] / len(fSets[0]) 
                    _avg = [x+y/n for x,y in zip(_avg, _ret[1])]

                    # Update weights and biases for next iteration
                    self.update(_ret[2], _ret[3])

                # Global cost update in a cycle
                _gcost[i] += self.global_cost(_lcost, _avg)
                    
                # Global perf update in a cycle
                _perf[i]  += self.evaluate(_test)

            #Iteration information
            _benchmark = tm.clock() - _benchmark
            print "Iteration {0} in {1}s".format(i, _benchmark)
                
            # Global cost for one cycle
            _gcost[i] /= self.mCycle  
            print "Global cost of iteration :", _gcost[i]
            
            # Global perf for one cycle
            _perf[i] /= self.mCycle
            print "Current performances :", _perf[i]
            
            # Learning rate update
            if(i > 0):
                if(_gcost[i-1] - _gcost[i]  < 0 or
                   _perf[i]    - _perf[i-1] < 0):
                    _done = i + 1
                    break

        self.plot(np.arange(_done), _gcost, "img/"+fName+"_cost.png")
        self.plot(np.arange(_done), _perf,  "img/"+fName+"_perf.png")
                    
#####################################################################

    # Algorithm which test the neural network over a test sets
    def evaluate(self, fSets):

        _out = []
        
        for i in xrange(len(fSets[0])):

            # Input and labels references initialization
            _input, _lbls = self.build_batch(fSets, 1, [i])

            # Output computation
            _out.append(self.computation(_input)[-1])

        # Benchmarking    
        _valid = 0.
        for i in xrange(len(_out)):
            _valid += (fSets[1][i] == np.argmax(_out[i]))
                
        return (_valid * 100.) / len(_out)
