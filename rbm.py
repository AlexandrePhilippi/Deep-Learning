from neural_network import NEURAL_NETWORK

import numpy as np
import tools as tl
import time  as tm

import matplotlib.pyplot as plt

class RBM(NEURAL_NETWORK):

    def __init__(self, fNeurons):

        # Mother class initialization
        NEURAL_NETWORK.__init__(self,
                                len(fNeurons),
                                fNeurons)

        # Contrastive divergence number of iteration
        self.cdk = 1

        # RBM constraint
        self.mWeights[1] = np.transpose(self.mWeights[0])


#####################################################################
        
    def up_propagation_probability(self, fInput):

        _presigm = self.mBiases[0] + np.dot(self.mWeights[0], fInput)
        
        return tl.sigmoid(_presigm)

#####################################################################
    
    def h_given_v(self, fInput):

        _sigm = self.up_propagation_probability(fInput)

        _activation = np.random.binomial(1, _sigm, _sigm.shape)

        return _activation
        
#####################################################################

    def down_propagation_probability(self, fInput):

        _presigm = self.mBiases[1] + np.dot(self.mWeights[1], fInput)

        return tl.sigmoid(_presigm)

#####################################################################

    def v_given_h(self, fInput):

        _sigm = self.down_propagation_probability(fInput)

        _activation = np.random.binomial(1, _sigm, _sigm.shape)

        return _activation

#####################################################################
    
    def gibbs_sampling(self, fInput):

        _data = []; _reconstruction = []
        
        _data.append(fInput)
        _hid = self.h_given_v(fInput)
        _data.append(_hid)

        for i in xrange(self.cdk):
            _vis = self.v_given_h(_hid)
            _hid = self.h_given_v(_vis)

        _reconstruction.append(_vis)
        _reconstruction.append(_hid)

        return _data, _reconstruction
    
#####################################################################

    def update(self, fData, fReco, fSize):

        _eps = self.mEpsilon
        _dec = self.mTeta
        _mom = self.mMomentum
        
        # Weights update
        _grad  = (np.dot(fData[1], fData[0].T) - np.dot(fReco[1], fReco[0].T)) / fSize

        self.mDWgrad[0] = _eps * (_grad - _dec * self.mWeights[0]) + _mom * self.mDWgrad[0]
        
        self.mWeights[0] += self.mDWgrad[0]
        self.mWeights[1]  = self.mWeights[0].T

        # Biases update
        self.mBiases[0] +=  _eps * (fData[1] - fReco[1]).mean(1, keepdims=True)
        self.mBiases[1] +=  _eps * (fData[0] - fReco[0]).mean(1, keepdims=True)

        
#####################################################################

    def train(self, fSets, fIter, fSize, fName):

        _sets  = tl.binary(fSets[0])
        _gcost = []
        _done  = fIter

        _train = _sets
        self.mCycle = 1

        n = len(_sets) / fSize

        print "Training..."
        for i in xrange(fIter):

            _benchmark = tm.clock()
            _gcost.append(0)
            
            for j in xrange(self.mCycle):

                # _train, _test = self.cross_validation(_sets)
                
                _lcost = 0
                
                for k in xrange(n):

                    # _clock = tm.clock()
                    # Inputs and labels batch
                    _input = self.build_batch(_train, fSize)
                    # print "Inputs built in", tm.clock() - _clock

                    # _clock = tm.clock()
                    # Gibbs sampling over k-iterations
                    _data, _reco = self.gibbs_sampling(_input)
                    # print "Samples built in", tm.clock() - _clock
                    
                    # Cost over batch
                    _lcost += self.computation_cost(_reco[0],
                                                    _data[0])

                    # _clock = tm.clock()
                    # Update weights and biases
                    self.update(_data, _reco, fSize)
                    # print "Update done in", tm.clock() - _clock

                # Global cost update in a cycle
                _gcost[i] += self.global_cost(_lcost / len(_sets))

            # Iteration information
            _benchmark = tm.clock() - _benchmark    
            print "Iteration {0} in {1}s".format(i, _benchmark)
                
            # Global cost for one cycle
            _gcost[i] = _gcost[i] / self.mCycle            
            print "Global cost of iteration : ", _gcost[i]

            # Learning rate update
            if(i > 0):
                self.decrease_learning_rate(_gcost[i-1], _gcost[i])

                if(abs(_gcost[i] - _gcost[i-1]) < 0.00001):
                    _done = i
                    break

        self.plot(xrange(_done), _gcost, "img/"+ fName +"_cost.png")

#####################################################################

    def test(self, fSets):

        print "Testing the neural networks..."
        
        _gcost = 0
        _out   = []
        fSets  = tl.binary(fSets)
            
        for i in xrange(len(fSets)):
            
            _input = self.build_batch(fSets, 1, [1])

            _data, _reco = self.gibbs_sampling(_input)
                
            _gcost += self.computation_cost(_reco[0], _data[0])

            _out.append(_reco[0])
                
        _gcost = _gcost / len(fSets)
        print "Global cost of iteration : ", _gcost

        return _out

#####################################################################
# FOR DEEP NEURAL NETWORKS
#####################################################################
    
    # Create a new datasets for next layers
    def create_datasets(self, fSets):

        fSets = tl.binary(fSets)
        _out  = []

        for data in fSets:

            _input = data.reshape(len(data),1)
            _out.append(self.h_given_v(_input))
            
        return _out
