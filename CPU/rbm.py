from neural_network import NEURAL_NETWORK

import numpy   as np
import tools   as tl
import time    as tm
import display as dy

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

        _activation = _sigm
        # _activation = np.random.binomial(1, _sigm, _sigm.shape)

        return _activation
        
#####################################################################

    def down_propagation_probability(self, fInput):

        _presigm = self.mBiases[1] + np.dot(self.mWeights[1], fInput)

        return tl.sigmoid(_presigm)

#####################################################################

    def v_given_h(self, fInput):

        _sigm = self.down_propagation_probability(fInput)

        _activation = _sigm
        # _activation = np.random.binomial(1, _sigm, _sigm.shape)

        return _activation

#####################################################################

    # Gibbs sampling
    def propagation(self, fInput):
    
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
        
        # Weights update
        _grad  = (np.dot(fData[1], fData[0].T) - np.dot(fReco[1], fReco[0].T)) / fSize

        self.mWeights[0] += _eps * (_grad - _dec * self.mWeights[0])
        self.mWeights[1]  = self.mWeights[0].T

        # Biases update
        self.mBiases[0] +=  _eps * (fData[1] - fReco[1]).mean(1, keepdims=True)
        self.mBiases[1] +=  _eps * (fData[0] - fReco[0]).mean(1, keepdims=True)

        
#####################################################################

    def train(self, fSets, fIter, fSize, fName):

        print "Training..."
        
        _sets  = tl.binary(fSets[0])
        _gcost = []
        _done  = fIter

        n = len(_sets) / fSize

        for i in xrange(fIter):

            _benchmark = tm.time()
            _gcost.append(0)
            
            for j in xrange(self.mCycle):

                _train, _test = self.cross_validation(_sets)
                
                for k in xrange(n):

                    # Inputs and labels batch
                    _input = self.build_batch(_train, None, fSize)
                    
                    # Gibbs sampling over k-iterations
                    _data, _reco = self.propagation(_input)

                    # Update weights and biases
                    self.update(_data, _reco, fSize)

                _gcost[i] += self.evaluate(_test)

            # Iteration information
            _benchmark = tm.time() - _benchmark    
            print "Iteration {0} in {1}s".format(i, _benchmark)
                
            # Global cost for one cycle
            _gcost[i] = _gcost[i] / self.mCycle            
            print "Global cost of iteration : ", _gcost[i]

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i]) < 0.001 or
                       _gcost[i-1] - _gcost[i]  < 0):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")

        return _data[1]

#####################################################################

    def evaluate(self, fTests):

        _cost = 0
            
        for data in fTests:
            _in = data.reshape(len(data), 1)
            _data, _reco = self.propagation(_in)                
            _cost += self.propagation_cost(_reco[0], _data[0])
                
        return _cost / len(fTests)

#####################################################################

    def test(self, fSets, fName, fPsize):

        print "Testing the neural networks..."
        
        _cost  = 0
        _out   = []
            
        for data in tl.binary(fSets):
            _in = data.reshape(len(data), 1)
            _data, _reco = self.propagation(_in)                
            _cost += self.propagation_cost(_reco[0], _data[0])
            _out.append(_reco[0])
                
        _cost = _cost / len(fSets)
        print "Cost :", _cost

        # Displaying the results
        dy.display(fName, [fSets, _out], len(fSets), fPsize, "out")
    
        # Save output in order to have a testset for next layers
        self.save_output(fName, "test", _out)

        # Approximated vision of first hidden layer neurons
        _res = self.neurons_visions()
        dy.display(fName, [_res], self.mNeurons[1], fPsize,
                   "neurons", 5, 5)
