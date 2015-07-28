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
    
        _data = []
        _reco = []
        
        _data.append(fInput)        
        _hid = self.h_given_v(fInput)
        _data.append(_hid)

        for i in xrange(self.cdk):
            _vis = self.v_given_h(_hid)
            _hid = self.h_given_v(_vis)

        _reco.append(_vis)
        _reco.append(_hid)

        return _data, _reco

#####################################################################
    
    def compute_grad(self, fData, fReco, fSize):

        _wGrad  = np.dot(fData[1], fData[0].T) / fSize
        _wGrad -= np.dot(fReco[1], fReco[0].T) / fSize

        _bGrad = [(fData[1] - fReco[1]).mean(1, keepdims=True),
                  (fData[0] - fReco[0]).mean(1, keepdims=True)]

        return _wGrad, _bGrad

#####################################################################

    def weight_variation(self, fVar, fWGrad):

        return -self.mEpsilon[0] * fWGrad + self.mMomentum[0] * fVar
    
#####################################################################

    def update(self, fVar, fBGrad):

        # Weights update
        self.mWeights[0] += fVar
        self.mWeights[1]  = self.mWeights[0].T

        # Biases update
        self.mBiases[0]  -=  self.mEpsilon[0] * fBGrad[0] 
        self.mBiases[1]  -=  self.mEpsilon[1] * fBGrad[1] 

        
#####################################################################

    def train(self, fSets, fIter, fSize, fName):

        print "Training..."
        
        _sets  = tl.binary(fSets[0])
        _var   = np.zeros(self.mWeights[0].shape)
        _gcost = []
        _gtime = []

        _done  = fIter

        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)

            for j in xrange(self.mCycle):

                _train, _test = self.cross_validation(_sets)
                
                for k in xrange(len(_sets) / fSize):

                    # Inputs and labels batch
                    _input = self.build_batch(_train, None, fSize)
                    
                    # Gibbs sampling over k-iterations
                    _data, _reco = self.propagation(_input)

                    # Gradient
                    _grad = self.compute_grad(_data, _reco, fSize)

                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_var, _grad[0])
                    
                    # Weight variation
                    _var = self.weight_variation(_var, _grad[0])
                    
                    # Update weights and biases
                    self.update(_var, _grad[1])

                _gcost[i] += self.evaluate(_test)

            # Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])
                
            # Global cost for one cycle
            _gcost[i] = _gcost[i] / self.mCycle            
            print "Global cost of iteration : ", _gcost[i]

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i]) < 0.001):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")        

        return self.propagation(_sets.T)[0][1].T

#####################################################################

    def evaluate(self, fTests):

        _cost = 0
            
        for data in fTests:
            _in = data.reshape(len(data), 1)
            _data, _reco = self.propagation(_in)                
            _cost += self.propagation_cost(_reco[0], _data[0])
                
        return _cost / len(fTests)

#####################################################################

    def test(self, fSets, fName):

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

        # Save output in order to have a testset for next layers
        self.save_output(fName, "test", _out)
        
        # Check if it's possible to print the image
        _psize = [np.sqrt(self.mNeurons[0]) for i in xrange(2)]

        if self.mNeurons[0] != (_psize[0] * _psize[1]):
            return
        
        # Displaying the results
        dy.display(fName, [fSets, _out], len(fSets), _psize, "out")

        # Approximated vision of first hidden layer neurons
        _res = self.neurons_visions()
        dy.display(fName, [_res], self.mNeurons[1], _psize,
                   "neurons", 5, 5)
