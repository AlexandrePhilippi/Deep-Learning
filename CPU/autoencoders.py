from neural_network import NEURAL_NETWORK

import numpy   as np
import tools   as tl
import time    as tm
import display as dy

class AUTOENCODERS(NEURAL_NETWORK):
    
    def __init__(self, fNeurons):

        # Mother class initialization
        NEURAL_NETWORK.__init__(self, len(fNeurons), fNeurons)

#####################################################################

    # Compute all the layer output for a given input (can be a batch)
    def propagation(self, fInput):

        i = 0
        _out = []
        _out.append(fInput)
        
        for w,b in zip(self.mWeights, self.mBiases):
            _out.append(tl.sigmoid(np.dot(w, _out[i]) + b))
            i = i + 1
            
        return _out

#####################################################################
    
    # Compute all the locals error in order to compute gradient
    def compute_layer_error(self, fOut, fIn, fSize):

        # Sparsity coefficients
        _beta = self.mBeta
        _rho  = self.mRho

        # Average activation
        _avg = self.average_activation(fOut, fSize)
        
        _err  = []

        # Last layer local error
        _err.append((fOut[-1] - fIn) * fOut[-1] * (1 - fOut[-1]))

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers - 1):
            _sparsity = -_rho / _avg[-i] + (1 - _rho)/(1 - _avg[-i])
            
            _err.append((np.dot(self.mWeights[-i].T, _err[i-1]) + _beta * _sparsity) * fOut[-i-1] * (1 - fOut[-i-1]))
            
        _err.reverse()

        return _err

#####################################################################

    # Compute the gradient according to W and b in order to
    # realize the mini-batch gradient descent    
    def cost_grad(self, fErr, fOut, fSize):

        _wGrad = []
        _bGrad = []

        for err, out in zip(fErr, fOut):
            _wGrad.append(np.dot(err, out.T) / fSize)
            _bGrad.append(err.mean(1, keepdims=True))
            
        return _wGrad, _bGrad

#####################################################################

    def weight_variation(self, fVar, fWGrad):

        return [-self.mEpsilon[i] * fWGrad[i] + self.mMomentum[i] * fVar[i] for i in xrange(self.mNbLayers-1)]

#####################################################################
    
    # Update weights and biases parameters with gradient descent
    def update(self, fVar, fBGrad):

        for i in xrange(self.mNbLayers-1):
            
            # Update weights
            self.mWeights[i] += fVar[i]

            # Update biases
            self.mBiases[i]  -= self.mEpsilon[i] * fBGrad[i]

#####################################################################
    
    # One step of the training algorithm
    def train_one_step(self, fIn, fRef, fSize):

        # Activation propagation
        _out  = self.propagation(fIn)

        # Local error for each layer
        _err = self.compute_layer_error(_out, fRef, fSize)
        
        # Gradient for stochastic gradient descent    
        _wGrad, _bGrad = self.cost_grad(_err, _out, fSize)

        return (_out, _wGrad, _bGrad)

#####################################################################
    
    # One step of the train algorithm to get output and cost
    def output_and_cost(self, fIn, fRef):

        # All the output generated according to the batch
        _out  = self.propagation(fIn)
        
        # Cost linked to the batch passed in argument
        _cost = self.propagation_cost(_out[-1], fRef)

        return _out, _cost

#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName):

        print "Training..."
        
        _sets  = fSets[0]
        _var   = [np.zeros(_w.shape) for _w in self.mWeights]
        _gcost = []
        _gtime = []
        
        _done  = fIter

        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)
            
            for j in xrange(self.mCycle):
                
                _train, _test = self.cross_validation(_sets)

                for k in xrange(len(_sets) / fSize):

                    # Only for gradient checking
                    # self.mBeta = 0
                    # self.mTeta = 0
                    
                    # Inputs and labels batch
                    _in = self.build_batch(_train, None, fSize)
                    
                    # One training step
                    _ret = self.train_one_step(_in, _in, fSize)
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # _grad = self.numerical_gradient(_in,_in,fSize)
                    
                    # self.gradient_checking(_grad[0], _grad[1],
                    #                        _ret[1] , _ret[2])

                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_var, _ret[1])

                    # Weight variation computation
                    _var = self.weight_variation(_var, _ret[1])
                        
                    # Update weights and biases
                    self.update(_var, _ret[2])

                _gcost[i] += self.evaluate(_test)

            # Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])    

            # Global cost for one cycle
            _gcost[i] /= self.mCycle
            print "Cost of iteration : {0}".format(_gcost[i])

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i]) < 0.001):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")        

        return self.propagation(_sets.T)[1]

#####################################################################

    def evaluate(self, fTests):

        _cost = 0

        for data in fTests:
            _in    = data.reshape(len(data), 1)
            _out   = self.propagation(_in)
            _cost += self.propagation_cost(_out[-1], _in)

        return _cost / len(fTests)
        
#####################################################################

    # Test the neural network over a test set
    def test(self, fSets, fName, fPsize):

        print "Testing the neural networks..."

        _cost = 0
        _out  = []

        for data in fSets:
            _in = data.reshape(len(data),1)
            _out.append(self.propagation(_in)[-1])
            _cost  += self.propagation_cost(_out[-1], _in)

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
