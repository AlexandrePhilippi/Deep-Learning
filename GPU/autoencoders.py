from neural_network import NEURAL_NETWORK

import numpy  as np
import gnumpy as gpu
import tools  as tl
import time   as tm

class AUTOENCODERS(NEURAL_NETWORK):
    
    def __init__(self, fNeurons):

        # Mother class initialization
        NEURAL_NETWORK.__init__(self, len(fNeurons), fNeurons)

#####################################################################

    # Compute all the layer output for a given input (can be a batch)
    def computation(self, fInput):

        i = 0
        _out = []
        _out.append(fInput)
        
        for w,b in zip(self.mWeights, self.mBiases):
            _out.append(gpu.logistic(gpu.dot(w, _out[i]) + b))
            i = i + 1
            
        return _out

#####################################################################
    
    # Compute all the locals error in order to compute gradient
    def compute_layer_error(self, fOut, fIn, fAvg):

        # Sparsity coefficients
        _beta = self.mBeta
        _rho  = self.mRho

        _err  = []

        # Last layer local error
        _err.append((fOut[-1] - fIn) * fOut[-1] * (1 - fOut[-1]))

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers - 1):
            _sparsity = -_rho / fAvg[-i] + (1 - _rho)/(1 - fAvg[-i])
            
            _err.append((gpu.dot(self.mWeights[-i].T, _err[i-1]) + _beta * _sparsity) * fOut[-i-1] * (1 - fOut[-i-1]))
            
        _err.reverse()

        return _err

#####################################################################

    # Compute the gradient according to W and b in order to
    # realize the mini-batch gradient descent    
    def cost_grad(self, fErr, fOut, fSize):

        _wGrad = []
        _bGrad = []

        for err, out in zip(fErr, fOut):
            _wGrad.append(gpu.dot(err, out.T) / fSize)
            _bGrad.append(err.mean(1))
            
        return _wGrad, _bGrad

#####################################################################
    
    # Update weights and biases parameters with gradient descent
    def update(self, fWGrad, fBGrad):

        for i in xrange(self.mNbLayers - 1):
            self.mWeights[i] -= self.mEpsilon * (self.mTeta * self.mWeights[i] + fWGrad[i])

            self.mBiases[i]  -= self.mEpsilon * fBGrad[i].reshape(len(fBGrad[i]), 1)

#####################################################################
        
    # One step of the train algorithm to get output and cost
    def output_and_cost(self, fIn, fRef):

        # All the output generated according to the batch
        _out  = self.computation(fIn)
        
        # Cost linked to the batch passed in argument
        _cost = self.computation_cost(_out[-1], fRef)

        return _out, _cost

#####################################################################
    
    # One step of the training algorithm
    def train_one_step(self, fIn, fRef, fSize):

        # Output for each layer and cost linked to the batch
        _out, _cost = self.output_and_cost(fIn, fRef) 

        # Average activation
        _avg = self.average_activation(_out, fSize) 
        
        # Local error for each layer
        _err = self.compute_layer_error(_out, fRef, _avg)
        
        # Gradient for stochastic gradient descent    
        _wGrad, _bGrad = self.cost_grad(_err, _out, fSize)

        return (_cost, _avg, _wGrad, _bGrad)

#####################################################################
    
    # Algorithm which train the neural network to reduce cost
    def train(self, fSets, fIter, fSize, fName):

        _sets  = fSets[0] 
        _gcost = []
        _done  = fIter + 1

        # Batch-subiteration index 
        n = len(_sets) / fSize
        
        print "Training..."
        for i in xrange(fIter):

            _benchmark = tm.clock()
            _gcost.append(0)
            
            for j in xrange(self.mCycle):
                
                _train, _test = self.cross_validation(_sets)
                _avg          = self.init_average_list()

                _lcost = 0
                
                for k in xrange(n):
                    
                    # Only for gradient checking
                    # self.mBeta = 0
                    # self.mTeta = 0
                    
                    # Inputs and labels batch
                    _input = self.build_batch(_train, fSize)
                    
                    # One training step
                    _ret = self.train_one_step(_input, _input, fSize)
                    
                    # Cost over batch
                    _lcost += _ret[0] 
                    
                    _avg = [x+y/n for x,y in zip(_avg, _ret[1])]
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # _grad = self.numerical_gradient(_input,
                    #                                 _input,
                    #                                  fSize)
                    
                    # self.gradient_checking(_grad[0], _grad[1],
                    #                        _ret[2] , _ret[3])

                    # Update weights and biases
                    self.update(_ret[2], _ret[3])

                # Global cost update in a cycle
                _lcost /= len(fSets[0])       
                _gcost[i] += self.global_cost(_lcost, _avg)

            # Iteration information
            _benchmark = tm.clock() - _benchmark
            print "Iteration {0} in {1}s".format(i, _benchmark)    

            # Global cost for one cycle
            _gcost[i] /= self.mCycle
            print "Global cost of iteration :", _gcost[i]

            # Learning rate update
            if(i > 0):
                if(_gcost[i-1] - _gcost[i] < 0):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName + "_cost.png")
                
#####################################################################

    # Algorithm which test the neural network over a test sets
    def test(self, fSets):

        print "Testing the neural networks..."

        _cost = 0
        _out  = []

        for data in fSets:
            _input = data.reshape(len(data),1)
            _out.append(self.computation(_input)[-1])
            _cost  += self.computation_cost(_out[-1], _input)

        _cost = _cost / len(fSets)
        print "Cost :", _cost

        return _out

#####################################################################
# FOR DEEP NEURAL NETWORKS
#####################################################################
    
    # Create a new datasets for next layers
    def create_datasets(self, fSets):

        _output = []

        for data in fSets:

            _input = data.reshape(len(data),1)
            _out   = self.computation(_input)

            _output.append(_out[1])
            
        return _output
    
#####################################################################
# FOLLOWING METHODS USED FOR VERIFICATIONS
#####################################################################
            
    # Compute numerical gradient value in order to check results
    def numerical_gradient(self, fInput, fRef, fSize):

        _numWgrad = []
        _numBgrad = []

        # Numerical gradient according to W
        print "\t Numerical gradient according to Weights."
        for i in np.arange(len(self.mWeights)):

            print "\t \t -> Layer", i + 1
            _m = np.zeros(self.mWeights[i].shape)
            
            for j in np.arange(len(self.mWeights[i])):
                for k in np.arange(len(self.mWeights[i][j])):
                    self.mWeights[i][j,k] += self.mEpsilon
                    _left = self.output_and_cost(fInput, fRef)

                    self.mWeights[i][j,k] -= 2. * self.mEpsilon
                    _right = self.output_and_cost(fInput, fRef)

                    _res = (_left[1] - _right[1])/(2.*self.mEpsilon)
                    _m[j][k] = _res / fSize
                    
                    self.mWeights[i][j,k] += self.mEpsilon

            _numWgrad.append(_m)

        # Numerical gradient according to b
        print "\t Numerical gradient according to Biases."    
        for i in np.arange(len(self.mBiases)):

            print "\t \t -> Layer", i + 1
            _v = np.zeros(self.mBiases[i].shape)
            
            for j in np.arange(len(self.mBiases[i])):
            
                self.mBiases[i][j] += self.mEpsilon
                _left = self.output_and_cost(fInput, fRef)

                self.mBiases[i][j] -= 2. * self.mEpsilon
                _right = self.output_and_cost(fInput, fRef)

                _res  = (_left[1] - _right[1]) / (2. * self.mEpsilon)
                _v[j] = _res / fSize
                
                self.mBiases[i][j] += self.mEpsilon

            _numBgrad.append(_v)
                      
        return _numWgrad, _numBgrad

#####################################################################
    
    # Check gradient results
    def gradient_checking(self, _nWgrad, _nBgrad, _wGrad, _bGrad):

        _wError = np.zeros(len(_nWgrad))
        _bError = np.zeros(len(_nBgrad))
        
        for i in np.arange(len(_nWgrad)):
            _wError[i]  = np.linalg.norm(_nWgrad[i] - _wGrad[i]) / np.linalg.norm(_nWgrad[i] + _wGrad[i])

        for i in np.arange(len(_nBgrad)):
            _bError[i]  = np.linalg.norm(_nBgrad[i] - _bGrad[i]) / np.linalg.norm(_nBgrad[i] + _bGrad[i])

        print _wError
        print _bError
