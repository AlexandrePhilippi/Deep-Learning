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

        _out = [fInput]

        for w,b in zip(self.mWeights, self.mBiases):
            _out.append(tl.sigmoid(np.dot(w, _out[-1]) + b))

        return _out

#####################################################################
    
    # Compute all the locals error in order to compute gradient
    def compute_layer_error(self, fOut, fIn, fSize):

        # Last layer local error
        _err = []
        _err.append(-(fIn - fOut[-1]) * fOut[-1] * (1 - fOut[-1]))

        # Intermediate layer local error
        for i in xrange(1, self.mNbLayers-1):
            _err.append((np.dot(self.mWeights[-i].T, _err[i-1])) * fOut[-i-1] * (1 - fOut[-i-1]))

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

    def weight_variation(self, fWvar, fWgrad):

        return [-self.mEpsilon * fWgrad[i] + self.mMomentum * fWvar[i] for i in xrange(self.mNbLayers-1)]

#####################################################################
    
    # Update weights and biases parameters with gradient descent
    def update(self, fWvar, fBgrad):

        for i in xrange(self.mNbLayers-1):
            
            # Update weights
            self.mWeights[i] += fWvar[i]

            # Update biases
            self.mBiases[i]  += self.mEpsilon * fBgrad[i]

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
    def train(self, fSets, fIter, fSize, fName, fCyc=6, fSlc=10000):

        print "Training...\n"

        _Wvar  = [np.zeros(_w.shape) for _w in self.mWeights]
        _sets  = fSets[0]
        _gcost = []
        _gtime = []
        
        _done  = fIter

        # Cross validation index
        _idx   = 0
        
        for i in xrange(fIter):

            _gtime.append(tm.time())
            _gcost.append(0)
            
            for j in xrange(fCyc):
                
                _idx, _trn, _tst = self.cross_validation(_sets,
                                                         None,
                                                         _idx,
                                                         fSlc,
                                                         fCyc)

                for k in xrange(len(_trn) / fSize):

                    # Inputs and labels batch
                    _in = self.build_batch(_trn, None, fSize)
                    
                    # One training step
                    _ret = self.train_one_step(_in, _in, fSize)
                    
                    # Gradient checking
                    # print "Gradient checking ..."
                    # _grad = self.numerical_gradient(_in,_in,fSize)

                    # self.gradient_checking(_grad[0], _grad[1],
                    #                        _ret[1] , _ret[2])

                    # Adapt learning rate
                    if(i > 0 or j > 0 or k > 0):
                        self.angle_driven_approach(_Wvar, _ret[1])

                    # Weight variation computation
                    _Wvar = self.weight_variation(_Wvar, _ret[1])
                        
                    # Update weights and biases
                    self.update(_Wvar, _ret[2])

                _gcost[i] += self.evaluate(_tst)

            # Iteration information
            _gtime[i] = tm.time() - _gtime[i]
            print "Iteration {0} in {1}s".format(i, _gtime[i])    

            # Global cost for one cycle
            _gcost[i] /= fCyc
            print "Cost of iteration : {0}".format(_gcost[i])

            # Parameters
            print "Epsilon {0} Momentum {1}\n".format(self.mEpsilon,
                                                      self.mMomentum)

            # Learning rate update
            if(i > 0):
                if(abs(_gcost[i-1] - _gcost[i]) < 0.001):
                    _done = i + 1
                    break

        self.plot(xrange(_done), _gcost, fName, "_cost.png")
        self.plot(xrange(_done), _gtime, fName, "_time.png")        

        return self.propagation(_sets.T)[1].T

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
    def test(self, fSets, fName):

        print "Testing the neural networks..."

        _cost = 0
        _out  = []

        for data in fSets:
            _in = data.reshape(len(data),1)
            _out.append(self.propagation(_in)[-1])
            _cost  += self.propagation_cost(_out[-1], _in)

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
