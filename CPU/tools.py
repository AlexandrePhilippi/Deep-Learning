import sys
import warnings          as wn
import scipy.special     as ss

# Global parameters
SIG_EXP      = True
SIG_TANH     = False

NO_FLAT_SPOT = 0.01
SIGOUT_SLOPE = 0.001

#####################################################################
# LOGISTIC FUNCTION
#####################################################################

def sigmoid_exp(fX, fCoef, fFlat=NO_FLAT_SPOT):
    '''Compute the sigmoid function : 1 / (1 + e^(-i*x)) + ax
    With a small linear coefficient to avoid flat spot.
    
    INPUT  : A single value, vector, matrix
    A coefficient for the slope
    A small coefficient to avoid flat spot
    OUTPUT : Sigmoid-value of the given input'''
    
    try:
        _sigmoid = ss.expit(fX * fCoef) + fFlat * fX

    except Warning:
        print sys.exc_info()[1]
        np.savetxt("log/error_sigexp.log", fX)
        sys.exit(-1)
        
    return _sigmoid

#####################################################################

def dsigmoid_exp(fX, fCoef, fFlat=NO_FLAT_SPOT):
    '''Compute the derived sigmoid function following the 
    derived formula : f'(x) = i * f(x) (1 - f(x)) + a.
    
    INPUT  : Sigmoid output
    Coefficient for the slope
    Small coefficient to avoid flat spot
    OUTPUT : The derived value''' 
    
    try:
        _dsigmoid = fFlat + fCoef * fX * (1 - fX)
        
    except Warning:
        print sys.exc_info()[1]
        np.savetxt("log/error_dsigexp.log", fX)
        sys.exit(-1)
        
    return _dsigmoid
    
#####################################################################

def sigmoid_tanh(fX, fA=NO_FLAT_SPOT):
    '''Compute the sigmoid function : 1.7159*tanh(2/3 * x) + ax
    With a small linear coefficient to avoid flat spot.
    
    INPUT  : A single value, vector, matrix and small coefficient
    OUTPUT : Sigmoid-value of the given input'''
    
    try:
        _sigmoid = 1.17159 * np.tanh((2. / 3.) * fX) + fA * fX
        
    except Warning:
        print sys.exc_info()[1]
        np.savetxt("log/error_sigtanh.log", fX)
        sys.exit(-1)
        
    return _sigmoid 
    
#####################################################################

def dsigmoid_tanh(fX, fA=NO_FLAT_SPOT):
    '''Compute the derived sigmoid function following the 
    derived formula : f'(x) = a + (1 - f(x)^2)
    
    INPUT  : Sigmoid output, small coefficient
    OUTPUT : The derived value'''
    
    try:
        _dsigmoid = fA + 1.17159 * (2. / 3.) * (1 - fX**2)
        
    except Warning:
        print sys.exc_info()[1]
        np.savetxt("log/error_dsigtanh.log", fX)
        sys.exit(-1)
        
    return _dsigmoid

#####################################################################

def sigmoid(fX):
    '''Compute a sigmoid function with a small linear 
    from developers implementation (see neural_network.py)
    coefficient to avoid flat spot. Sigmoid function depend
    
    INPUT  : A single value, vector, matrix
    OUTPUT : Sigmoid-value of the given input'''
    
    if SIG_EXP:
        return sigmoid_exp(fX, 1.)

    elif SIG_TANH:
        return sigmoid_tanh(fX)
    
    else:
        print "No activation function selected.\n"
        sys.exit(-1)
        
#####################################################################
        
def dsigmoid(fX):
    '''Compute a derived sigmoid function. The derived formula
    depend from developers implementation (see neural_network.py)
    
    INPUT  : Sigmoid output, small coefficient
    OUTPUT : The derived value''' 
    
    if SIG_EXP:
        return dsigmoid_exp(fX, 1.)
    
    elif SIG_TANH:
        return dsigmoid_tanh(fX)
    
    else:
        print "No activation function selected.\n"
        sys.exit(-1)
        
#####################################################################
            
def sigmoid_output(fX):

    return sigmoid_exp(fX, SIGOUT_SLOPE)

#####################################################################

def dsigmoid_output(fX):

    return dsigmoid_exp(fX, SIGOUT_SLOPE)
