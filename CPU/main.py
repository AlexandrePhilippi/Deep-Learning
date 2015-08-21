import autoencoders as ac
import decision     as dc
import loader       as ld
import display      as dy
import numpy        as np

import sys, getopt
    
def main(argv):

    np.set_printoptions(precision=3, threshold='nan')
    
    # Parameters initialization
    _neurons     = [784, 25, 784]
    _iterations  = 2000
    _batchSize   = 50

    _data    = "mnist"
    _save    = None
    _load    = None


    # Some can be passed in argument
    try:
        opts, args = getopt.getopt(argv, "hi:b:s:l:d:", ["list="])

    except getopt.GetoptError:
        print 'main.py -i <iterations> -b <batch> -s <save> -l <load> -d <data> --list <neurons>'
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -i <iterations> -b <batch> -s <save> -l <load> -d <data> --list <neurons>'
            sys.exit()

        elif opt == '-i':
            _iterations = int(arg)
            
        elif opt == '-b':
            _batchSize = int(arg)

        elif opt == '-s':
            _save = arg

        elif opt == '-l':
            _load = arg

        elif opt == '-d':
            _data = arg
            
        elif opt == '--list':
            _neurons = map(int, arg.split(','))

    # Creating the neural network
    if _neurons[0] == _neurons[-1]:
        _nnet = ac.AUTOENCODERS(_neurons, _batchSize)
    else:
        _nnet = dc.DECISION(_neurons, _batchSize)

    # Loading pretrained state if _load is given
    if _load is not None:
        _nnet.load_state(_load)

    # Loading the datasets for training
    _train = ld.load_datasets(_data, "train")

    # Training the network
    _nnet.train(_train[0], _train[1], _iterations, _save)

    # Saving states if _save is given
    if _save is not None:
        _nnet.save_state(_save)

    # Loading the datasets for testing    
    _test = ld.load_datasets(_data, "test")

    # Testing the network
    _nnet.test(_test[0], _test[1], _save)

#####################################################################
        
if __name__ == "__main__":
   main(sys.argv[1:])
