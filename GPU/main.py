import autoencoders as ac
import rbm          as rbm
import decision     as dc
import loader       as ld
import numpy        as np
import display      as dy

import sys, getopt
    
def main(argv):

    # Parameter initialization
    _imgSize     = 784
    _nbPatch     = 60000
    _pSize       = (28,28)
    _neuronsList = [_imgSize, 25, _imgSize]
    _iter        = 2000
    _batchSize   = 50
    _type        = 0 # 0: Autoencoders, 1: RBM, 2: Decision

    _savename    = None
    _loadname    = None
    _dataname    = None

    # Some can be passed in argument
    try:
        opts, args = getopt.getopt(argv, "hp:i:b:s:l:d:t:", ["list="])

    except getopt.GetoptError:
        print 'main.py -i <nb_iters> -p <patch_size> -b <batch_size> -s <save_name> -l <load_name> -d <datasets> --list <neurons_list> -t<type of network>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'main.py -i <nb_iters> -p <patch_size> -b <batch_size> -s <save_name> -l <load_name> -d <datasets> --list <neurons_list> -t <type of network>'
            sys.exit()

        elif opt == '-i':
            _iter = int(arg)

        elif opt == '-p':
            _pSize = map(int, arg.split(','))
            
        elif opt == '-b':
            _batchSize = int(arg)

        elif opt == '-s':
            _savename = arg

        elif opt == '-l':
            _loadname = arg

        elif opt == '-d':
            _dataname = arg
            
        elif opt == '--list':
            _neuronsList = map(int, arg.split(','))

        elif opt == '-t':
            _type = int(arg)
            
    # Creating the neural network
    if _type == 1:
        _nnet = rbm.RBM(_neuronsList)
    elif _type == 2:
        _nnet = dc.DECISION(_neuronsList)
    else:
        _nnet = ac.AUTOENCODERS(_neuronsList)

    # Loading pretrained state
    if _loadname is not None:
        _nnet.load_state(_loadname)

    # Loading the MNIST datasets for training
    if _dataname is not None:
        _trainsets = ld.load_datasets(_dataname, "train")
    else:
        _train = [ld.mnist_train_img("../datasets"),
                  ld.mnist_train_lbl("../datasets")]

    # Training the network
    _nnet.train(_train, _iter, _batchSize, _savename)

    # Saving states
    if _savename is not None:
        _nnet.save_state(_savename)

    # Save output in order to have a trainset for next layers
    if _savename is not None:    
        _out = _nnet.create_datasets(_train[0])
        _nnet.save_output(_savename, "train", _out)

    # Loading the MNIST datasets for testing    
    if _dataname is not None:
        _testsets = ld.load_datasets(_dataname, "test")
    else:
        _test = [ld.mnist_test_img("../datasets"),
                 ld.mnist_test_lbl("../datasets")]

    # Should not stay like this ...
    if _type == 2:
        print "Testing the network ..."
        print "Performances :", _nnet.evaluate(_test)
        
    else:
        # Testing the network
        _out = _nnet.test(_test[0])

        # Displaying the results
        dy.display(_savename, [_test[0], _out],
                   10000, _pSize, "output")
    
        # Save output in order to have a testset for next layers
        if _savename is not None:
            _out = _nnet.create_datasets(_test[0])
            _nnet.save_output(_savename, "test", _out)

        # Displaying an approximated vision of first hidden
        # layer neurons
        _res = _nnet.neurons_visions()
        dy.display(_savename, [_res], _neuronsList[1],
                   _pSize, "neurons", 5, 5)


#####################################################################
        
if __name__ == "__main__":
   main(sys.argv[1:])
