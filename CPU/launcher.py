import os, sys, getopt

import numpy as np

def launcher(argv):

    # Parameter initialization
    _imgSize     = 784
    _pSize       = (28,28)
    _neuronsList = [_imgSize, 25, _imgSize]
    _iter        = 2000
    _batchSize   = 50
    _type        = 0 # 0: Autoencoders, 1: RBM, 2: Decision

    _savename    = None

    # Some can be passed in argument
    try:
        opts, args = getopt.getopt(argv, "hp:i:b:s:t:", ["list="])

    except getopt.GetoptError:
        print 'launcher.py -i <nb_iters> -p <patch_size> -b <batch_size> -s <save_name> --list <neurons_list> -t <type of network>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'launcher.py -i <nb_iters> -b <batch_size> -s <save_name> --list <neurons_list> -t <type of network>'
            sys.exit()

        elif opt == '-i':
            _iter = int(arg)
            
        elif opt == '-b':
            _batchSize = int(arg)

        elif opt == '-s':
            _savename = arg
            
        elif opt == '--list':
            _neuronsList = map(int, arg.split(','))

        elif opt == '-t':
            _type = arg

    for i in np.arange(len(_neuronsList) / 2):

        print "Training {0} layer.".format{i}
        
        # Use MNIST or intermediate dataset
        try:
            _str
        except NameError:
            _dataname = None
        else:
            _dataname = _str
            
        _str = _savename + str(i)
        
        # Intermediate neural network list of neurons
        _nlist = "{0},{1},{0}".format(_neuronsList[i],
                                      _neuronsList[i+1])

        # Patch dimension
        _pSize = np.sqrt(_neuronsList[i])
        _dim   = "{0},{0}".format(int(_pSize))

        # Calling main program
        if _dataname is None:
            _call = "python main.py -i {0} -p {1} -b {2} -s {3} --list {4} -t {5}".format(_iter, _dim, _batchSize, _str, _nlist, _type)
            
        else:
            _call = "python main.py -i {0} -p {1} -b {2} -s {3} -d {4} --list {5} -t {6}".format(_iter, _dim, _batchSize, _str, _dataname, _nlist, _type)

        os.system(_call)

######################################################################
            
if __name__ == "__main__":
    launcher(sys.argv[1:])
