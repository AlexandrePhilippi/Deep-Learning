import autoencoders as ac
import rbm          as rbm
import decision     as dc
import loader       as ld

import sys, getopt
    
def main(argv):

    # Parameter initialization
    _pSize       = (28,28)
    _neuronsList = [784, 25, 784]
    _iter        = 2000
    _batchSize   = 50
    _type        = "AC"

    _savename    = "default"
    _loadname    = None
    _dataname    = "MNIST"

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
            _type = arg

    # Creating the neural network
    if   _type == "AC":
        _nnet = ac.AUTOENCODERS(_neuronsList)
    elif _type == "RBM":
        _nnet = rbm.RBM(_neuronsList)
    elif _type == "DC":
        _nnet = dc.DECISION(_neuronsList)

    # Loading pretrained state if _loadname is given
    _nnet.load_state(_loadname)

    # Loading the MNIST datasets for training
    if _dataname == "MNIST":
        _train = [ld.mnist_train_img("../datasets/mnist"),
                  ld.mnist_train_lbl("../datasets/mnist")]

    elif _dataname == "CIFAR":
        _train = ld.cifar_10_train("../datasets/cifar-10")
        
    else:
        _train = ld.load_datasets(_dataname, "train")



    # Training the network
    _out = _nnet.train(_train, _iter, _batchSize, _savename)

    # Saving states if _savename is given
    _nnet.save_state(_savename)

    # Save output if _savename is given
    if _type != "DC" :
        _nnet.save_output(_savename, "train", _out)

    # Loading the MNIST datasets for testing    
    if _dataname == "MNIST":
        _test = [ld.mnist_test_img("../datasets/mnist"),
                 ld.mnist_test_lbl("../datasets/mnist")]

    elif _dataname == "CIFAR":
        _test = ld.cifar_10_test("../datasets/cifar-10")
        
    else:
        _test = ld.load_datasets(_dataname, "test")

    # Testing the network
    if _type == "DC":
        _nnet.test(_test)
    else:
        _nnet.test(_test[0], _savename, _pSize)


#####################################################################
        
if __name__ == "__main__":
   main(sys.argv[1:])
