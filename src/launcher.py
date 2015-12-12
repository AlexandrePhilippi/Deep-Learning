import sys, getopt
import loader       as ld
import display      as dy
import autoencoders as ac

def main(argv):

    DATASETS  = "mnist"
    SAVENAME  = "cross"
    
    EPOCHS    = 200
    BATCHSIZE = 1000
    
    LEARNING_RATE   = 0.1
    MOMENTUM        = 0.5
    SPARSITY_TARGET = 0.05
    SPARSITY_COEF   = 3
    
    NETWORK = (784,64,784)

    try:
        opts, args = getopt.getopt(argv, "", ["help", "epochs=", "batchsize=", "epsilon=", "momentum=", "sparsity=", "layers=", "dataset=", "save="])

        for opt, arg in opts:

            if opt == '--help':
                print 'launcher.py --epochs <value> --batchsize <value> --epsilon <value> --momentum <value> --sparsity <value> --layers <list> --dataset <name> --save <name>'
                sys.exit()
                
            elif opt == '--epochs':
                EPOCHS = int(arg)
                
            elif opt == '--batchsize':
                BATCHSIZE = int(arg)
                
            elif opt == "--epsilon":
                LEARNING_RATE = int(arg)
                
            elif opt == "--momentum":
                MOMENTUM = int(arg)
                
            elif opt == "--sparsity":
                SPARSITY_COEF = int(arg)
                
            elif opt == "--layers":
                NETWORK = map(int, arg.split(','))
                
            elif opt == "--dataset":
                DATASETS = arg
                
            elif opt == "--save":
                SAVENAME = arg
                
    except getopt.GetoptError:
        pass
                
    print "launcher.py --epochs {0} --batchsize {1} --epsilon {2} --momentum {3} --sparsity {4} --layers {5} --dataset {6} --save {7}".format(EPOCHS, BATCHSIZE, LEARNING_RATE, MOMENTUM, SPARSITY_COEF, NETWORK, DATASETS, SAVENAME)

    _nnet = ac.Autoencoders(NETWORK,
                            LEARNING_RATE,
                            MOMENTUM,
                            SPARSITY_TARGET,
                            SPARSITY_COEF)


    if DATASETS == "mnist":
        _trainsets = ld.mnist_train_img("../datasets/mnist")
        _testsets = ld.mnist_test_img("../datasets/mnist")

        
    elif DATASETS == "cifar10":
        _trainsets = ld.cifar10_train("../datasets/cifar-10")[0]
        _testsets = ld.cifar10_test("../datasets/cifar-10")[0]
        
    else:
        print "Wrong datasets."
        exit(1)
        
    _nnet.train(EPOCHS, _trainsets, BATCHSIZE)

    _out = _nnet.test(_testsets)

    dy.display(SAVENAME, "out", _testsets, _out)

    _neurons = _nnet.neurons_vision()

    dy.display(SAVENAME, "neurons", _neurons)

##################################################################

if __name__ == "__main__":
    main(sys.argv[1:])
