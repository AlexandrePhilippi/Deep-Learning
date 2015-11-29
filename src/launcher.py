import sys
import loader       as ld
import display      as dy
import autoencoders as ac

DATASETS  = "cifar10"
SAVENAME  = "cifar"

EPOCHS    = 200
BATCHSIZE = 1000

LEARNING_RATE   = 0.1
MOMENTUM        = 0.5
SPARSITY_TARGET = 0.05
SPARSITY_COEF   = 3

NETWORK = (1024,256,1024)

def main(args):

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
