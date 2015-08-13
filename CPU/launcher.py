import os, sys, getopt

import numpy as np

def launcher(argv):

    # Parameter initialization
    _neurons = [784, 25, 784]
    _iter    = 2000
    _batch   = 50
    _type    = "AC"
    _save    = "default"

    _set     = "MNIST"
    _data    = "MNIST"

    # Some can be passed in argument
    try:
        opts, args = getopt.getopt(argv, "hd:i:b:s:t:", ["list="])

    except getopt.GetoptError:
        print 'launcher.py -i <nb_iters> -b <batch_size> -s <save_name> -d <data_name> --list <neurons_list> -t <type of network>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'launcher.py -i <nb_iters> -b <batch_size> -s <save_name> -d <data_name> --list <neurons_list> -t <type of network>'
            sys.exit()

        elif opt == '-i':
            _iter = int(arg)
            
        elif opt == '-b':
            _batch = int(arg)

        elif opt == '-s':
            _save = arg

        elif opt == '-d':
            _data = arg
            _set  = arg
            
        elif opt == '--list':
            _neurons = map(int, arg.split(','))

        elif opt == '-t':
            _type = arg

    for i in np.arange(len(_neurons)-1):

        _str = _save + str(i)
        
        # Intermediate neural network list of neurons
        _nlist = "{0},{1},{0}".format(_neurons[i],
                                      _neurons[i+1])

        # Calling main program
        _call = "/opt/anaconda/bin/python main.py -i {0} -b {1} -s {2} -d {3} --list {4}".format(_iter, _batch, _str, _set, _nlist)
            
        print _call            
        os.system(_call)

        _set = _str

    # Final part, train the deep network for decision
        
    os.system("mkdir ../states/{0}; mv ../states/*.txt $_; mkdir ../datasets/{0}; mv ../datasets/*.txt $_; mkdir ../img/{0}; mv ../img/*.png $_".format(_save))
    
    for i in np.arange(len(_neurons)-1):

        os.system("cp ../states/{0}/{0}{1}_W0.txt ../states/{0}_W{1}.txt; cp ../states/{0}/{0}{1}_B0.txt ../states/{0}_B{1}.txt".format(_save, i))


    print "Final training...\n"

    _neurons = str(_neurons).strip('[]').replace(' ', '')

    _call = "/opt/anaconda/bin/python main.py -i {0} -b {1} -s {2} -d {3} --list {4} -t {5}".format(_iter, _batch, _save, _data, _neurons, _type)

    print _call
    os.system(_call)

######################################################################
            
if __name__ == "__main__":
    launcher(sys.argv[1:])
