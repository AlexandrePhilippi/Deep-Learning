import os, sys, getopt

import numpy as np

def launcher(argv):

    # Parameter initialization
    _neurons = [784, 25, 784]
    _iter    = 2000
    _batch   = 50

    _data    = "mnist"
    _save    = "default"

    # Some can be passed in argument
    try:
        opts, args = getopt.getopt(argv, "hd:i:b:s:", ["list="])

    except getopt.GetoptError:
        print 'launcher.py -i <iterations> -b <batch> -s <save> -d <data> --list <neurons>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'launcher.py -i <iterations> -b <batch> -s <save> -d <data> --list <neurons>'
            sys.exit()

        elif opt == '-i':
            _iter = int(arg)
            
        elif opt == '-b':
            _batch = int(arg)

        elif opt == '-s':
            _save = arg

        elif opt == '-d':
            _data = arg
            
        elif opt == '--list':
            _neurons = map(int, arg.split(','))


    _set = _data

    for i in xrange(len(_neurons)-1):

        _str = _save + str(i)
        
        # Intermediate neural network list of neurons
        _list = "{0},{1},{0}".format(_neurons[i], _neurons[i+1])

        # Calling main program
        _call = "/opt/anaconda/bin/python main.py -i {0} -b {1} -s {2} -d {3} --list {4}".format(_iter, _batch, _str, _set, _list)
            
        print _call            
        os.system(_call)

        _set = _str

    # Final part, train the deep network for decision

    _call = "mkdir ../states/{0}".format(_save)
    os.system(_call)
    _call = "mv ../states/*.txt $_".format(_save)
    os.system(_call)
    _call = "mkdir ../datasets/{0}".format(_save)
    os.system(_call)
    _call = "mv ../datasets/*.txt $_".format(_save)
    os.system(_call)
    _call = "mkdir ../img/{0}".format(_save)
    os.system(_call)
    _call = " mv ../img/*.png $_".format(_save)
    os.system(_call)

    
    for i in xrange(len(_neurons)-1):
        
        os.system("cp ../states/{0}/{0}{1}_W0.txt ../states/{0}_W{1}.txt; cp ../states/{0}/{0}{1}_B0.txt ../states/{0}_B{1}.txt".format(_save, i))


    print "Final training...\n"

    _neurons = str(_neurons).strip('[]').replace(' ', '')

    _call = "/opt/anaconda/bin/python main.py -i {0} -b {1} -s {2} -d {3} --list {4}".format(_iter, _batch, _save, _data, _neurons)

    print _call
    os.system(_call)

######################################################################
            
if __name__ == "__main__":
    launcher(sys.argv[1:])
