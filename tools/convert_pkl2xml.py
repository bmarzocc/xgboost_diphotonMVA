#import FWCore.ParameterSet.Config as cms
from time import time,ctime
import sys,os
from tree_convert_pkl2xml import tree_to_tmva, BDTxgboost, BDTsklearn
import sklearn
from collections import OrderedDict
#print('The scikit-learn version is {}.'.format(sklearn.__version__))
import pandas
#print('The pandas version is {}.'.format(pandas.__version__))
import pickle
#print('The pickle version is {}.'.format(pickle.__version__))
import numpy as np
#print('The numpy version is {}.'.format(np.__version__))
#sys.path.insert(0, '/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/py2-pippkgs_depscipy/3.0-njopjo7/lib/python2.7/site-packages')
import xgboost as xgb
#print('The xgb version is {}.'.format(xgb.__version__))
import subprocess
# from sklearn.externals import joblib
# from itertools import izip
from optparse import OptionParser, make_option
from  pprint import pprint

def main(options,args):

    inputFile = options.inFile
    outputFile = inputFile.split('/')[-1].replace('.pkl','_weights.xml')

    features=options.inVars.split(',')
    print('Features:',features)
    
    result=-20
    fileOpen = None
    try:
        fileOpen = open(inputFile, 'rb')
    except IOError as e:
        print('Couldnt open or write to file (%s).' % e)
    else:
        print ('file opened')
        try:
            file = options.inFile
            print("the pkl file is:", file) 
            with open(file, 'rb') as f:  
                pkldata = pickle.loads(f.read(), encoding='bytes')
            print( pkldata)
        except :
            print('Oops!',sys.exc_info()[0],'occured.')
        else:
            print ('pkl loaded')
            print (pkldata)

            bdt = BDTxgboost(pkldata, features, ["Background", "Signal"])
            bdt.to_tmva(outputFile)
            print("xml file is created with name : ", outputFile)

            if options.test:#this is just for testing if you want to check on one event uncomment here
                proba = pkldata.predict_proba([[new_dict[feature] for feature in features]])
                #proba = pkldata.predict_proba([[ new_dict[feature] for feature in features]])[:,pkldata.n_classes_-1].astype(np.float64)
                print ("proba= ",proba)
                result = proba[:,1][0]
                print ('predict BDT to one event',result)
                

             #   test_eval = bdt.eval([ new_dict[feature] for feature in features])
             #   print "XGboost test_eval = ", test_eval
             #   test_eval_tmva = bdt.eval_tmva([ new_dict[feature] for feature in features])
             #   print "TMVA test_eval = ", test_eval_tmva

            fileOpen.close()
    return result

if __name__ == "__main__":
    parser = OptionParser(option_list=[
            make_option("-i", "--infile",
                        action="store", type="string", dest="inFile",
                        default="",
                        help="input file",
                        ),
            make_option("-v", "--vars",
                        action="store", type="string", dest="inVars",
                        default="",
                        help="input variables",
                        ),            
            make_option("-t", "--test",
                        action="store_true", dest="test",
                        default=False,
                        help="test on one event",
                        ),
            ]
                          )

    (options, args) = parser.parse_args()
    sys.argv.append("-b")

    
    pprint(options.__dict__)

    import ROOT

    main(options,args)
