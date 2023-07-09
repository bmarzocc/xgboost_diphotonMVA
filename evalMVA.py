
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import uproot3 as uproot
import operator
import pickle
import ROOT
import os
import math

from array import array
from optparse import OptionParser
from xgboost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from copy import copy
from datetime import datetime
from sklearn import metrics

def transform(mva):
  return (1. / ( 1. + math.exp( 0.5*math.log( 2./(1. + mva) - 1 ) ) ) )
  
def getVal(tree,var):

  val = -999.
  if(var=='leadptom'): val = tree.leadptom 
  if(var=='subleadptom'): val = tree.subleadptom
  if(var=='leadeta'): val = tree.leadeta
  if(var=='subleadeta'): val = tree.subleadeta 
  if(var=='leadmva'): val = tree.leadmva 
  if(var=='subleadmva'): val = tree.subleadmva  
  if(var=='vtxprob'): val = tree.vtxprob
  if(var=='CosPhi'): val = tree.CosPhi 
  if(var=='sigmawv'): val = tree.sigmawv 
  if(var=='sigmarv'): val = tree.sigmarv
  if(var=='pho1_ptOverMgg'): val = tree.pho1_ptOverMgg
  if(var=='pho2_ptOverMgg'): val = tree.pho2_ptOverMgg
  if(var=='pho1_eta'): val = tree.pho1_eta 
  if(var=='pho2_eta'): val = tree.pho2_eta
  if(var=='pho1_idmva'): val = tree.pho1_idmva
  if(var=='pho2_idmva'): val = tree.pho2_idmva
  
  #else: print 'getVal ---> WARNING MISSING VAR: ',var
  #print("getVal: ",var,val)

  return val
  
def addBranches(tree,weights,names,vars):

 copyTree = tree.CopyTree('')
 copyTree.SetBranchStatus('*',1)
 
 mvaNew = [] 
 mvaNew_transf = [] 
 mvaNew_branch = []
 mvaNew_transf_branch = []
 for i,name in enumerate(names):
   mvaNew.append(array( 'd', [ -999. ] ))
   mvaNew_transf.append(array( 'd', [ -999. ] ))
   mvaNew_branch.append(copyTree.Branch(str(name), mvaNew[i], str(name)+'/D'))
   mvaNew_transf_branch.append(copyTree.Branch(str(name)+'_transformed', mvaNew_transf[i], str(name)+'_transformed/D')) 

 nameVars = vars
 nameVarsOther = vars
 #nameVarsOther =   ['pho1_ptOverMgg', 'pho2_ptOverMgg','pho1_eta', 'pho2_eta', 'pho1_idmva', 'pho2_idmva', 'vtxprob', 'CosPhi', 'sigmawv', 'sigmarv']

 ROOT.TMVA.Tools.Instance()
 ROOT.TMVA.PyMethodBase.PyInitialize()
 
 models = []
 for i in range(0,len(weights)): 
   models.append(ROOT.TMVA.Reader("Color:!Silent"))
 
 inputVars = {}
 for i,var in enumerate(nameVars):
   inputVars[var] = array('f', [-999.])
   for j in range(0,len(models)):
     models[j].AddVariable(var, inputVars[var])
   copyTree.SetBranchAddress(nameVarsOther[i], inputVars[var])
   
 for j in range(0,len(models)):  
   models[j].BookMVA("BDT",weights[j])

 for i,event in enumerate(copyTree):
   if i>copyTree.GetEntries():
     break 
   if i%100000==0:
     print("Reading Entry - ",i)
       
   for var in nameVarsOther:
     inputVars[var] = getVal(copyTree,var)
   
   for i in range(0,len(models)): 
     mva = models[i].EvaluateMVA("BDT")
     transformed_mva = transform(mva)
     mvaNew[i][0] = mva
     mvaNew_transf[i][0] = transformed_mva
     mvaNew_branch[i].Fill()
     mvaNew_transf_branch[i].Fill()
    

if __name__ == '__main__': 

 ROOT.gROOT.SetBatch(ROOT.kTRUE)

 parser = OptionParser()
 parser.add_option("-t", "--inTree", action="store", type="string", dest="inTree")
 parser.add_option("-w", "--weights", action="store", type="string", dest="weights")
 parser.add_option("-v", "--vars", action="store", type="string", dest="vars")
 (options, args) = parser.parse_args() 
 
 inTree = options.inTree 
 weights = options.weights.split(',') 
 vars = options.vars.split(',')
 
 names = ['newMVA']
 for i in range(1,len(weights)):
   names.append('newMVA'+str(i+1)) 
 
 outName = ''
 for string in inTree.split('/'):
   outName += string+'/' 
   if '.root' in string: break
 outName=outName.replace('.root/','_newMVA.root')
 
 outFile = ROOT.TFile(outName,'RECREATE')
 tree = ROOT.TChain()
 tree.AddFile(inTree)
 addBranches(tree,
 weights,
 names,
 vars)
 outFile.Write()
 outFile.Close()

 
