
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

def loadSignal():

  df_sig = uproot.open('Sig125_UL16_preVFP_dataDriven_absWeights_wSig_newMVA.root')['Sig125'].pandas.df()
  df_sig['weight'] = df_sig['weight_pure']
  return df_sig
    
def loadBkgs():

  df_bkg = uproot.open('Bkgs_UL16_preVFP_dataDriven_absWeights_wSig_newMVA.root')['Bkgs'].pandas.df()
  df_bkg['weight'] = df_bkg['weight_pure'] * df_bkg['Norm_SFs']
  return df_bkg  
  
def normalizeWeight(df,weight,negW0,negW1,absW):

  sumW = df.sum(0)[str(weight)]
  if negW0:
    df[str(weight)] = df[str(weight)].apply(lambda x: x if x>=0. else 0.)  
  if negW1:   
    df[str(weight)] = df[str(weight)].apply(lambda x: x if x>=0. else 1.)
  if absW:
    df[str(weight)] = df[str(weight)].abs()  
  sumW_norm = df.sum(0)[str(weight)]
  df[str(weight)] = df[str(weight)].apply(lambda x: x*(sumW/sumW_norm))
  
  return df
  
def drawScores(predY_train,predY_test,W_train,W_test,nBins,c_min,c_max,name):

  train_Sig = predY_train[Y_train==1]
  train_Bkg = predY_train[Y_train==0]
  test_Sig = predY_test[Y_test==1]
  test_Bkg = predY_test[Y_test==0]
  
  W_trainSig_sum = W_train[Y_train==1].sum(0) 
  W_trainBkg_sum = W_train[Y_train==0].sum(0)
  W_testSig_sum = W_test[Y_test==1].sum(0)
  W_testBkg_sum = W_test[Y_test==0].sum(0)
  
  #Get histograms of the classifiers
  Histo_train_Sig = np.histogram(train_Sig,bins=int(nBins),range=(c_min,c_max),weights=W_train[Y_train==1])
  Histo_train_Bkg = np.histogram(train_Bkg,bins=int(nBins),range=(c_min,c_max),weights=W_train[Y_train==0])
  Histo_test_Sig = np.histogram(test_Sig,bins=int(nBins),range=(c_min,c_max),weights=W_test[Y_test==1]*(W_trainSig_sum/W_testSig_sum))
  Histo_test_Bkg = np.histogram(test_Bkg,bins=int(nBins),range=(c_min,c_max),weights=W_test[Y_test==0]*(W_trainBkg_sum/W_testBkg_sum))
 
  #Lets get the min/max of the Histograms
  AllHistos= [Histo_train_Sig,Histo_train_Bkg,Histo_test_Sig,Histo_test_Bkg]
  h_max = max([histo[0].max() for histo in AllHistos])*1.2
  h_min = max([histo[0].min() for histo in AllHistos])
 
  #Get the histogram properties (binning, widths, centers)
  bin_edges = Histo_train_Sig[1]
  bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
  bin_widths = (bin_edges[1:] - bin_edges[:-1])
 
  #To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
  ErrorBar_test_Sig = np.sqrt(Histo_test_Sig[0])
  ErrorBar_test_Bkg = np.sqrt(Histo_test_Bkg[0])  
  
  #Plot  
  ax1 = plt.subplot(111)
  ax1.bar(bin_centers,Histo_train_Sig[0],facecolor='blue',linewidth=0,width=bin_widths,label='Sig (Train)',alpha=0.5)
  ax1.bar(bin_centers,Histo_train_Bkg[0],facecolor='red',linewidth=0,width=bin_widths,label='Bkg (Train)',alpha=0.5)
  ax1.errorbar(bin_centers, Histo_test_Sig[0], yerr=ErrorBar_test_Sig, xerr=None, ecolor='blue',c='blue',fmt='o',label='Sig (Test)')
  ax1.errorbar(bin_centers, Histo_test_Bkg[0], yerr=ErrorBar_test_Bkg, xerr=None, ecolor='red',c='red',fmt='o',label='Bkg (Test)')
  ax1.axvspan(0.0, c_max, color='blue',alpha=0.08)
  ax1.axvspan(c_min,0.0, color='red',alpha=0.08)
  ax1.axis([c_min, c_max, h_min, h_max])
  plt.title("BDT Signal--Bkg separation")
  plt.xlabel("BDT output")
  plt.ylabel("frequency")
  legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
  for alabel in legend.get_texts():
    alabel.set_fontsize('small')
  plt.savefig(name+".png",dpi=300)
  plt.savefig(name+".pdf",dpi=300)  
    
def drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,name):

  f,ax = plt.subplots(figsize=(8,8))

  fpr, tpr, threshold = metrics.roc_curve(Y_train,  predY_train, sample_weight=W_train)
  sorted_index = np.argsort(fpr)
  fpr_sorted =  np.array(fpr)[sorted_index]
  tpr_sorted = np.array(tpr)[sorted_index]
  train_auc = metrics.auc(fpr_sorted, tpr_sorted)
  ax.plot(fpr_sorted, tpr_sorted,label=f'Train:  AUC = {round(train_auc,3)}')

  fpr, tpr, threshold = metrics.roc_curve(Y_test,  predY_test, sample_weight=W_test)
  sorted_index = np.argsort(fpr)
  fpr_sorted =  np.array(fpr)[sorted_index]
  tpr_sorted = np.array(tpr)[sorted_index]
  test_auc = metrics.auc(fpr_sorted, tpr_sorted)
  ax.plot(fpr_sorted, tpr_sorted,label=f'Test:  AUC = {round(test_auc,3)}')

  plt.legend(loc='lower right')
  plt.title("ROC")
  plt.xlabel('1 - Background Rejection')
  plt.ylabel('Signal Efficiency')
  plt.savefig(name+".png",dpi=300)
  plt.savefig(name+".pdf",dpi=300)  
  
  print('AUC for training = %1.6f'%(train_auc) )
  print('AUC for test     = %1.6f'%(test_auc) )  


if __name__ == '__main__':
  
  ROOT.gROOT.SetBatch(ROOT.kTRUE)
  
  parser = OptionParser()
  parser.add_option("", "--negW1", action="store_true", dest="negW1")
  parser.add_option("", "--negW0", action="store_true", dest="negW0") 
  parser.add_option("", "--absW",  action="store_true", dest="absW")
  parser.add_option("", "--wSig",  action="store_true", dest="wSig")
  (options, args) = parser.parse_args() 
  
  negW1 = options.negW1
  negW0 = options.negW0  
  absW  = options.absW  
  wSig  = options.wSig  
  
  if negW1==0 and negW0==0 and absW==0:
    print("WARNING: you should speficy either 'negW1', either 'negW0', or 'absW'...")
    sys.exit()
  if (negW1==1 and negW0==1) or (negW0==1 and absW==1) or (negW1==1 and absW==1):
    print("WARNING: these options are exclusive, speficy either 'negW1', either 'negW0', either 'absW'...")
    sys.exit()
  
  print("Loading the Signal...")
  df_sig = loadSignal()
  
  print("Loading the Bkgs...")
  df_bkg = loadBkgs() 
  
  print("Sample loading done...")
  
  #Add target for signal (1)
  df_sig['target'] = np.ones(len(df_sig.index))
  df_sig['target'] = df_sig['target'].astype('int')

  #Add tasrget for backgrounds (0)
  df_bkg['target'] = np.zeros(len(df_bkg.index))
  df_bkg['target'] = df_bkg['target'].astype('int')
  
  #Apply additional weights
  if wSig:
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_wSig']
  if negW0:
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_posRatio']   
  if absW:       
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_absRatio']    

  #Deal with negative weights
  df_sig = normalizeWeight(df_sig,'weight',negW0,negW1,absW)
  sig_sum = df_sig.sum(0)['weight']
  
  #Deal with negative weights
  df_bkg = normalizeWeight(df_bkg,'weight',negW0,negW1,absW)
  bkg_sum = df_bkg.sum(0)['weight']
  df_bkg['weight'] = df_bkg['weight'].apply(lambda x: x*(sig_sum/bkg_sum))
  
  #Define inputs
  X_train_sig = df_sig[df_sig['isTest']<0.5]
  X_test_sig = df_sig[df_sig['isTest']>0.5]
  X_train_bkg = df_bkg[df_bkg['isTest']<0.5]
  X_test_bkg = df_bkg[df_bkg['isTest']>0.5]
  
  X_train = pd.concat((X_train_sig,X_train_bkg)) 
  X_test = pd.concat((X_test_sig,X_test_bkg)) 
  
  W_train = X_train['weight'].values
  Y_train = X_train['target'].values
  predY_train = X_train['newMVA'].values
  #predY_train = X_train['newMva_transformed'].values
  
  W_test = X_test['weight'].values
  Y_test = X_test['target'].values
  predY_test = X_test['newMVA'].values  
  #predY_test = X_test['newMVA_transformed'].values  
  
  if wSig:
    print('ROC with wSig:')  
    drawScores(predY_train,predY_test,W_train,W_test,40,-1.,1.,'BDT_wSig')
    drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,'ROC_wSig') 
  else:  
    print('ROC without wSig:') 
    drawScores(predY_train,predY_test,W_train,W_test,40,-1.,1.,'BDT_noWSig')
    drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,'ROC_noWSig')   
  
