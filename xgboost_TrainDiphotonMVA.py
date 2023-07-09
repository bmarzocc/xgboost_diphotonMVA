
import ROOT
import sys
import operator
import pickle
import pickle
import json 
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import uproot3 as uproot

from xgboost import *
from optparse import OptionParser
from copy import copy
from datetime import datetime
from scipy import integrate
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

def optimization(X_train,X_test,Y_train,Y_test,W_train,W_test,max_evals,n_estimators,seed):

  trials = Trials()
  
  space = {
        'booster': 'gbtree',
        'eval_metric': 'auc',  
        'objective': 'binary:logistic',  
        'learning_rate': hp.uniform('learning_rate',0.,1.),
        'max_depth': hp.quniform('max_depth', 1, 100, 1),
        'n_estimators': 10,
        'seed': seed
  }
  '''
  space = {
        'booster': 'gbtree',
        'eval_metric': 'auc', 
        'objective': 'binary:logistic',  
        'max_depth': hp.quniform("max_depth", 0, 100, 1),
        'gamma': hp.uniform ('gamma', 0.,10.),
        'learning_rate': hp.uniform('learning_rate',0.,1.),
        'reg_alpha' : hp.quniform('reg_alpha', 0,200,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0.,1.),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.,1.),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': n_estimators,
        'seed': seed
  }
  '''
  
  def objective(space):
    clf = xgb.XGBRegressor(
      booster=space['booster'], eval_metric=space['eval_metric'], objective=space['objective'], learning_rate=space['learning_rate'], max_depth=int(space['max_depth']), n_estimators=int(space['n_estimators']), seed=int(space['seed'])
    )     
    #clf = xgb.XGBRegressor(
    #  booster=space['booster'], eval_metric=space['eval_metric'], objective=space['objective'], max_depth=int(space['max_depth']), gamma=space['gamma'], learning_rate=space['learning_rate'], reg_alpha=int(space['reg_alpha']), reg_lambda=space['reg_lambda'], colsample_bytree=space['colsample_bytree'], min_child_weight=int(space['min_child_weight']), n_estimators=int(space['n_estimators']), seed=int(space['seed'])
    #) 
    evaluation = [( X_train, Y_train), ( X_test, Y_test)]  
    clf.fit(X_train, Y_train,sample_weight=W_train,eval_set=evaluation,verbose=False)
    predY_test = clf.predict(X_test)
    loss = -1*roc_auc_score(Y_test, predY_test, sample_weight=W_test) 
    accuracy = accuracy_score(Y_test, predY_test>0.5)
    fpr, tpr, threshold = metrics.roc_curve(Y_test,  predY_test, sample_weight=W_test)
    test_auc = metrics.auc(fpr, tpr)
    print ('SCORE:', accuracy, loss, test_auc)
    return {'loss': -test_auc, 'status': STATUS_OK }
  
  best_hyperparams = fmin(fn=objective,space=space,algo=tpe.suggest,max_evals=max_evals,trials=trials)
  print("The best hyperparameters are : ")
  print(best_hyperparams)
  
def makeTree(df,treeVars,treeName,fileName):
  branches = {}
  for var in treeVars:
    branch = df[var].to_numpy()
    branches[var] = branch 
  tree = ROOT.RDF.FromNumpy(branches)
  tree.Snapshot(treeName,fileName)
  
def loadSignal(wSig):

  df_sig = uproot.open("../Sig/Summer20UL16_preVFP_Untagged/Sig125_negReweighting.root")['Sig125'].pandas.df()
  df_sig['weight_pure'] = df_sig['weight']
  
  #Compute weight_wSig
  if wSig:
    df_sig['weight_wSig'] = (df_sig['vtxprob']/df_sig['sigmarv'] + (1-df_sig['vtxprob'])/df_sig['sigmawv'])
  else:
    df_sig['weight_wSig'] = np.ones(len(df_sig.index))   
    
  return df_sig

def loadDataDrivenBackground():

  df_PP = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/DataDriven_QCD_PP_Sig.root')['pp'].pandas.df()
  df_DataDriven_QCD = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/DataDriven_QCD_PP_Sig.root')['DataDriven_QCD'].pandas.df()
  
  df_bkg = df_PP
  df_bkg = pd.concat((df_bkg,df_DataDriven_QCD))
  df_bkg['weight_pure'] = df_bkg['weight']
  df_bkg['weight'] = df_bkg['weight'] * df_bkg['Norm_SFs']
  
  return df_bkg
  
def loadMCBackground(scaleQCD):
  
  df_DiPhotonJetsBox = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/DiPhotonJetsBox_MGG-80toInf_13TeV-sherpa.root')['tagsDumper/trees/DiPhotonJetsBox_MGG_80toInf_13TeV_sherpa_13TeV_UntaggedTag'].pandas.df()[vars]
  
  df_GJet_Pt_20to40 = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/GJet_Pt-20to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root')['tagsDumper/trees/GJet_Pt_20to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_UntaggedTag'].pandas.df()[vars]
  
  df_GJet_Pt_40toInf = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8.root')['tagsDumper/trees/GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8_13TeV_UntaggedTag'].pandas.df()[vars]

  df_QCD_Pt_30to40 = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/QCD_Pt-30to40_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV-pythia8.root')['tagsDumper/trees/QCD_Pt_30to40_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_pythia8_13TeV_UntaggedTag'].pandas.df()[vars]
  df_QCD_Pt_30to40['weight'] = df_QCD_Pt_30to40['weight'].apply(lambda x: x/(scaleQCD)) #scale QCD to 1/40.

  df_QCD_Pt_40ToInf = uproot.open('../Bkgs/Summer20UL16_preVFP_Untagged/QCD_Pt-40ToInf_DoubleEMEnriched_MGG-80ToInf_TuneCP5_13TeV-pythia8.root')['tagsDumper/trees/QCD_Pt_40ToInf_DoubleEMEnriched_MGG_80ToInf_TuneCP5_13TeV_pythia8_13TeV_UntaggedTag'].pandas.df()[vars]
  df_QCD_Pt_40ToInf['weight'] = df_QCD_Pt_40ToInf['weight'].apply(lambda x: x/(scaleQCD)) #scale QCD to 1/40.

  df_bkg = df_DiPhotonJetsBox
  df_bkg = pd.concat((df_bkg,df_GJet_Pt_20to40))
  df_bkg = pd.concat((df_bkg,df_GJet_Pt_40toInf))
  df_bkg = pd.concat((df_bkg,df_QCD_Pt_30to40))
  df_bkg = pd.concat((df_bkg,df_QCD_Pt_40ToInf)) 
  
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
  plt.title('ROC')
  plt.ylabel('1-Bkg Rejection')
  plt.xlabel('Signal Efficiency')
  plt.savefig(name+".png",dpi=300)
  plt.savefig(name+".pdf",dpi=300)  
  
  print('AUC for training = %1.6f'%(train_auc) )
  print('AUC for test     = %1.6f'%(test_auc) )  



if __name__ == '__main__':

  parser = OptionParser()
  parser.add_option("", "--vars",       action="store", type="string", dest="vars")
  parser.add_option("", "--useDMatrix", action="store_true",  dest="useDMatrix")  
  parser.add_option("", "--useMC"     , action="store_true",  dest="useMC")
  parser.add_option("", "--negW1"     , action="store_true",  dest="negW1")
  parser.add_option("", "--negW0"     , action="store_true",  dest="negW0") 
  parser.add_option("", "--absW"      ,  action="store_true", dest="absW")
  parser.add_option("", "--wSig"      ,  action="store_true", dest="wSig")
  parser.add_option("", "--opt"       ,  action="store_true", dest="opt")
  parser.add_option("", "--nSteps"    ,  type="int"         , dest="nSteps")
  (options, args) = parser.parse_args() 
  
  vars       = options.vars 
  useDMatrix = options.useDMatrix  
  useMC      = options.useMC  
  negW1      = options.negW1
  negW0      = options.negW0  
  absW       = options.absW  
  wSig       = options.wSig  
  opt        = options.opt  
  nSteps     = options.nSteps
  
  inputVars = vars.split(',')
  n_estimators = 20
  if nSteps:
    n_estimators = nSteps   

  if negW1==0 and negW0==0 and absW==0:
    print("WARNING: you should speficy either 'negW1', either 'negW0', or 'absW'...")
    sys.exit()
  if (negW1==1 and negW0==1) or (negW0==1 and absW==1) or (negW1==1 and absW==1):
    print("WARNING: these options are exclusive, speficy either 'negW1', either 'negW0', either 'absW'...")
    sys.exit()
        
  label = '_UL16_preVFP'
  if useMC:
    label = label + '_MC' 
  else:
    label = label + '_dataDriven' 
  if negW0:
    label = label + '_negWeights0'  
  if negW1:
    label = label + '_negWeights1' 
  if negW0:
    label = label + '_negWeights0' 
  if absW:  
    label = label + '_absWeights'
  if wSig:  
    label = label + '_wSig'
  if useDMatrix:  
    label = label + '_dMatrix'
     
  print("Loading Signal...")
  df_sig = loadSignal(wSig)
        
  print("Loading Bkgs...")
  if useMC:
    df_bkg = loadMCBackground(40.)
  else:
    df_bkg = loadDataDrivenBackground()
    
  #Add target for signal (1)
  df_sig['target'] = np.ones(len(df_sig.index))
  df_sig['target'] = df_sig['target'].astype('int')

  #Add tasrget for backgrounds (0)
  df_bkg['target'] = np.zeros(len(df_bkg.index))
  df_bkg['target'] = df_bkg['target'].astype('int')
  
  #Define standard weights
  df_sig['weight_noWSig'] = df_sig['weight']
  df_bkg['weight_noWSig'] = df_bkg['weight']
  
  #Apply additional weights
  if wSig:
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_wSig']
  if negW0:
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_posRatio']  
    df_sig['weight_noWSig'] = df_sig['weight_noWSig'] * df_sig['weight_posRatio']  
  if absW:       
    df_sig['weight'] = df_sig['weight'] * df_sig['weight_absRatio']    
    df_sig['weight_noWSig'] = df_sig['weight_noWSig'] * df_sig['weight_absRatio']  
      

  #Deal with negative weights
  df_sig = normalizeWeight(df_sig,'weight',negW0,negW1,absW)
  df_sig = normalizeWeight(df_sig,'weight_noWSig',negW0,negW1,absW)
  sig_sum = df_sig.sum(0)['weight']
  sig_sum_noWSig = df_sig.sum(0)['weight_noWSig']
  
  #Deal with negative weights
  df_bkg = normalizeWeight(df_bkg,'weight',negW0,negW1,absW)
  df_bkg = normalizeWeight(df_bkg,'weight_noWSig',negW0,negW1,absW)
  bkg_sum = df_bkg.sum(0)['weight']
  bkg_sum_noWSig = df_bkg.sum(0)['weight']
  df_bkg['weight'] = df_bkg['weight'].apply(lambda x: x*(sig_sum/bkg_sum))
  df_bkg['weight_noWSig'] = df_bkg['weight_noWSig'].apply(lambda x: x*(sig_sum_noWSig/bkg_sum_noWSig))

  print("Loading done!...")
  
  #seed = int(datetime.now().timestamp())
  seed = 12345
  np.random.seed(seed)
  print("Seed:",seed)

  df_sig['isTest'] = np.zeros(len(df_sig.index))
  df_bkg['isTest'] = np.zeros(len(df_bkg.index))
  
  X = pd.concat((df_sig,df_bkg))
  Y = pd.concat((df_sig['target'],df_bkg['target']))

  test_size = 0.33
  #test_size = 0.5
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed,shuffle=1)
  
  X_train['isTest'] = X_train['isTest'].apply(lambda x: 0)
  X_test['isTest'] = X_test['isTest'].apply(lambda x: 1)

  X_sig = pd.concat((X_train[Y_train==1],X_test[Y_test==1])) 
  X_bkg = pd.concat((X_train[Y_train==0],X_test[Y_test==0])) 
  
  #Save Bkgs tree:
  treeVars = inputVars
  for var in ['weight_pure','Norm_SFs','isTest']:
    if var not in treeVars:
      treeVars.append(var)
  makeTree(X_bkg,treeVars,'Bkgs','Bkgs'+label+'.root')
 
  #Save Sig tree:
  for var in ['weight_pure','weight_wSig','weight_posRatio','weight_absRatio','isTest']:
    if var not in treeVars:
      treeVars.append(var) 
  makeTree(X_sig,treeVars,'Sig125','Sig125'+label+'.root')

  inputVars = vars.split(',')           
  print('Training inputVars:',inputVars)
  
  W_train = X_train['weight'].values
  W_train_noWSig = X_train['weight_noWSig'].values
  X_train = X_train[inputVars].values
  Y_train = Y_train.values
  
  W_test = X_test['weight'].values
  W_test_noWSig = X_test['weight_noWSig'].values
  X_test = X_test[inputVars].values
  Y_test = Y_test.values
  
  if opt:
    optimization(X_train,X_test,Y_train,Y_test,W_train,W_test,100,n_estimators,seed)
    sys.exit()
    
  params = {
    'booster': 'gbtree',
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'seed': seed,
    'learning_rate': 0.6777906311912216, 
    'max_depth': 61
    #'gamma': 0.1
  }
  
  if useDMatrix:    
    
    training = xgb.DMatrix(X_train,label=Y_train, weight=W_train, feature_names=inputVars)
    testing  = xgb.DMatrix(X_test, label=Y_test , weight=W_test, feature_names=inputVars) 
    
    model = xgb.train(params,training,n_estimators,evals=[(training, 'validation_0')],verbose_eval=True)
    predY_train = model.predict(training)
    predY_test  = model.predict(testing)
  
    config = json.loads(model.save_config())
    
  else:  
  
    clf = xgb.XGBRegressor(n_estimators=n_estimators, **params)
    model = clf.fit(X_train,Y_train,sample_weight=W_train,eval_set=[(X_train, Y_train)],sample_weight_eval_set=[W_train],verbose=True)
    predY_train = model.predict(X_train)
    predY_test  = model.predict(X_test)
    
    config = json.loads(model.get_booster().save_config())
  
  #json_formatted_str = json.dumps(config, indent=2)
  #print('Model:')
  #print(json_formatted_str)
                   
  drawScores(predY_train,predY_test,W_train,W_test,40,0.,1.,'BDT'+label)
  print('ROC with wSig:')
  drawROC(Y_train,Y_test,predY_train,predY_test,W_train,W_test,'ROC'+label)
  print('ROC without wSig:')
  drawROC(Y_train,Y_test,predY_train,predY_test,W_train_noWSig,W_test_noWSig,'ROC'+label+'_noWSig')
  
  input_vars=[]
  var_map = {}
  if useDMatrix:  
    sys.path.append('tools/')
    from xgboost2tmva_dmatrix import *        
    mdl = model.get_dump()
    input_vars=[]
    for i,key in enumerate(inputVars):
      input_vars.append((key,'F'))
    convert_model(mdl,input_variables=input_vars,output_xml='diphotonMVA_xgboost_model'+label+'_weights.xml')  
  else:
    pickle.dump(model, open('diphotonMVA_xgboost_model'+label+'.pkl', "wb"))
  
