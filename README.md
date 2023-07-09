# xgboost_diphotonMVA

1) Before running, setup an environment, e.g. conda-environment:
```
mkdir <some_path_where_you_have_more_disk_space>/.conda
mkdir <some_path_where_you_have_more_disk_space>/.conda/pkgs
mkdir <some_path_where_you_have_more_disk_space>/.conda/envs
chmod 755 -R <some_path_where_you_have_more_disk_space>/.conda/pkgs
conda config --prepend pkgs_dirs <some_path_where_you_have_more_disk_space>/.conda/pkgs
conda config --prepend envs_dirs <some_path_where_you_have_more_disk_space>/.conda/envs
conda env create -f environment.yml
```

Once the environment is set, activate it:

```
conda activate conda-env
```

2) Launch the training:

```
python3 xgboost_TrainDiphotonMVA.py --vars 'leadptom,subleadptom,leadeta,subleadeta,leadmva,subleadmva,vtxprob,CosPhi,sigmawv,sigmarv' --absW --wSig --nSteps 20
```

Options:

    * --vars: input training variables 
    * --useDMatrix: use xgboost DMatrix for training (NOTE: no difference wrt the standard training algorithm) 
    * --useMC: use MC as background, instead of the default DataDriven+PP backgrounds
    * --negW1: set all negative weights to 1 
    * --negW0: set all negative weights to 0              
    * --absW: set all negative weights to their absolute values  
    * --wSig: apply wSig
    * --opt: run model parameters optimization (NOTE: change manually the parameters you want to optimize in 'optimization' function)   
    * --nSteps: number of 'n_estimators' to be used in the training
            
3) Convert to pkl file to xml weight file (if --useDMatrix option wasn't used):

```
python3 tools/convert_pkl2xml.py --infile diphotonMVA_xgboost_model_UL16_preVFP_dataDriven_absWeights_wSig.pkl --vars 'leadptom,subleadptom,leadeta,subleadeta,leadmva,subleadmva,vtxprob,CosPhi,sigmawv,sigmarv'
```

Options:

    * --infile: input pkl file
    * --vars: input training variables 

4) Evaluate the MVA on signal and bkgs:

```
python3 evalMVA.py --inTree Sig125_UL16_preVFP_dataDriven_absWeights_wSig.root/Sig125 --weights 'diphotonMVA_xgboost_model_UL16_preVFP_dataDriven_absWeights_wSig_weights.xml' --vars 'leadptom,subleadptom,leadeta,subleadeta,leadmva,subleadmva,vtxprob,CosPhi,sigmawv,sigmarv'
python3 evalMVA.py --inTree Bkgs_UL16_preVFP_dataDriven_absWeights_wSig.root/Bkgs --weights 'diphotonMVA_xgboost_model_UL16_preVFP_dataDriven_absWeights_wSig_weights.xml' --vars 'leadptom,subleadptom,leadeta,subleadeta,leadmva,subleadmva,vtxprob,CosPhi,sigmawv,sigmarv'
```

Options:

    * --inTree: input tree
    * --weights: input xml weight files (NOTE: more then one file can be given as an input as a string: '... , ... , ...')
    * --vars: input training variables 
    
5) Draw the ROC curve with the weights you want:

```
python3 draw_ROC.py --absW --wSig
```

options:

    * --negW1: set all negative weights to 1 
    * --negW0: set all negative weights to 0              
    * --absW: set all negative weights to their absolute values  
    * --wSig: apply wSig
    


