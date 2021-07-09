import pandas as pd
import json
import os,sys
sys.path.append('../')
import pyarrow
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib.ticker as tck
plt.style.use('classic')
rcParams['figure.facecolor'] = '1'
rcParams['patch.force_edgecolor'] = False

import tensorflow as tf
from joblib import load, dump

from IFAE_NNTools.TrainingFrame import *
from IFAE_NNTools.WeightScaler import *
from IFAE_NNTools.BackgroundRandomizer import *
from IFAE_NNTools.WeightedStandardScaler import *
from IFAE_NNTools.NNTools import *

outputdir = "./Plots"
user = "salvador"
pandainput = "pandas_Wh_v1_6jin4bin.feather"

ifold = 0
#optmass = [400]
optmass = "MP"

print(ifold, optmass)

if True:
    #Nicola Build
    learningrate = 0.00428095
    dropout = 0.4
    bsize = 2048
    structure = [196,196,196,196]
    verbose = True
    masses = optmass
    if optmass =="MP":
        masses = [250,300,350,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000]

if False:
    #Our build
    learningrate = 0.001
    dropout = 0.1
    bsize = 128
    structure = [64,64]
    verbose = True
    masses = [400]

print(tf.python.client.device_lib.list_local_devices())
print(tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

if not os.path.isdir(outputdir):
    print("Making output dir",outputdir)
    os.makedirs(outputdir, exist_ok=True)
if not os.path.isdir("/tmp/"+user):
    print ("Making tmp dir")
    os.mkdir("/tmp/"+user)

df_mc = pd.read_feather("/tmp/"+user+"/"+pandainput)
print (df_mc.shape)
print (df_mc.columns.unique())

columns_suffix= ['pt','eta','phi','e']
for s in columns_suffix:
    df_mc['el_'+s]=pd.DataFrame(df_mc['el_'+s].tolist(), index= df_mc.index)
    df_mc['mu_'+s]=pd.DataFrame(df_mc['mu_'+s].tolist(), index= df_mc.index)
    df_mc.loc[np.isnan(df_mc['el_'+s])==False,'lepton_'+s]=df_mc[np.isnan(df_mc['el_'+s])==False]['el_'+s]
    df_mc.loc[np.isnan(df_mc['mu_'+s])==False,'lepton_'+s]=df_mc[np.isnan(df_mc['mu_'+s])==False]['mu_'+s]
df_mc= df_mc.drop(columns=['el_pt','el_eta','el_phi','el_e','mu_pt','mu_eta','mu_phi','mu_e']) 

features = ['MaxMVA_Response','mVH','H_mass','higgs_pt','wlep_pt','whad_m','whad_pt','mWT', 'Hpt_over_mVH', 'Wpt_over_mVH','DeltaPhi_HW','min_DeltaPhiJETMET', 'HT_jets', 'Centrality','met_met']

print("%5s %20s   %12s   %12s"%("index","Variable","min","max"))
j=0
var_min = []
var_max = []
for i in features:    #Table to save the min and max ranges of variables for the histogram
    var_min.append(df_mc[i].min())
    var_max.append(df_mc[i].max())
    print("%5d %20s   %12.3f   %12.3f"%(j,i,var_min[j],var_max[j]))
    j=j+1

tframe = TrainingFrame(df_mc[df_mc.weight!=0],features)
tframe.signalprefix="Hp"
tframe.massvar="Hp_"
tframe.foldvar="eventNumber"
tf_ = tframe.prepare(masses=masses,addmass=True,ifold=ifold)

X_sets=tf_[0:3]
y_sets=tf_[3:6]
w_sets=tf_[6:]
del df_mc, tf_

print("Shape of training, validation and test datasets",X_sets[0].shape,X_sets[1].shape,X_sets[2].shape)

msb = WeightScaler()
for i in range(0,2):
    msb.fit(X_sets[i],y_sets[i],w_sets[i])
    msb.transform(X_sets[i],y_sets[i],w_sets[i])
    print("Final sum for bkg and signal",w_sets[i][y_sets[i]==-1].sum(),w_sets[i][y_sets[i]!=-1].sum())
    
rnd = BackgroundRandomizer(verbose=True)
rnd.massvar = "Hp_mass"
for i in range(0,2):
    print("Set",i,"\n")
    rnd.fit(X_sets[i],y_sets[i],w_sets[i])
    rnd.transform(X_sets[i],y_sets[i],w_sets[i])
    print("output",w_sets[i][y_sets[i]<0].sum(),w_sets[i][y_sets[i]>0].sum())

wss = WeightedStandardScaler( )
wss.fit(X_sets[0],y_sets[0],w_sets[0])
for i in range(0,2):
    wss.transform(X_sets[i],y_sets[i],w_sets[i])
wss.export(X_sets[0],y_sets[0],w_sets[0],outputdir,str(ifold))
print(wss.mean_,wss.var_,wss.scale_)
print(np.average(X_sets[0],axis=0,weights=w_sets[0]))
print(variance(X_sets[0],weights=w_sets[0]))

for i in range(0,2):
    if optmass!="MP":
        X_sets[i] = X_sets[i].drop(columns=["Hp_mass"])
    y_sets[i] = ( y_sets[i] > 0)

print("Shape of training, validation and test datasets",X_sets[0].shape,X_sets[1].shape,X_sets[2].shape)

modelNN=FeedForwardModel(configuration=structure,dropout=dropout,verbose=verbose, input_dim=X_sets[0].shape[1],learningr=learningrate)
resultNN=modelNN.train(X_sets[0], y_sets[0], w_sets[0],(X_sets[1], y_sets[1], w_sets[1]),batch_size = bsize, epochs=300,patience=5)
print(resultNN)
arch_file=open(outputdir+'/architecture_'+str(ifold)+'.h5','w') #save architecture for analysis framework
arch_file.write(modelNN.model.to_json())
arch_file.close()
modelNN.model.save_weights(outputdir+'/weights_'+str(ifold)+'.h5') #Save weights for analysis framework
modelNN.plotTrainingValidation(outputdir+'/Validation_'+str(ifold)) #in HpKerasUtils to save performance of each epoch
PlotAUCandScore(modelNN,X_sets[1],y_sets[1],w_sets[1],outputdir+'/Validation_'+str(ifold)) #Validating the NN input
