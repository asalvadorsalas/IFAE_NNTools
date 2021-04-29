import pandas as pd
import json
import os,sys
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

outputdir = "./tXqML/Models/Test_"
user = "salvador"
pandainput = "pandas_tqXv2.feather"

reframelep_phi = True
reframelep_eta = True

ifold = 0
optmass = [90]
(uquark,cquark) = (True,False)

if len(sys.argv)>1:
    if "MP" in sys.argv[1]:
        optmass="MP"
        outputdir += sys.argv[1]
    elif "[" in sys.argv[1]:
        optmass = list(map(int, sys.argv[1].strip('[]').split(',')))
    else:
        optmass = [int(sys.argv[1])]
        outputdir += sys.argv[1]

if len(sys.argv)>2:
    uquark = sys.argv[2].lower() == 'true'
if len(sys.argv)>3:
    cquark = sys.argv[3].lower() == 'true'
if len(sys.argv)>4:
    ifold = int(sys.argv[4])

print(ifold, optmass, (uquark,cquark))

if True:
    #Nicola Build
    learningrate = 0.00428095
    dropout = 0.4
    bsize = 2048
    structure = [196,196,196,196]
    verbose = True
    masses = optmass
    if optmass =="MP":
        masses = [20,30,40,50,60,70,80,90,100,120,140,150,160]
    
if False:
    #Our build
    learningrate = 0.001
    dropout = 0.1
    bsize = 128
    structure = [64,64]
    verbose = True
    masses = [125]

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
#df_mc.loc[df_mc.process=="uX_160","weight"]=1.323172591E-05*df_mc.nomWeight_weight_btag*df_mc.nomWeight_weight_jvt*df_mc.nomWeight_weight_leptonSF*df_mc.nomWeight_weight_mc*df_mc.nomWeight_weight_pu
#df_mc.loc[df_mc.process=="cX_160","weight"]=1.323899104E-05*df_mc.nomWeight_weight_btag*df_mc.nomWeight_weight_jvt*df_mc.nomWeight_weight_leptonSF*df_mc.nomWeight_weight_mc*df_mc.nomWeight_weight_pu
df_mc = df_mc[~((df_mc.X_mass==160)&(df_mc.nomWeight_weight_mc > 700))]
df_mc["weight"]*=139000.0
print (df_mc.shape)
print (df_mc.columns.unique())

features = []
for i in range(0,6):
    features+=['jet'+str(i)+'_pt_bord', 'jet'+str(i)+'_eta_bord', 'jet'+str(i)+'_phi_bord', 'jet'+str(i)+'_m_bord', 'jet'+str(i)+'_btagw_discrete_bord']
features+=["lep1_pt","lep1_eta","lep1_phi","met","met_phi"]
features+=["mbb_leading_bjets","mbb_maxdr","mbb_mindr","m_jj_leading_jets"]

features.remove('jet0_btagw_discrete_bord')
features.remove('jet1_btagw_discrete_bord')
features.remove('jet2_btagw_discrete_bord')

if reframelep_phi:
    for i in range(0,6):
        df_mc["jet"+str(i)+"_phi_bord"]=df_mc["jet"+str(i)+"_phi_bord"]-df_mc["lep1_phi"]
        df_mc.loc[(df_mc["jet"+str(i)+"_phi_bord"]>np.pi),"jet"+str(i)+"_phi_bord"] -= 2*np.pi
        df_mc.loc[(df_mc["jet"+str(i)+"_phi_bord"]< -np.pi),"jet"+str(i)+"_phi_bord"] += 2*np.pi

    df_mc["met_phi"]=df_mc["met_phi"]-df_mc["lep1_phi"]
    df_mc.loc[(df_mc["met_phi"]>np.pi),"met_phi"] -= 2*np.pi
    df_mc.loc[(df_mc["met_phi"]< -np.pi),"met_phi"] += 2*np.pi
    
    features.remove("lep1_phi")

if reframelep_eta:
    for i in range(0,6):
        df_mc.loc[(df_mc.lep1_eta<0),"jet"+str(i)+"_eta_bord"] *= -1
    df_mc.loc[(df_mc.lep1_eta<0),"lep1_eta"] *= -1

print("%5s %20s   %12s   %12s"%("index","Variable","min","max"))
j=0
var_min = []
var_max = []
for i in features:    #Table to save the min and max ranges of variables for the histogram
    var_min.append(df_mc[i].min())
    var_max.append(df_mc[i].max())
    print("%5d %20s   %12.3f   %12.3f"%(j,i,var_min[j],var_max[j]))
    j=j+1


tframe = TrainingFrame(df_mc[df_mc.weight!=0],features+["process"])
tf_ = tframe.prepare(masses=masses,addmass=True,ifold=ifold)

X_sets=tf_[0:3]
y_sets=tf_[3:6]
w_sets=tf_[6:]
del tf_

signal = "c"
if uquark:
    signal = "u"
if uquark and cquark: signal="q"
    
for i in range(0,3):
    if signal!="q":
        issignal = (y_sets[i] > 0) & (X_sets[i].process.str.contains(signal))
        isbackground = (y_sets[i] < 0)
        y_sets[i] = y_sets[i][issignal | isbackground]
        w_sets[i] = w_sets[i][issignal | isbackground]
        X_sets[i] = X_sets[i][issignal | isbackground]
    print(y_sets[i].unique(),X_sets[i].process.unique())
    X_sets[i] = X_sets[i].drop(columns=["process"])
print(w_sets[0][y_sets[0]==-1].sum(),w_sets[0][y_sets[0]!=-1].sum())

for i in range(0,3):
    print(X_sets[i].shape)

del df_mc
msb = WeightScaler()
for i in range(0,2):
    msb.fit(X_sets[i],y_sets[i],w_sets[i])
    msb.transform(X_sets[i],y_sets[i],w_sets[i])
    print(w_sets[i][y_sets[i]==-1].sum(),w_sets[i][y_sets[i]!=-1].sum())

rnd = BackgroundRandomizer(verbose=True)
for i in range(0,2):
    print("Set",i,"\n")
    rnd.fit(X_sets[i],y_sets[i],w_sets[i])
    rnd.transform(X_sets[i],y_sets[i],w_sets[i])
    print("output",w_sets[i][y_sets[i]<0].sum(),w_sets[i][y_sets[i]>0].sum())

wss = WeightedStandardScaler( )
wss.fit(X_sets[0],y_sets[0],w_sets[0])
for i in range(0,2):
    wss.transform(X_sets[i],y_sets[i],w_sets[i])
wss.export(X_sets[0],y_sets[0],w_sets[0],outputdir,"c1ltrain_"+signal+"_"+str(ifold))
print(wss.mean_,wss.var_,wss.scale_)
print(np.average(X_sets[0],axis=0,weights=w_sets[0]))
print(variance(X_sets[0],weights=w_sets[0]))



for i in range(0,2):
    if optmass!="MP":
        X_sets[i] = X_sets[i].drop(columns=["X_mass"])
    y_sets[i] = ( y_sets[i] > 0)

for i in range(0,3):
    print(X_sets[i].shape)

modelNN=FeedForwardModel(configuration=structure,dropout=dropout,verbose=verbose, input_dim=X_sets[0].shape[1],learningr=learningrate)
resultNN=modelNN.train(X_sets[0], y_sets[0], w_sets[0],(X_sets[1], y_sets[1], w_sets[1]),batch_size = bsize, epochs=300,patience=5,callbacks=getCallbacks(modelNN))
print(resultNN)
arch_file=open(outputdir+'/architecture_'+signal+'_'+str(ifold)+'.h5','w') #save architecture for analysis framework
arch_file.write(modelNN.model.to_json())
arch_file.close()
modelNN.model.save_weights(outputdir+'/weights_'+signal+'_'+str(ifold)+'.h5') #Save weights for analysis framework
modelNN.plotTrainingValidation(outputdir+'/Validation_'+signal+'_'+str(ifold)) #in HpKerasUtils to save performance of each epoch
PlotAUCandScore(modelNN,X_sets[1],y_sets[1],w_sets[1],outputdir+'/Validation_'+signal+'_'+str(ifold)) #Validating the NN input
