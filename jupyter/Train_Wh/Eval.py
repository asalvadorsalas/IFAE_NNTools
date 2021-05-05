import pandas as pd
import json
import os, sys
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
fpath = os.path.join(rcParams["datapath"], "/nfs/pic.es/user/s/salvador/arial.ttf")
fbipath = os.path.join(rcParams["datapath"], "/nfs/pic.es/user/s/salvador/ArialBoldItalic.ttf")

from sklearn.metrics import roc_auc_score
from joblib import dump, load

def ATLAS(ylims,perc):
    #For plots ets axis coordinates in units of %
    width=ylims[1]-ylims[0]
    return ylims[0]+width*perc

from IFAE_NNTools.TrainingFrame import *
from IFAE_NNTools.WeightedStandardScaler import *

inputdir = "./Plots"
user = "salvador"
pandainput = "pandas_Wh_v1_6jin4bin.feather"

masses = [250,300,350,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2500,3000]

ifold = 0

outputdir = inputdir + "/Eval"

print("Running eval for",inputdir,pandainput)

if not os.path.isdir(outputdir):
    print("Making output dir",outputdir)
    os.makedirs(outputdir, exist_ok=True)

if not os.path.isdir("/tmp/"+user):
    print ("Making tmp dir")
    os.mkdir("/tmp/"+user)

bookAUC = {}

labels = {'all': "Inclusive","6jin4bin": r'$\geq$6j$\geq$4b'}
regions= ['6jin4bin']

for region in regions:
    bookAUC[region]={}

df_mc = pd.read_feather("/tmp/"+user+"/"+pandainput) #2:53
print (df_mc.shape)
print (df_mc.columns.unique())

columns_suffix= ['pt','eta','phi','e']
for s in columns_suffix:
    df_mc['el_'+s]=pd.DataFrame(df_mc['el_'+s].tolist(), index= df_mc.index)
    df_mc['mu_'+s]=pd.DataFrame(df_mc['mu_'+s].tolist(), index= df_mc.index)
    df_mc.loc[np.isnan(df_mc['el_'+s])==False,'lepton_'+s]=df_mc[np.isnan(df_mc['el_'+s])==False]['el_'+s]
    df_mc.loc[np.isnan(df_mc['mu_'+s])==False,'lepton_'+s]=df_mc[np.isnan(df_mc['mu_'+s])==False]['mu_'+s]
#df_mc= df_mc.drop(columns=['el_pt','el_eta','el_phi','el_e','mu_pt','mu_eta','mu_phi','mu_e']) #no need to drop them, the trainingframe will not use them if not specified

features = ['MaxMVA_Response','mVH','H_mass','higgs_pt','wlep_pt','whad_m','whad_pt','mWT', 'Hpt_over_mVH', 'Wpt_over_mVH','DeltaPhi_HW','min_DeltaPhiJETMET', 'HT_jets', 'Centrality','met_met']

tframe = TrainingFrame(df_mc[df_mc.weight!=0],features+["region","process"])
tframe.signalprefix="Hp"
tframe.massvar="Hp_"
tframe.foldvar="eventNumber"
tf_ = tframe.prepare(masses="all",addmass=True,ifold=ifold)
X_eval,y_eval,w_eval = [tf_[1],tf_[4],tf_[7]]
del df_mc, tf_

wss = load(inputdir+'/wss_'+str(ifold)+'.joblib')

json_file = open(inputdir+'/architecture_'+str(ifold)+'.h5')
loaded_model_json = json_file.read()
json_file.close()

modelNN = model_from_json(loaded_model_json)
modelNN.load_weights(inputdir+'/weights_'+str(ifold)+'.h5')
    
reg_col = X_eval.region
proc_col = X_eval.process
X_eval = X_eval.drop(columns=["region","process"])
    
wss.transform(X_eval,y_eval,w_eval)
print("Shapes",X_eval.shape,y_eval.shape,w_eval.shape)

for mass in masses:
    X_=X_eval
    y_=y_eval
    w_=w_eval
    proc_ = proc_col
    reg_ = reg_col
    imassmask = ((y_==mass) | (y_<0.5))
    X_=X_[ imassmask ]
    y_=y_[ imassmask ]
    w_=w_[ imassmask ]    
    proc_ = proc_[ imassmask ]
    reg_ = reg_[ imassmask ]
    print ("mass",mass,X_.shape,y_.shape,w_.shape,X_[y_==mass]["Hp_mass"].unique()[0])
    X_["Hp_mass"]=X_[y_==mass]["Hp_mass"].unique()[0]
    #X_ = X_.drop(columns=["Hp_mass"])
    y_ = ( y_ > 0)
    y_pred_ = modelNN.predict(X_)
    print(X_.head(),y_[:10],y_pred_[:10])
    for region in regions:
        print(region,y_.shape,y_pred_.shape,w_.shape,reg_.shape,proc_.shape)
            
        if not region=="all":
            y,y_pred,w = [y_[reg_==region],y_pred_[reg_==region],w_[reg_==region]]
            proc = proc_[reg_==region]
        else:
            y,y_pred,w,proc = [y_,y_pred_,w_,proc_]
        print(region,y.shape,y_pred.shape,w.shape,reg_.shape,proc.shape)
        auc = roc_auc_score(y, y_pred, sample_weight=w)

        bookAUC[region][mass]=auc
        print(bookAUC[region][mass])
        sumwsig = w[y>0.5].sum()
        sumwbkg = w[y<0.5].sum()
            
        if sumwsig ==0: sumwsig = 1 
        if sumwbkg ==0: sumwbkg = 1
                
        w_sig = w[y>0.5]/sumwsig
        w_bkg = w[y<0.5]/sumwbkg
        print(w_bkg.sum(),w_sig.sum())
        bins = 50
        ymulti = 1.35
        yscale = 0.95

        plt.figure(figsize=(6.4,4.8),linewidth=0)
        plt.hist(y_pred[y>0.5],weights=w_sig,bins=bins,range=[0,1],density=False,fc=(1,0,0,0.125),ec="r",linewidth=1.5,label=r'H$^{+}$ '+str(mass)+" AUC:"+str(round(auc,3)),histtype='stepfilled') #Signal is everything with label y==1
        plt.hist(y_pred[y<0.5],weights=w_bkg,bins=bins,range=[0,1],density=False,fc=(0,0,1,0.125),ec="b",linewidth=1.5,label="Background",histtype='stepfilled')
        plt.xlabel("NN output",horizontalalignment='right',x=1,fontproperties=fm.FontProperties(fname=fpath,size=18))
        plt.ylabel("Entries",horizontalalignment='right',y=1,fontproperties=fm.FontProperties(fname=fpath,size=18))
        plt.xlim(0.,1.)
        plt.ylim(0.,plt.gca().get_ylim()[1]*ymulti)
        plt.gca().set_xticks(plt.gca().get_xticks().tolist())
        plt.gca().set_yticks(plt.gca().get_yticks().tolist())
        plt.gca().set_yticklabels([round(num,2) for num in plt.gca().get_yticks()], fontproperties=fm.FontProperties(fname=fpath,size=16))
        plt.gca().yaxis.set_minor_locator(tck.AutoMinorLocator())
        plt.gca().set_xticklabels([round(num,2) for num in plt.gca().get_xticks()], fontproperties=fm.FontProperties(fname=fpath,size=16))
        plt.gca().xaxis.set_minor_locator(tck.AutoMinorLocator())
        plt.gca().tick_params(length=10, width=0.5)
        plt.gca().tick_params(which="minor",length=5, width=0.5)
        plt.text(0.04,plt.gca().get_ylim()[1]*yscale,labels[region],va='top',ha='left',fontproperties=fm.FontProperties(fname=fpath,size=18))

        plt.legend(loc="best",prop=fm.FontProperties(fname=fpath,size=18),frameon=False)
        plt.savefig(outputdir+'/Score_'+str(ifold)+"_"+str(mass)+"_"+region+'.png',bbox_inches='tight')
        plt.show()
            
dump(bookAUC,outputdir+'/AUC_'+str(ifold)+'.sav')
