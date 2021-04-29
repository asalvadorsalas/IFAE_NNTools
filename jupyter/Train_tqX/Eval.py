import pandas as pd
import json
import os, sys
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

inputdir = "./tXqML/Models/Test_"
user = "salvador"
pandainput = "pandas_tqXv2.feather"
sets = ["q","u","c"]

reframelep_phi = True
reframelep_eta = True

masses = [125,120,140,150,160,20,30,40,50,60,70,80,90,100]

ifold = 0
optmass = 20

if len(sys.argv)>1:
    if sys.argv[1]!="MP":
        optmass = int(sys.argv[1])
    else:
        optmass = sys.argv[1]

if len(sys.argv)>2:
    ifold = int(sys.argv[2])
        
inputdir+=str(optmass)

outputdir = inputdir + "/Eval"

print("Running eval for",inputdir,pandainput)

for i in ["fold0","fold1","fold2","fold3","fold4"]:
    for j in ["tr_u","tr_c","tr_q"]:
        if not os.path.isdir(outputdir+"/"+i+"/"+j):
            print("Making output dir",outputdir+"/"+i+"/"+j)
            os.makedirs(outputdir+"/"+i+"/"+j, exist_ok=True)

if not os.path.isdir("/tmp/"+user):
    print ("Making tmp dir")
    os.mkdir("/tmp/"+user)

def getseparation(nu,n,bins):
    print("getsepa")
    separation = 0
    binsize = float(bins[-1]-bins[0])/float(len(n))
    print(binsize)
    S = sum(nu)*binsize
    B = sum(n)*binsize
    print(S,B)
    for i in range(len(n)):
        s = nu[i]/S
        b = n[i]/B
        if (s+b)>0:
            separation+=(s-b)*(s-b)/(s+b)
        print(i, s, b, separation)
    separation*=binsize*0.5
    return separation

bookAUC = {}
bookSep = {}

labels = {'all': "Inclusive","c1l4jex3bex": "4j 3b","c1l4jex4bin": r'4j 4b', "c1l5jex3bex":"5j 3b", "c1l5jex4bin": "5j "+r'$\geq$'+"4b",
                  "c1l6jin3bex": r'$\geq$6j'+" 3b","c1l6jin4bin": r'$\geq$6j$\geq$4b'}
regions= ['all'             ,'c1l4jex3bex'         ,'c1l4jex4bin'               , 'c1l5jex3bex'        , 'c1l5jex4bin',
                  'c1l6jin3bex'                   ,'c1l6jin4bin']

for training in sets:
    bookAUC[training]={}
    bookSep[training]={}
    for region in regions:
        bookAUC[training][region]={}
        bookSep[training][region]={}
    #if training=="c": continue
    df_mc = pd.read_feather("/tmp/"+user+"/"+pandainput) #2:53
    df_mc = df_mc[~((df_mc.X_mass==160)&(df_mc.nomWeight_weight_mc > 700))]
    df_mc["weight"]*=139000.0
    print (df_mc.shape)
    print (df_mc.columns.unique())

    #df_mc["X_mass"]=125

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

    tframe = TrainingFrame(df_mc[df_mc.weight!=0],features+["region","process"])
    tf_ = tframe.prepare(masses="all",addmass=True,ifold=ifold)
    X_eval,y_eval,w_eval = [tf_[1],tf_[4],tf_[7]]
    #X_eval,y_eval,w_eval = [tf_[2],tf_[5],tf_[8]]
    del df_mc, tf_

    wss = load(inputdir+'/wss_c1ltrain_'+training+'_'+str(ifold)+'.joblib')

    json_file = open(inputdir+'/architecture_'+training+'_'+str(ifold)+'.h5')
    loaded_model_json = json_file.read()
    json_file.close()

    modelNN = model_from_json(loaded_model_json)
    modelNN.load_weights(inputdir+'/weights_'+training+'_'+str(ifold)+'.h5')
    
    reg_col = X_eval.region
    proc_col = X_eval.process
    X_eval = X_eval.drop(columns=["region","process"])
    
    wss.transform(X_eval,y_eval,w_eval)
    print("Shapes",X_eval.shape,y_eval.shape,w_eval.shape)

    masses = [125,120,140,150,160,20,30,40,50,60,70,80,90,100]
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
        print ("mass",mass,X_.shape,y_.shape,w_.shape,X_[y_==mass]["X_mass"].unique()[0])
        X_["X_mass"]=X_[y_==mass]["X_mass"].unique()[0]
        #X_ = X_.drop(columns=["X_mass"])
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
            
            umask = ((proc=="uX_"+str(mass)) | (proc=="ubarX_"+str(mass)))
            cmask = ((proc=="cX_"+str(mass)) | (proc=="cbarX_"+str(mass)))
            
            print(region,auc,umask.shape,cmask.shape)
            
            aucu = -1
            aucc = -1
            if (y[y<0.5].shape[0]!=0):
                if y[umask].shape[0]!=0:
                    aucu = roc_auc_score(y[(y<0.5) | (umask)], y_pred[(y<0.5) | (umask)], sample_weight=w[(y<0.5) | (umask)])
                if y[cmask].shape[0]!=0:
                    aucc = roc_auc_score(y[(y<0.5) | (cmask)], y_pred[(y<0.5) | (cmask)], sample_weight=w[(y<0.5) | (cmask)])

            bookAUC[training][region][mass]=[aucu,aucc]
            print(bookAUC[training][region][mass])
            #sumwsig = w[y>0.5].sum()
            sumwsigu = w[umask].sum()
            sumwsigc = w[cmask].sum()
            sumwbkg = w[y<0.5].sum()
            #w_sig = w[y>0.5]/sumwsig
            
            if sumwsigu ==0: sumwsigu = 1 
            if sumwsigc ==0: sumwsigc = 1
            if sumwbkg ==0: sumwbkg = 1
                
            w_sigu = w[umask]/sumwsigu
            w_sigc = w[cmask]/sumwsigc
            w_bkg = w[y<0.5]/sumwbkg
            print(w_bkg.sum(),w_sigu.sum(),w_sigc.sum())
            bins = 50
            ymulti = 1.35
            yscale = 0.95

            plt.figure(figsize=(6.4,4.8),linewidth=0)
            #plt.hist(y_pred[y>0.5],weights=w_sig,bins=bins,range=[0,1],density=False,fc=(1,0,0,0.25),ec="r",linewidth=1.5,label="Signal",histtype='stepfilled') #Signal is everything with label y==1
            plt.hist(y_pred[umask],weights=w_sigu,bins=bins,range=[0,1],density=False,fc=(1,0,0,0.125),ec="r",linewidth=1.5,label="uX "+str(mass)+" AUC:"+str(round(aucu,3)),histtype='stepfilled') #Signal is everything with label y==1
            plt.hist(y_pred[cmask],weights=w_sigc,bins=bins,range=[0,1],density=False,fc=(0,1,0,0.125),ec="g",linewidth=1.5,label="cX "+str(mass)+" AUC:"+str(round(aucc,3)),histtype='stepfilled') #Signal is everything with label y==1
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
            #plt.text(0.04,plt.gca().get_ylim()[1]*yscale,"ATLAS",va='top',ha='left',fontproperties=fm.FontProperties(fname=fbipath,size=20))
            #plt.text(0.04,plt.gca().get_ylim()[1]*yscale,"               Simulation\nInternal",va='top',ha='left',fontproperties=fm.FontProperties(fname=fpath,size=18))
            plt.text(0.04,plt.gca().get_ylim()[1]*yscale,labels[region],va='top',ha='left',fontproperties=fm.FontProperties(fname=fpath,size=18))
            #plt.text(0.04,plt.gca().get_ylim()[1]*(yscale-0.025),"\nAUC:"+str(round(auc,3)),va='top',ha='left',fontproperties=fm.FontProperties(fname=fpath,size=18))

            plt.legend(loc="best",prop=fm.FontProperties(fname=fpath,size=18),frameon=False)
            plt.savefig(outputdir+'/fold'+str(ifold)+'/tr_'+training+'/Score_'+str(mass)+"_"+region+'.png',bbox_inches='tight')
            plt.show()
            
            plt.figure(figsize=(6.4,4.8),linewidth=0)
            (nu, binsu, patchesu) = plt.hist(y_pred[umask],weights=w_sigu,bins=10,range=[0,1],density=False,fc=(1,0,0,0.125),ec="r",linewidth=1.5,label="uX "+str(mass),histtype='stepfilled') #Signal is everything with label y==1
            (nc, binsc, patchesc) = plt.hist(y_pred[cmask],weights=w_sigc,bins=10,range=[0,1],density=False,fc=(0,1,0,0.125),ec="g",linewidth=1.5,label="cX "+str(mass),histtype='stepfilled') #Signal is everything with label y==1
            (n, bins, patches) = plt.hist(y_pred[y<0.5],weights=w_bkg,bins=10,range=[0,1],density=False,fc=(0,0,1,0.125),ec="b",linewidth=1.5,label="Background",histtype='stepfilled')
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
            bookSep[training][region][mass]=[100*getseparation(nu,n,bins),100*getseparation(nc,n,bins)]
            print(bookSep[training][region][mass])
            plt.legend(["uX "+str(mass)+" S:"+str(round(bookSep[training][region][mass][0],1))+"%","cX "+str(mass)+" S:"+str(round(bookSep[training][region][mass][1],1))+"%","Background"],loc="best",prop=fm.FontProperties(fname=fpath,size=18),frameon=False)
            plt.savefig(outputdir+'/fold'+str(ifold)+'/tr_'+training+'/Sepscore_'+str(mass)+"_"+region+'.png',bbox_inches='tight')
            plt.show()
            
    dump(bookAUC[training],outputdir+'/fold'+str(ifold)+'/AUC_'+training+'.sav')
    dump(bookSep[training],outputdir+'/fold'+str(ifold)+'/Sep_'+training+'.sav')