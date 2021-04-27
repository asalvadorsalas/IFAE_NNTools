import pandas as pd
import json
import os, sys
import pyarrow
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib.ticker as tck
from sklearn.metrics import roc_auc_score
from joblib import dump, load

import numpy as np
import pandas as pd
from sklearn.utils import _safe_indexing, check_random_state
from itertools import chain
from sklearn.utils import shuffle
from scipy.stats import rv_discrete

plt.style.use('classic')
rcParams['figure.facecolor'] = '1'
rcParams['patch.force_edgecolor'] = False
fpath = os.path.join(rcParams["datapath"], "/nfs/pic.es/user/s/salvador/arial.ttf")
fbipath = os.path.join(rcParams["datapath"], "/nfs/pic.es/user/s/salvador/ArialBoldItalic.ttf")

def ATLAS(ylims,perc):
    #gets axis coordinates in units of %
    width=ylims[1]-ylims[0]
    return ylims[0]+width*perc

from joblib import load, dump
inputdir = "./tXqML/Models/NewTrain3_"
user = "salvador"
pandainput = "pandas_tqXv2.feather"

reframelep_phi = True
reframelep_eta = True

masses = [125,120,140,150,160,20,30,40,50,60,70,80,90,100]

optmass = 20

if len(sys.argv)>1:
    if sys.argv[1]!="MP":
        optmass = int(sys.argv[1])
    else:
        optmass = sys.argv[1]

inputdir+=str(optmass)

outputdir = inputdir + "/Eval"

reframelep_phi = True
reframelep_eta = True

print("Running eval for",inputdir,pandainput)

if not os.path.isdir(outputdir):
    print("Making output dir",outputdir)
    os.makedirs(outputdir, exist_ok=True)

if not os.path.isdir("/tmp/"+user):
    print ("Making tmp dir")
    os.mkdir("/tmp/"+user)


def variance(values, weights=None, axis=0):
    """ returns weighted (biased) variance
        values: array/series with values
        weights: array/series with weights (same dimension as values)
    """
    
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return variance

class TrainingFrame:
    """Class that provides the tXq data for machine learning"""
    
    def __init__(self, pandasframe, feature_names, backgroundclass=-1):
        """Constructor to set data frame, features columns name and class label for background"""
        self.pandasframe = pandasframe
        self.backgroundclass = backgroundclass
        self.foldvar = "event_number"
        self.feature_names = feature_names
        self.random_state = 123456789
        self.mask = -1
    
    def get_pandasframe_mask(self, region, masses):
        """return a mask (true/false) for the specified options
           region: string name of the region
           masses: string "all" or list of masses to return
        """
        if region!="":
            regionseries = self.pandasframe.region==region
        else:
            regionseries = pd.Series([True]*self.pandasframe.shape[0],index=self.pandasframe.index)
        
        if masses=="all":
            return regionseries
        else:
            issignalseries = self.pandasframe.process.str.contains("X_"+str(masses[0]))
            for imass in masses[1:]:
                issignalseries = (issignalseries | self.pandasframe.process.str.contains("X_"+str(imass)) )
        
        isbackgroundseries =~ self.pandasframe.process.str.contains("X_")
            
        return regionseries & (issignalseries | isbackgroundseries)
                                    
    def get_features_classes_weights(self,region,masses,addmass,absoluteWeight):
        """returns features data frame, classes series and weights series
           region: string, region of the events to be returned
           masses: string "all" or list specifying the masses to return
           addmass: bool, if True the mass is added to the feature matrix (parameterised ML training)
           abosluteWeight: bool, if True the absolute value of weight will be returned (default)
        """
        
        self.mask = self.get_pandasframe_mask(region,masses) #filters by region and masses
        if addmass:
            features = self.pandasframe[self.mask].loc[:,self.feature_names+["X_mass"]].copy()
            classes = self.pandasframe[self.mask].process.apply(lambda proc:self.backgroundclass if not "X_" in proc else int(proc.split("X_")[1]))

        else:
            features = self.pandasframe[self.mask].loc[:,self.feature_names].copy() 
            classes = self.pandasframe[self.mask].process.apply(lambda proc:self.backgroundclass if not "X_" in proc else 1)

        
        weights = self.pandasframe[self.mask].weight
        if absoluteWeight:
            weights = abs(weights)
        
        features.reset_index(inplace = True, drop = True)
        weights.reset_index(inplace = True, drop = True)
        classes.reset_index(inplace = True, drop = True)
        return features, classes, weights
        
    def get_split_series(self,rows):
        """return a series of integers 0,1,2 which sample of training, testing and evaluation belongs to."""
        #splitdf = pd.DataFrame([True]*rows)
        #splitdf = splitdf.apply(lambda x: 0 if x.name%10 in [0,1] else 1 if x.name%10 in [2,3] else 2 if x.name%10 in [4,5] else 3 if x.name%10 in [6,7] else 4, axis=1)
        #splitdf = splitdf.apply(lambda x: x.name%2, axis=1)
        #return #pd.Series(splitdf)
        #pd.Series([i%2 for i in range(rows)])
        return pd.Series([i%5 for i in range(rows)])
    
    
    def prepare(self, masses="all", region="", addmass=False, absoluteWeight=True,ifold=0):
        """returns feature matrices, class labels and weights for training, testing and evaluation
           region: string, region of the events to be returned
           masses: string "all" or list specifying the masses filter
           addmass: bool if True the mass column is added to the feature matrix
           absoluteWeight: bool, if true the absolute value of the weights will be returned
        """
        features, classes, weights = self.get_features_classes_weights(region,masses,addmass,absoluteWeight)
        split_series = self.pandasframe[self.mask][self.foldvar]%5  #self.get_split_series(classes.shape[0])
        
        trainset = np.where( (split_series==ifold%5) | (split_series==(ifold+1)%5) | (split_series==(ifold+2)%5) )[0]
        valset   = np.where( split_series==(ifold+3)%5 )[0]
        #trainset = np.where( split_series==ifold )[0]
        #valset   = np.where( split_series==ifold+1 )[0]
        testset  = np.where( split_series==(ifold+4)%5 )[0]
        
        rng = check_random_state(self.random_state)
        rng.shuffle(trainset)
        rng.shuffle(valset)
        rng.shuffle(testset)
        
        return list(chain.from_iterable((_safe_indexing(a,trainset),
                                         _safe_indexing(a,valset),
                                         _safe_indexing(a,testset)) for a in [features,classes,weights]))

class WeightScaler():
    """Class that makes the integral of the signal/bkg weights be equal, for several categories the distribution as a function of the class variable is flattened"""
    def __init__(self, backgroundclass=-1,norm=0.5):
        """
        backgroundlcass: label for bgk
        norm: value for the integral of weights
        """
        self.backgroundclass = backgroundclass
        self.norm = norm
        self.scale_={}
        
    def fit(self,X,y,w):
        """
        learns the sum of weights for all classes and calculates a scale factor for each class so that the sum of weights for bkg is 'norm' and the sum of weights for signal is flattened with tota integral 'norm' 
        X: feature matrix, to keep structure (ignored)
        y: series of class labels
        w: Series of sample weights
        """
        classes=sorted(np.unique(y))
        classes.remove(self.backgroundclass)
        
        differences={}
        if len(classes)>1: #more than 1 signal
            #set the differences between signal points
            differences={classes[i]:(classes[i+1]-classes[i-1])/2 for i in range(1,len(classes)-1) if classes[i]>0}
            differences[classes[0]]=classes[1]-classes[0]
            differences[classes[-1]]=classes[-1]-classes[-2]
            diffsum=sum(differences.values())
            print (differences, "->", diffsum)
        else:
            differences[classes[0]]=1
            diffsum=1
        
        for classlabel in classes:
            sumweight=w[y==classlabel].sum()
            self.scale_[classlabel]=differences[classlabel]/(2*sumweight*diffsum)
        sumweight=w[y==self.backgroundclass].sum()
        self.scale_[self.backgroundclass]=self.norm/sumweight
        return
        
    def transform(self,X,y,w,copy=None):
        """
        Transforms the sum of weights for all classes applying the corresponding scale of each label
        """
        
        for classlabel in self.scale_:
            w[y==classlabel]*=self.scale_[classlabel]
        return X,y,w          


class BackgroundRandomizer():
        """
        Class that assigns random signal mass hypotheses to bkg training data with the PDF learned from signal for parameterised training
        """
        
        def __init__(self,backgroundclass=-1,verbose=False):
            """
            backgroundclass: label for background
            verbose: print debug information if True
            """
            
            self.backgroundclass = backgroundclass
            self.verbose = verbose
            self.randomseed = 123456789
            self.xk = []
            self.pk = []
        
        def fit(self, X, y, w):
            """
            determines the mass PDF for signal events
            X: features
            y: parameter for signal, background class for bkg
            w: weights for events (ignored)
            """
            self.xk = []
            self.pk = []
            signalsum = np.sum(w[y!=self.backgroundclass])
            for name, group in w.groupby(y):
                print("name",name, "group",group,"sum",group.sum())
                if name!=self.backgroundclass:
                    self.xk.append((-1)*name)
                    self.pk.append(group.sum()/signalsum)
            if self.verbose:
                print("Signal PDF:",self.xk,self.pk)
        
        def transform(self, X, y, w):
            """
            randomly assigns signal labels according to fitted pdf
            X: features
            y: parameter for signal, background class for bkg
            w: weights for events. Bkg weights are going to be set to have same sum as the weights of the signal label assigned
            """
            
            np.random.seed(seed=self.randomseed+len(y)) #not to have the same random seed for test and train
            custm=rv_discrete(values=(self.xk,self.pk))
            y.loc[y==self.backgroundclass]=custm.rvs(size=len(y[y==self.backgroundclass].index))
            X.X_mass=y.abs()
            #labels=y.apply(lambda val: val if val!=self.backgroundclass else custm.rvs())
            #X.X_mass=labels.abs()
            if self.verbose and not w is None:
                print("the following is the difference between + and - mass")
                print((w*((y>0)-0.5)*2).groupby(y.abs()).sum())
                print("the following is the sum of weights")
                print(w.groupby(y).sum())

            return X, y, w

class WeightedStandardScaler():
    """Class which transforms all features to have average 0 and variance 1, same as sklearn StandardScaler but taking weights into account"""
    
    def __init__(self):
        self.scale_ = None
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y, w):
        """
        Compute the mean and std to be used for scaling
        X: feature matrix
        y: labels (ignored)
        w: event weights
        """
            
        self.mean_ = np.average(X,axis=0,weights=w)
        self.var_ = variance(X,weights=w)
        self.scale_ = np.sqrt(self.var_)
    
    def transform(self, X, y, w):
        """
        Performs standarization by centering and scaling features
        X: feature matrix
        y: labels (ignored)
        w: event weights (ignored)
        """
        
        X -= self.mean_
        print(X)
        X /= self.scale_
        print(X)
        
        nonzerovariance = np.where(self.scale_!=0)[0]
        print("Non zero variance columns!",X.shape,nonzerovariance)
        return X.iloc[:,nonzerovariance],y,w
    
    def export(self, X, y, w, path, classlabel):
        """
        Saves scaler information as json for lwnn c++ package
        """
        dump(self,outputdir+'/wss_'+classlabel+'.joblib')
        with open(path,"w+") as outfile:
            print("Saving info in",path)
            outfile.write('{\n')
            outfile.write('  "inputs": [\n')
            for feat,mean,scale in zip(X.columns,self.mean_,self.scale_):
                outfile.write('    {\n')
                outfile.write('      "name": "'+feat+'" ,\n')
                outfile.write('      "offset": '+str(-mean)+" ,\n")
                outfile.write('       "scale":'+str(1./scale)+" \n")
                #print (feat,"\t",-mean,1./scale)
                if feat==X.columns[-1]: outfile.write('    }\n')
                else: outfile.write('    },\n')
            outfile.write('  ],\n')
            outfile.write('  "class_labels": ["'+classlabel+'"]\n')
            outfile.write('}')    
        print (X.shape,X.mean(), X.var())

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
sets = ["u","c"]
sets = ["q"]
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
    print("shape w/o 0",df_mc[df_mc.weight!=0].shape)
    tframe = TrainingFrame(df_mc[df_mc.weight!=0],features+["region","process","event_number"])
    Xf_eval = []
    yf_eval = []
    wf_eval = []
    regf_col = []
    procf_col = []
    eventf_col = []
    models_all = []
    for ifold in [0,1,2,3,4]:
        tf_ = tframe.prepare(masses="all",addmass=True,ifold=ifold)
        X_eval,y_eval,w_eval = [tf_[1],tf_[4],tf_[7]]
        #X_eval,y_eval,w_eval = [tf_[2],tf_[5],tf_[8]]
        del tf_

        wss = load(inputdir+'/wss_c1ltrain_'+training+'_'+str(ifold)+'.joblib')

        json_file = open(inputdir+'/architecture_'+training+'_'+str(ifold)+'.h5')
        loaded_model_json = json_file.read()
        json_file.close()

        modelNN = model_from_json(loaded_model_json)
        modelNN.load_weights(inputdir+'/weights_'+training+'_'+str(ifold)+'.h5')
        models_all.append(modelNN)
        reg_col = X_eval.region
        proc_col = X_eval.process
        event_col = X_eval.event_number
        X_eval = X_eval.drop(columns=["region","process","event_number"])
    
        wss.transform(X_eval,y_eval,w_eval)
        print("Shapes",X_eval.shape,y_eval.shape,w_eval.shape)
        
        if ifold == 0:
            Xf_eval = X_eval
            yf_eval = y_eval
            wf_eval = w_eval
            regf_col = reg_col
            procf_col = proc_col
            eventf_col = event_col
        else:
            Xf_eval = Xf_eval.append(X_eval)
            yf_eval = yf_eval.append(y_eval)
            wf_eval = wf_eval.append(w_eval)
            regf_col = regf_col.append(reg_col)
            procf_col = procf_col.append(proc_col)
            eventf_col = eventf_col.append(event_col)
        print(Xf_eval.shape,yf_eval.shape,wf_eval.shape,regf_col.shape,procf_col.shape,eventf_col.shape,models_all)
        print("xcheck",(eventf_col%5).unique())
    print(Xf_eval.shape,yf_eval.shape,wf_eval.shape,regf_col.shape,procf_col.shape,eventf_col.shape,models_all,type(yf_eval))
            
    masses = [125,120,140,150,160,20,30,40,50,60,70,80,90,100]
    for mass in masses:
        X_=Xf_eval
        y_=yf_eval
        w_=wf_eval
        proc_ = procf_col
        reg_ = regf_col
        evnt_ = eventf_col
        imassmask = ((y_==mass) | (y_<0.5))
        X_=X_[ imassmask ]
        y_=y_[ imassmask ]
        w_=w_[ imassmask ]    
        proc_ = proc_[ imassmask ]
        reg_ = reg_[ imassmask ]
        evnt_ = evnt_[ imassmask ]
        print ("mass",mass,X_.shape,y_.shape,w_.shape,X_[y_==mass]["X_mass"].unique()[0])
        X_["X_mass"]=X_[y_==mass]["X_mass"].unique()[0]
        #X_ = X_.drop(columns=["X_mass"])
        y_ = ( y_ > 0)
        #for ifold in [0,1,2,3,4]:
        y_pred_ = y_.copy()
        for ifold in [0,1,2,3,4]:
            print("evaluating",ifold,y_pred_)
            y_pred_.loc[(evnt_%5)==(ifold+3)%5] = models_all[ifold].predict(X_[(evnt_%5)==(ifold+3)%5]).ravel()
            print("testing",y_pred_,y_pred_[(evnt_%5)==4][:10],y_pred_[(evnt_%5)==2][:10])

        print("testing",y_pred_[(evnt_%5)==4][:10],y_pred_[(evnt_%5)==0][:10])
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
            plt.savefig(outputdir+'/tr_'+training+'/Score_'+str(mass)+"_"+region+'.png',bbox_inches='tight')
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
            plt.savefig(outputdir+'/tr_'+training+'/Sepscore_'+str(mass)+"_"+region+'.png',bbox_inches='tight')
            plt.show()
            
    dump(bookAUC[training],outputdir+'/AUC_'+training+'.sav')
    dump(bookSep[training],outputdir+'/Sep_'+training+'.sav')
