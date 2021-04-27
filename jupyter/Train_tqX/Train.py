import pandas as pd
import json
import pyarrow
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
import matplotlib.ticker as tck
plt.style.use('classic')
rcParams['figure.facecolor'] = '1'
rcParams['patch.force_edgecolor'] = False

import os,sys
from joblib import load, dump
import tensorflow as tf

outputdir = "./tXqML/Models/NewTrain3_"
user = "salvador"

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

pandainput = "pandas_tqXv2.feather"

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
import numpy as np
import pandas as pd
from sklearn.utils import _safe_indexing, check_random_state
from itertools import chain
from sklearn.utils import shuffle
from scipy.stats import rv_discrete

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
wss.export(X_sets[0],y_sets[0],w_sets[0],outputdir+"/Variables_"+signal+"_"+str(ifold)+".json","c1ltrain_"+signal+"_"+str(ifold))
print(wss.mean_,wss.var_,wss.scale_)
print(np.average(X_sets[0],axis=0,weights=w_sets[0]))
print(variance(X_sets[0],weights=w_sets[0]))

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt

def getCallbacks(model):
    """ Standard callbacks for Keras Early stopping and checkpoint"""
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        #ModelCheckpoint(filepath='model_nn_'+str(model.configuration)+"_dropout"+str(model.dropout)+"_l2threshold"+str(model.l2threshold)+".hdf5",
        #               monitor='val_loss',
        #                save_best_only=True)  
    ]

def PlotAUCandScore(model,X,y,w,path=""):
    """Function to plot AUC and Score given a HplusNNmodel, features, labels, weights and path to save the plots"""
    y_pred = model.model.predict(X).ravel()
    print(y.unique())
    roc_auc = roc_auc_score(y,y_pred,sample_weight=w)
    fpr, tpr, thresholds = roc_curve(y,y_pred,sample_weight=w)
    plt.figure(figsize=(6.4,4.8),linewidth=0)
    plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',horizontalalignment='right',x=1,fontsize=14)
    plt.ylabel('True Positive Rate',horizontalalignment='right',y=1,fontsize=14)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc="lower right",fontsize=14,frameon=False)
    plt.grid()
    plt.savefig(path+'_AUC.png',bbox_inches='tight')
    plt.show()

    sumwsig = w[y>0.5].sum()
    sumwbkg = w[y<0.5].sum()
    w_sig = w[y>0.5]/sumwsig
    w_bkg = w[y<0.5]/sumwbkg
    print(w_sig.sum(),w_bkg.sum())
    bins = 50
    plt.figure(figsize=(6.4,4.8),linewidth=0)
    plt.hist(y_pred[y>0.5],weights=w_sig,alpha=0.5,color='r',bins=bins,range=[0,1],density=False,label="Signal") #Signal is everything with label y==1
    plt.hist(y_pred[y<0.5],weights=w_bkg,alpha=0.5,color='b',bins=bins,range=[0,1],density=False,label="Background")
    plt.xlabel("NN score",horizontalalignment='right',x=1,fontsize=14)
    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in plt.gca().yaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    plt.legend(loc="best",fontsize=14,frameon=False)
    plt.savefig(path+'_Score.png',bbox_inches='tight')
    plt.show()
    return 1.-roc_auc

class FeedForwardModel():
    """ A simple feed forward NN based on Keras"""

    def __init__(self, configuration, l2threshold=None, dropout=None, input_dim=15, verbose=True, activation='relu',learningr=0.00428095):
        """ constructor
        configuration: list of the number of nodes per layer, each item is a layer
        l2threshold: if not None a L2 weight regularizer with threshold <l2threshold> is added to each leayer
        dropout: if not None a dropout fraction of <dropout> is added after each internal layer
        input_dim: size of the training input data
        verbose: if true the model summary is printed
        """
        
        self.callbacks = []
        self.verbose=verbose
        self.configuration=configuration
        self.dropout=dropout
        self.l2threshold=l2threshold
        self.model = Sequential()
        for i,layer in enumerate(configuration):
            if i==0:
                if l2threshold==None:
                    self.model.add(Dense(layer, input_dim=input_dim, activation=activation))    
                else:
                    self.model.add(Dense(layer, input_dim=input_dim, activation=activation, kernel_regularizer=regularizers.l2(l2threshold)))    
            else:
                if l2threshold==None:
                    self.model.add(Dense(layer, activation=activation))
                else:
                    self.model.add(Dense(layer, activation=activation, kernel_regularizer=regularizers.l2(l2threshold)))
            if dropout!=None:
                self.model.add(Dropout(rate=dropout))
        #final layer is a sigmoid for classification
        self.model.add(Dense(1, activation='sigmoid'))
        #model.add(Dense(5, activation='relu'))

        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learningr))
        self.model.summary()

    def train(self, X_train, y_train, w_train , testData, epochs=100, patience=15, callbacks=None, batch_size=50):
        """ train the Keras model with Early stopping, will return test and training ROC AUC
        trainData: tuple of (X_train, y_train, w_train)
        trainData: tuple of (X_test, y_test, w_test)
        epochs: maximum number of epochs for training
        patience: patience for Early stopping based on validation loss
        callbacks: 
        """

        if callbacks is None:
            self.callbacks.append(EarlyStopping(monitor='val_loss', 
                                                patience=patience))
            self.callbacks.append(ModelCheckpoint(filepath='model_nn_'+str(self.configuration)+"_dropout"+str(self.dropout)+"_l2threshold"+str(self.l2threshold)+".hdf5", 
                                                  monitor='val_loss',
                                                  save_best_only=True))
            self.callbacks.append(RocCallback(training_data=trainData,validation_data=testData))
        else:
            self.callbacks=callbacks

        if(self.verbose): self.history=self.model.fit(X_train,y_train, sample_weight=w_train,
                                    batch_size=batch_size, epochs=epochs, callbacks=self.callbacks,
                                    validation_data=testData,verbose=1)
        else: self.history=self.model.fit(X_train,y_train, sample_weight=w_train,
                                    batch_size=batch_size, epochs=epochs, callbacks=self.callbacks,
                                    validation_data=testData,verbose=2)

        #self.model.load_weights("model_nn_"+str(self.configuration)+"_dropout"+str(self.dropout)+"_l2threshold"+str(self.l2threshold)+".hdf5")
        y_pred_test=self.model.predict(testData[0]).ravel()
        y_pred_train=self.model.predict(X_train).ravel()
        roc_test =roc_auc_score(testData[1],  y_pred_test,  sample_weight=testData[2])
        roc_train=roc_auc_score(y_train, y_pred_train, sample_weight=w_train)
        #print(self.configuration, roc_test, roc_train)
        
        return roc_test, roc_train
    
    def plotTrainingValidation(self,path=""):
        """draws plots for loss, binary accuracy and ROC AUC"""

        loss_values=self.history.history['loss']
        val_loss_values=self.history.history['val_loss']
        #acc_values=self.history.history['binary_accuracy']
        #val_acc_values=self.history.history['val_binary_accuracy']

        rocauc_values=None
        val_rocauc_values=None
        bestepoch=None
        for cb in self.callbacks:
            if hasattr(cb, 'roc') and hasattr(cb, 'roc_val'):
                rocauc_values=cb.roc
                val_rocauc_values=cb.roc_val
            if hasattr(cb, 'stopped_epoch') and hasattr(cb, 'patience'):
                bestepoch=cb.stopped_epoch-cb.patience+1
  
        epochs=range(1,len(loss_values)+1)
        plt.figure()
        plt.plot(epochs, loss_values, "bo",label="Training loss")
        plt.plot(epochs, val_loss_values, "b",label="Validation loss")
        plt.legend(loc=0)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        if not bestepoch is None:
            plt.axvline(x=bestepoch)
        if path!="":
            plt.savefig(path+'_loss.png')
        plt.show()
            
        #ax=plt.figure()
        #plt.plot(epochs, acc_values, "bo",label="Training acc")
        #plt.plot(epochs, val_acc_values, "b",label="Validation acc")
        #plt.legend(loc=0)
        #plt.xlabel("Epochs")
        #plt.ylabel("Accuracy")
        #if not bestepoch is None:
        #    plt.axvline(x=bestepoch)
        #if path!="":
        #    plt.savefig(path+'_acc.png')
        #plt.show()
        
        if not rocauc_values is None:
            ax=plt.figure()
            plt.plot(epochs, rocauc_values, "bo",label="Training ROC AUC")
            plt.plot(epochs, val_rocauc_values, "b",label="Validation ROC AUC")
            plt.legend(loc=0)
            plt.xlabel("Epochs")
            plt.ylabel("ROC AUC")
            if not bestepoch is None:
                plt.axvline(x=bestepoch)
            plt.show()

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
