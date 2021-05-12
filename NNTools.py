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
            self.callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
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
