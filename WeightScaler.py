"""Module with weight transformer functions for machine learning algorithms"""

import numpy as np

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
        learns the sum of weights for all classes and calculates a scale factor for each class so that the sum of weights for bkg is 'norm' and the sum of weights for signal is flattened with total integral 'norm' 
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