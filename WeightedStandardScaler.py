import numpy as np
from joblib import dump

def variance(values, weights=None, axis=0):
    """ returns weighted (biased) variance
        values: array/series with values
        weights: array/series with weights (same dimension as values)
    """
    
    average = np.average(values, weights=weights, axis=axis)
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return variance

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
    
    def transform(self, X, y, w,verbose=False):
        """
        Performs standarization by centering and scaling features
        X: feature matrix
        y: labels (ignored)
        w: event weights (ignored)
        """
        
        X -= self.mean_
        X /= self.scale_
        
        nonzerovariance = np.where(self.scale_!=0)[0]
        if verbose: print("Non zero variance columns!",X.shape,nonzerovariance)
        return X.iloc[:,nonzerovariance],y,w
    
    def export(self, X, y, w, path, classlabel):
        """
        Saves scaler information as json for lwnn c++ package
        """
        dump(self,path+'/wss_'+classlabel+'.joblib')
        with open(path+"/Variables_"+classlabel+".json","w+") as outfile:
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