import numpy as np
from scipy.stats import rv_discrete

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
            self.massvar = "X_mass"
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
            X[self.massvar]=y.abs()
            #labels=y.apply(lambda val: val if val!=self.backgroundclass else custm.rvs())
            if self.verbose and not w is None:
                print("the following is the difference between + and - mass")
                print((w*((y>0)-0.5)*2).groupby(y.abs()).sum())
                print("the following is the sum of weights")
                print(w.groupby(y).sum())

            return X, y, w
