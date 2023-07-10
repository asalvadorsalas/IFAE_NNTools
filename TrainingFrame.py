"""Module with class that provides the data for machine learning functions"""

import numpy as np
import pandas as pd
from sklearn.utils import _safe_indexing, check_random_state
from itertools import chain
from sklearn.utils import shuffle

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
        self.signalprefix = "X_"
        self.massvar = "X_"
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
        else: #This wont work for Hpmass style of process
            issignalseries = self.pandasframe.process.str.contains(self.signalprefix+str(masses[0]))
            for imass in masses[1:]:
                issignalseries = (issignalseries | self.pandasframe.process.str.contains(self.signalprefix+str(imass)) )
        
        isbackgroundseries =~ self.pandasframe.process.str.contains(self.signalprefix)
            
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
            features = self.pandasframe[self.mask].loc[:,self.feature_names+[self.massvar+"mass"]].copy()
            classes = self.pandasframe[self.mask].process.apply(lambda proc:self.backgroundclass if not self.signalprefix in proc else int(proc.split(self.signalprefix)[1]))

        else:
            features = self.pandasframe[self.mask].loc[:,self.feature_names].copy() 
            classes = self.pandasframe[self.mask].process.apply(lambda proc:self.backgroundclass if not self.signalprefix in proc else 1)

        
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
        
        #trainset = np.where( (split_series==ifold%5) | (split_series==(ifold+1)%5) | (split_series==(ifold+2)%5) )[0]
        #valset   = np.where( (split_series==(ifold+3)%5) | (split_series==(ifold+4)%5) )[0]
        #testset  = np.where( split_series==(ifold+99)%5 )[0]
        
        trainset = np.where( (split_series==ifold%5) | (split_series==(ifold+1)%5) | (split_series==(ifold+2)%5) )[0]
        valset   = np.where( split_series==(ifold+3)%5 )[0]
        testset  = np.where( split_series==(ifold+4)%5 )[0]
        
        #trainset = np.where( split_series==ifold%2 )[0]
        #valset   = np.where( split_series==(ifold+1)%2 )[0]
        #testset  = np.where( split_series==(ifold+99) )[0]        
        
        rng = check_random_state(self.random_state)
        rng.shuffle(trainset)
        rng.shuffle(valset)
        rng.shuffle(testset)
        
        return list(chain.from_iterable((_safe_indexing(a,trainset),
                                         _safe_indexing(a,valset),
                                         _safe_indexing(a,testset)) for a in [features,classes,weights]))
