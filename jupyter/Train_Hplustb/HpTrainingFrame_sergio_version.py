"""Module with class that provides the H+ data for machine learning functions"""

import numpy as np
import HpMLUtils
from sklearn.model_selection import train_test_split
import pandas as pd

class HpTrainingFrame:
    """Class that provides the H+ data for machine learning functions"""
    
    def __init__(self, pandasframe, feature_names,  backgroundclass=-1):
        """Constructor which sets the pandas data frame and the class label for background"""

        self.pandasframe=pandasframe.copy()
        self.backgroundclass=backgroundclass
        self.feature_names=feature_names
        self.random_state=123456789

    def load_vec_features(self, conserveorder=True):
        """If the data frame contains lists in any of its columns, and if one of the components of this list is among the features names,returns a new data frame with the list column deleted and a new column added with the component of the list desired.
	    
	conserveorder: if true the new column will be in the same position as the corresponding list column
	"""

        vec_features = {}
        df_columns = list(self.pandasframe.columns)
	    
        for feature in self.feature_names:
            if feature not in df_columns:
                if feature[:-1] not in vec_features.keys():
                    vec_features[feature[:-1]]=[feature]
                else:
                    vec_features[feature[:-1]].append(feature)


        for vec_feature_name in vec_features.keys():
            components = []
            for i in range(len(self.pandasframe[vec_feature_name][0])):
                components.append(vec_feature_name+str(i+1))

            self.pandasframe[components] = pd.DataFrame(self.pandasframe[vec_feature_name].tolist(), index=self.pandasframe.index)
            index = df_columns.index(vec_feature_name)
            removed = df_columns.pop(index)

            i=0
            for name in vec_features[vec_feature_name]:
                df_columns.insert(index+i,name)
                i+=1

        self.pandasframe = self.pandasframe.drop(columns=vec_features.keys())
        if conserveorder==True: self.pandasframe = self.pandasframe[df_columns]

        return self.pandasframe

    def get_pandasframe_mask(self, region, hpmass, invertsignal=False):
        """return as mask (true/false) of which events are to be included in the different samples (e.g. removes unwanted H+ masses)
           region: string name of the region
           hpmass: string "multi" or integer specifying the H+ mass
           invertsignal: if true all signal points except for hpmass are selected
        """
        
        if region!=None:
            regionseries=self.pandasframe.region==region
        else:
            regionseries=pd.Series([True]*self.pandasframe.shape[0], index=self.pandasframe.index) 
            
        if hpmass=="multi":
            return regionseries

        if invertsignal:
            issignalseries=self.pandasframe.process!="Hp"+str(hpmass)
        else:
            issignalseries=self.pandasframe.process=="Hp"+str(hpmass)
        isbackgroundseries=~self.pandasframe.process.str.contains("Hp")
        return regionseries & (issignalseries | isbackgroundseries)

    def get_features_classes_weights(self,region, hpmass,addMass=False, invertsignal=False, regression=False, absoluteWeight=True, conserveorder=True):
        """returns a features data frame, a classes and a weights Series
           region: string, region of the events to be returned
           hpmass: string "multi" or integer specifying the H+ mass
           addMass: bool if True the truth H+ mass is added to the feature matrix (for parameterized ML training)
           invertsignal: if true all signal points except for hpmass are selected
           regression: bool, if false y will be 0=background or 1=signal, if true it will be the hp mass or 0 for background
           absoluteWeight: bool, true by default, if true the absolute value of the weights will be returned, if false also negative values will be returned
           conserveorder: if true the new column will be in the same position as the corresponding list column
        """
        
        mask=self.get_pandasframe_mask(region, hpmass, invertsignal=invertsignal)
        self.pandasframe = self.load_vec_features(conserveorder=conserveorder)
        if addMass:
            features=self.pandasframe[mask].loc[:,self.feature_names+["hpmass"]].copy()
        else:
            features=self.pandasframe[mask].loc[:,self.feature_names].copy()
        
        weights=self.pandasframe[mask].weight
        if absoluteWeight:
            weights=abs(weights)

        if regression: 
            classes=self.pandasframe[mask].process.apply(lambda mass: self.backgroundclass if not "Hp" in mass else float(mass.replace("Hp","")))
        else:
            classes=self.pandasframe[mask].process.apply(lambda mass: self.backgroundclass if not "Hp" in mass else 1)
        features.reset_index(inplace = True, drop=True)
        weights.reset_index(inplace = True, drop=True) 
        classes.reset_index(inplace = True, drop=True)
        return features, classes, weights

    def get_split_series(self, region, hpmass, digits_train=np.linspace(0,98,50), digits_test=np.linspace(1,99,50),invertsignal=False):
        """return a series of integers 0,1,2 which sample of training, testing and evaluation an event belongs to. The events are categorized according to whether the last two digits of the event number are in the np array digits_train, digits_test or none of the two.

           region: string, region of the events to be returned
           hpmass: string "multi" or integer specifying the H+ mass
           digits_train: see above
           digits_test: see above
           invertsignal: if true all signal points except for hpmass are selected
        """
        
        mask=self.get_pandasframe_mask(region, hpmass, invertsignal=invertsignal)
        return self.pandasframe[mask].eventNumber.apply(lambda x: 0 if x%100 in digits_train else 1 if x%100 in digits_test else 2)

    def prepare(self, region="INC_ge6jge4b", hpmass="multi", random=False, shuffle=True, addMass=False, invertsignal=False, regression=False, absoluteWeight=True, conserveorder=True):
        """ returns feature matrices, class labels and weights for training, testing and evaluation datasets (X_train, y_train, w_train, X_test, y_test, w_test, X_eval, y_eval, w_eval)
            region: string, region of the events to be returned
            hpmass: string "multi" or integer specifying the H+ mass
            random: do not use event number for putting into training, test and evaluation dataset, but a random number (50%:50%) split
            shuffle: bool, if true events will be shuffled
            addMass: bool if True the truth H+ mass is added to the feature matrix (for parameterized ML training)
            invertsignal: if true all signal points except for hpmass are selected
            regression: bool, if false y will be 0=background or 1=signal, if true it will be the hp mass or 0 for background
            absoluteWeight: bool, true by default, if true the absolute value of the weights will be returned, if false also negative values will be returned
            conserveorder: if true the new column will be in the same position as the corresponding list column
        """

        features, classes, weights=self.get_features_classes_weights(region,hpmass, addMass=addMass, invertsignal=invertsignal, regression=regression, absoluteWeight=absoluteWeight, conserveorder=conserveorder)
        if random:
            #mask=self.get_pandasframe_mask(region, hpmass, invertsignal=invertsignal)
            #print mask.shape, features.shape, classes.shape, weights.shape, mask.sum()
            X_train, X_test,  y_train, y_test, w_train, w_test = train_test_split(features, classes, weights, test_size=0.5,shuffle=True, random_state=self.random_state)
            return X_train, X_test, None, y_train, y_test, None, w_train, w_test, None #evaluation split is empty
        else:
            split_series=self.get_split_series(region, hpmass, invertsignal=invertsignal)
        return HpMLUtils.train_test_split3(features,classes,weights, shuffle=shuffle, test_fold=split_series, random_state=self.random_state)
