from __future__ import print_function
import json,os
import pandas as pd
from root_pandas import read_root
import copy
def getRWstring(rw_str,reg):
    rw_str = rw_str.replace("#","4j") if "4jex" in reg["Name"] else \
             rw_str.replace("#","5j") if "5jex" in reg["Name"] else \
             rw_str.replace("#","6j") if "6jex" in reg["Name"] else \
             ""
    print(rw_str)
    return rw_str

class tqXAnalysis:
    """Class that provides the tqX analysis pandas datasets"""

    def __init__(self):
        """Constructor with json with regions, samples and weight definitions"""

        self.config = None
        self.feature_names = []
        self.df_mc = None

        self.inputpath = ''
        self.treename = ''
        self.mcweight = ''
        self.lumiscale = 1.

    def getGeneralSettings(self,configfilename):
        with open(configfilename) as f:
            self.config = json.load(f)
        if not "Job" in self.config:
            print("ERROR: Did not find Job in fit configuration")
        if not "NtuplePaths" in self.config["Job"]:
            print("ERROR: Did not find NtuplePaths in fit configuration -> Job")
        self.inputpath=self.config["Job"]["NtuplePaths"]
        if not "NtupleName" in self.config["Job"]:
            print("ERROR: Did not find NtupleName in fit configuration -> Job")
        self.treename=self.config["Job"]["NtupleName"]
        if not "MCweight" in self.config["Job"]:
            print("ERROR: Did not find MCweight in fit configuration -> Job")
        self.mcweight=self.config["Job"]["MCweight"]
        self.lumiscale=1.

    def readData(self, samples=[], regions=[]):
        if len(samples)>1:
            self.df_mc=None
        print(samples)
        for sample in self.config["Sample"]:
            if len(samples)==0 or sample['Name'] in samples:
                #if sample['Name']=='data':
                #    continue
                for region in self.config["Region"]:
                    print(region)
                    if len(regions)==0 or region['Name'] in regions:
                        print("Reading",sample['Name'],"in region",region["Name"])
                        selection=region["Selection"]
                        if "Selection" in sample:
                            selection="("+selection+") && ("+sample["Selection"]+")"

                        mcweight="1.0"
                        RW=""
                        if sample['Name']!='data':
                            mcweight=self.mcweight
                            if "MCweight" in region:
                                mcweight="("+self.mcweight+")*("+region["MCweight"]+")"
                            if "MCweight" in sample:
                                mcweight="("+self.mcweight+")*("+sample["MCweight"]+")"
                        #get list of input files
                            if "RWeighting" in sample:
                                RW=getRWstring(sample["RWeighting"],region)
                        datafiles=[]
                        if not "NtuplePathSuffs" in region:
                            region["NtuplePathSuffs"]=""
                        if type(region["NtuplePathSuffs"])!=list:
                            region["NtuplePathSuffs"]=[region["NtuplePathSuffs"]]
                        inputpath = self.inputpath
                        if "Path" in sample:
                            inputpath = sample['Path']
                        for ntuplepath in region["NtuplePathSuffs"]:
                            if not "NtupleFiles" in sample and "NtupleFile" in sample:
                                sample["NtupleFiles"]=sample["NtupleFile"]
                            if type(sample["NtupleFiles"])!=list:
                                sample["NtupleFiles"]=[sample["NtupleFiles"]]
                            for ntuplefile in sample["NtupleFiles"]:
                                datafile=inputpath.rstrip("/")+"/"+ntuplepath.strip("/")+"/"+ntuplefile.lstrip("/")+".root"
                                if os.path.isfile(os.path.expanduser(datafile)):
                                    datafiles.append(datafile)
                        #get the data frame from the root file
                        #columns=self.feature_names+["eventNumber"]
                        columns = copy.copy(self.feature_names) 
                        torename = []
                        todelete = []
                        for ifeat in columns:
                            if ":" in ifeat:
                                todelete.append(ifeat)
                                if "sys" in ifeat.split(":")[0] and ("_alt_" in sample['Name'] or "data" in sample['Name']):
                                    continue
                                torename.append(ifeat.split(":"))
                                columns.append("noexpand:"+torename[-1][1])
                            if ";" in ifeat:
                                print("found",ifeat)
                                todelete.append(ifeat)
                                if "sys" in ifeat.split(";")[0] and ("_alt_" in sample['Name'] or "data" in sample['Name']):
                                    continue
                                torename.append(ifeat.split(";"))
                                columns.append(torename[-1][1])
                        for ifeat in todelete:
                            columns.remove(ifeat)
                            print(ifeat,"deleted")
                        if self.lumiscale!=1.:
                            mcweight=mcweight+"*"+str(self.lumiscale)
                        if RW!="":
                            mcweight=mcweight+"*"+RW
                        if sample['Name']!='data':
                            columns=columns+["noexpand:"+mcweight] 
                        
                        print("Reading:")
                        print(sample['Name'],datafiles,self.treename,columns,selection,mcweight)
                        tmpdf = read_root(datafiles,self.treename,columns=columns,where=selection)
                        print("read")
                        tmpdf.rename(columns={mcweight: 'weight'}, inplace=True)
                        if len(torename)!=0:
                            for newname,var in torename:
                                tmpdf.rename(columns={var:newname}, inplace=True)
                        print(tmpdf.columns.unique())
                        tmpdf["process"] = sample['Name']
                        #if not "Group" in sample:
                        #    tmpdf["group"]=sample['Name']
                        #else:
                        #    tmpdf["group"]=sample['Group']
                        tmpdf["X_mass"]=-1
                        if sample['Type']=="SIGNAL":
                            if not "X" in sample['Name']:
                                tmpdf["X_mass"]=125
                            else:
                                tmpdf["X_mass"]=int(sample['Name'].split("_")[-1])
                        tmpdf["region"]=region['Name']
                        print(tmpdf["process"],tmpdf.shape)
                        if type(self.df_mc)==None:
                            self.df_mc=tmpdf
                        else:
                            self.df_mc=pd.concat([self.df_mc,tmpdf],axis=0)


