#!/usr/bin/env python
#requires
from __future__ import print_function
from tqXData import *
from pandas import HDFStore

configfile = 'pandasconfig_tqX.json'
tqX = tqXAnalysis()

tqX.lumiscale = 139000.0
tqX.feature_names = [] 
for i in range(0,6):
    tqX.feature_names+=["jet"+str(i)+"_pt_bord","jet"+str(i)+"_eta_bord","jet"+str(i)+"_phi_bord","jet"+str(i)+"_m_bord","jet"+str(i)+"_y_bord","jet"+str(i)+"_btagw_discrete_bord"]
tqX.feature_names+=["lep1_pt","lep1_phi","lep1_eta","met","met_phi"]
tqX.feature_names+=["mbb_leading_bjets","mbb_maxdr","mbb_mindr","m_jj_leading_jets","jet0_pt","jets_n","bjets_n"]

#tqX.feature_names+=["nomWeight_weight_btag","nomWeight_weight_jvt","nomWeight_weight_leptonSF","nomWeight_weight_mc","nomWeight_weight_norm","nomWeight_weight_pu"]
tqX.getGeneralSettings(configfile)
samples = [] 
for i in ["20","30","40","50","60","70","80","90","100","120","125","140","160"]:
    samples+=["uX_"+i,"ubarX_"+i,"cX_"+i,"cbarX_"+i]
samples+=["ttb","ttc","ttbarlight","Wtocb","Singletop","topEW","ttH","Wjets22light","Zjets22light","Dibosons"]
tqX.readData(samples=samples)
tqX.df_mc.reset_index(inplace=True)
tqX.df_mc.to_feather("pandas_tqX.feather")
