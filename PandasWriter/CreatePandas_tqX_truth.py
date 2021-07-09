#!/usr/bin/env python
from __future__ import print_function
from tqXData import *

configfile = 'pandasconfig_tqX_truth.json'
tqX = tqXAnalysis()

tqX.lumiscale = 139000.0
tqX.feature_names = [] 
tqX.feature_names += ["jet_pt","jet_eta","jet_phi","jet_e","jet_DL1r","jet_tagWeightBin_DL1r_Continuous"]
tqX.feature_names += ["nJets","nBTags_DL1r_60","eventNumber"]
tqX.feature_names += ["index_q","index_x1","index_x2","DRreco_x1x2","DRreco_qX","DRreco_topX","DRtruth_x1x2","DRtruth_qX","DRtruth_topX"]
tqX.feature_names += ["reco_top_pt","reco_top_DR","reco_top_m","reco_X_pt","reco_X_DR","reco_X_m"]
tqX.feature_names += ["reco_x1_pt","reco_x1_DR","reco_x1_m","reco_x1_btag","reco_x2_pt","reco_x2_DR","reco_x2_m","reco_x2_btag","reco_q_pt","reco_q_DR","reco_q_m","reco_q_btag"]
tqX.feature_names += ["truth_top_pt","truth_top_m","truth_X_pt","truth_X_m"]
tqX.feature_names += ["truth_x1_pt","truth_x1_m","truth_x1_pdgid","truth_x2_pt","truth_x2_m","truth_x2_pdgid","truth_q_pt","truth_q_m","truth_q_pdgid"]

tqX.getGeneralSettings(configfile)
samples = [] 
for i in ["20","30","40","50","60","70","80","90","100","120","140","150","160"]:
    samples+=["uX_"+i,"ubarX_"+i,"cX_"+i,"cbarX_"+i]
tqX.readData(samples=samples)
tqX.df_mc.reset_index(inplace=True)
print(tqX.df_mc.columns.unique())
tqX.df_mc = tqX.df_mc[tqX.feature_names+["weight","process","X_mass","region"]]
print(tqX.df_mc.columns.unique())
tqX.df_mc.to_feather("pandas_tqXtruth_v0.feather")
