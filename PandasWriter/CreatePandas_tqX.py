#!/usr/bin/env python
from __future__ import print_function
from tqXData import *

configfile = 'pandasconfig_tqX_evalRW.json'
tqX = tqXAnalysis()

tqX.lumiscale = 139000.0
tqX.feature_names = [] 
for i in range(0,6):
    #tqX.feature_names+=["jet"+str(i)+"_pt","jet"+str(i)+"_eta","jet"+str(i)+"_phi","jet"+str(i)+"_m","jet"+str(i)+"_y","jet"+str(i)+"_btagw_discrete"]
    tqX.feature_names+=["jet"+str(i)+"_pt_bord","jet"+str(i)+"_eta_bord","jet"+str(i)+"_phi_bord","jet"+str(i)+"_m_bord","jet"+str(i)+"_y_bord","jet"+str(i)+"_btagw_discrete_bord"]
tqX.feature_names+=["lep1_pt","lep1_phi","lep1_eta","met","met_phi"]
tqX.feature_names+=["mb1b2","mb1b3","mb2b3","DRb1b2","DRb1b3","DRb2b3"]
tqX.feature_names+=["mbb_leading_bjets","mbb_maxdr","mbb_mindr","m_jj_leading_jets","jets_n","bjets_n","dRmin_bb","dRmax_bb","dRavg_bb"]
#tqX.feature_names+=["jet0_pt"]
tqX.feature_names+= ["Wlep_pt","Wlep_eta","Wlep_phi","Wlep_m"]
tqX.feature_names+=["nomWeight_weight_btag","nomWeight_weight_jvt","nomWeight_weight_leptonSF","nomWeight_weight_mc","nomWeight_weight_norm","nomWeight_weight_pu","event_number"]
for i in ["30","40","50","60","70","80","90","100","120","140","150","160"]:
    tqX.feature_names+=["NNrecotqXv12_q_"+i,"NNtqXv12_q_"+i,"NNrecoXv12_q_"+i]

tqX.getGeneralSettings(configfile)
samples = [] 
for i in ["20","30","40","50","60","70","80","90","100","120","125","140","150","160"]:
    samples+=["uX_"+i,"ubarX_"+i,"cX_"+i,"cbarX_"+i]
samples+=["data","ttb","ttc","ttbarlight","Wtocb","Singletop","topEW","topEWtZ","ttH","Wjets","Zjets","Dibosons","tH"]
tqX.readData(samples=samples)
tqX.df_mc.reset_index(inplace=True)
tqX.df_mc.to_feather("pandas_tqXv9.feather")

#v6 pt ordered
#v7 with data btag ordered
