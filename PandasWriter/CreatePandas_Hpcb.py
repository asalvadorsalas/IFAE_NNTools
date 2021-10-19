#!/usr/bin/env python
from __future__ import print_function
from HpcbData import *

configfile = 'pandasconfig_Hpcb.json'
Hpcb = HpcbAnalysis()

Hpcb.lumiscale = 139000.0
Hpcb.feature_names = [] 
for i in range(0,6):
    Hpcb.feature_names+=["jet"+str(i)+"_pt_bord","jet"+str(i)+"_eta_bord","jet"+str(i)+"_phi_bord","jet"+str(i)+"_m_bord","jet"+str(i)+"_y_bord","jet"+str(i)+"_btagw_discrete_bord"]
Hpcb.feature_names+=["lep1_pt","lep1_phi","lep1_eta","met","met_phi"]
Hpcb.feature_names+=["mbb_leading_bjets","mbb_maxdr","mbb_mindr","m_jj_leading_jets","jet0_pt","jets_n","bjets_n"]
Hpcb.feature_names+=["nomWeight_weight_btag","nomWeight_weight_jvt","nomWeight_weight_leptonSF","nomWeight_weight_mc","nomWeight_weight_norm","nomWeight_weight_pu","event_number","scores*"]

Hpcb.getGeneralSettings(configfile)
samples = [] 
for i in ["60","70","80","90","100","110","120","130","140","150","160"]:
    samples+=["Hp"+i]
samples+=["ttb","ttc","ttbarlight","Wtocb","Singletop","topEW","ttH","Wjetslight","Zjetslight","Dibosons","tH","ttH"]
Hpcb.readData(samples=samples)
Hpcb.df_mc.reset_index(inplace=True)
Hpcb.df_mc.to_feather("pandas_Hpcb_v4.feather")
