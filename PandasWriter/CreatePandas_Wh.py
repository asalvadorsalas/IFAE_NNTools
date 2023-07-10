#!/usr/bin/env python
from __future__ import print_function
from WhData import *

configfile = 'pandasconfig_Wh.json'
Wh = WhAnalysis()

Wh.feature_names = [] 
#Wh.feature_names+=["nJets","nBTags_DL1r_77","HT_all","HT_jets","Centrality","DeltaPhi_HW","DeltaPhi_HW","mVH","MaxMVA_Response","higgs_pt","wlep_pt","wlep_m","whad_pt","whad_m","H_mass","Hpt_over_mVH","Wpt_over_mVH","H1_index","H2_index","W1_index","W2_index","H1_index_btag","H2_index_btag","W1_index_btag","W2_index_btag","min_DeltaPhiJETMET","mWT"]
#Wh.feature_names+=["Muu_MindR_70","dRbb_avg_70","dRlepbb_MindR_70","Mbb_MindR_70","pT_jet5","H1_all","Mbb_MaxPt_70","Mbb_MaxM_70","Mjjj_MaxPt"]
#Wh.feature_names+=["jet_pt","jet_eta","jet_phi","jet_e","jet_DL1r","el_pt","el_eta","el_phi","el_e","mu_pt","mu_eta","mu_phi","mu_e","met_met","met_phi","eventNumber"]

Wh.feature_names+=["nJets","nBTags_DL1r_77","toplep_m","wlep_pt","wlep_eta","wlep_phi","wlep_m","whad_pt","whad_eta","whad_phi","whad_m","mWT","eventNumber"]
Wh.feature_names+=["mVH_lep","DeltaEta_VHlep","DeltaPhi_VHlep","Hleppt_over_mVH","Wleppt_over_mVH","higgs_lep_pt","higgs_lep_eta","higgs_lep_phi","higgs_lep_m","MaxMVA_LepResponse","nBTags_Higgs_lep","H1_index_lep","H2_index_lep","H1_index_lep_btag","H2_index_lep_btag","mVH_had","DeltaEta_had","DeltaPhi_had","Hhadpt_over_mVH","Whadpt_over_mVH","higgs_had_pt","higgs_had_eta","higgs_had_phi","higgs_had_m","MaxMVA_HadResponse","nBTags_Higgs_had","H1_index_had","H2_index_had","W1_index_had","W2_index_had","H1_index_had_btag","H2_index_had_btag","W1_index_had_btag","W2_index_had_btag"]
Wh.feature_names+=["jet_pt","jet_eta","jet_phi","jet_e","jet_tagWeightBin_DL1r_Continuous","lepton_charge","el_pt","el_eta","el_phi","el_e","mu_pt","mu_eta","mu_phi","mu_e","met_met","met_phi"]
Wh.feature_names+=["Centrality","min_DeltaPhiJETMET","dRbb_avg_77","H1_all","HT_jets","Mbb_MaxM_77","Mbb_MaxPt_77","Mbb_MindR_77","Muu_MindR_77"]

Wh.getGeneralSettings(configfile)
samples = [] 
for i in ["250","300","350","400","500","600","700","800","900","1000","1200","1400","1600","1800","2000","2500","3000"]:
    samples+=["Hp"+i]
samples+=["ttb","ttc","ttlight","diboson","zjets","wjets","othertop","st_tchan","st_wchan","ttH","ttW"]
Wh.readData(regions=["6jin4bin"],samples=samples)
Wh.df_mc.reset_index(inplace=True)
Wh.df_mc.to_feather("pandas_Wh_v2_6jin4bin.feather_2")
