#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:22:20 2024

@author: lucasdenis
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from dataset import RNAseq_data, risk_factor_data, filter_fpkm, log_trans 
from dataset import mean_fpkm, filter_qvalue, gene_extraction
from visualization import scree_plot, plot_PCA, log2_plot, heatmap_hc 
from visualization import elbow_plot, heatmap_km, volcano_plot


#Parser
parser = argparse.ArgumentParser()
parser.add_argument('--fpkm', default=1, type=float)
parser.add_argument('--qvalue', default=0.05, type=float)
parser.add_argument('--fc_hm', default=4, type=float)
parser.add_argument('--k', default=6, type=int)
parser.add_argument('--fc_vp', default=2, type=float)
args = parser.parse_args()

#Data structure
print(RNAseq_data.shape)
print(RNAseq_data.head())


###############################################################################
#Data processing

#fpkm_filtering
RNAseq_filt_fpkm = filter_fpkm(RNAseq_data,fpkm_threshold=args.fpkm)
print(RNAseq_filt_fpkm.shape)
print(RNAseq_filt_fpkm.head())

#log-transform fpkm to stabilize variance for the analysis
log_fpkm_data = log_trans(RNAseq_filt_fpkm)
print(log_fpkm_data.head())

#add the mean of log(fpkm) for each conditions
RNAseq_mean_fpkm = mean_fpkm(log_fpkm_data)
print(RNAseq_mean_fpkm.shape)
print(RNAseq_mean_fpkm.head())

###############################################################################
#Exploratory Data Analysis

#PCA plot
conditions = ['DKO_BMP','DKO_no','HM1_BMP','HM1_no','Px3m_BMP','Px3m_no','Px7m_BMP','Px7m_no']
plot_PCA(log_fpkm_data, conditions=conditions, columns='fpkm')

#Scree plot
scree_plot(log_fpkm_data, columns='fpkm')


###############################################################################
#Heatmaps generation with hierachical clustering

#filter the gene on qvalue (at least one significant difference)

RNAseq_diff = filter_qvalue(RNAseq_mean_fpkm, qvalue_threshold=args.qvalue)
print(RNAseq_diff.shape)
RNAseq_diff.head()

#Plot number of Genes vs. Fold Change Threshold
log2_plot(RNAseq_diff)

#display the heatmap (you can change 'mean' to 'fpkm' to have every replicats)
heatmap_hc(RNAseq_diff, columns='mean', fc_threshold=args.fc_hm)


###############################################################################
#K-means Clustering

#elbow plot
elbow_plot(RNAseq_diff, columns='mean', fc_threshold=args.fc_hm)

#k-mean clustering plot
heatmap_km(RNAseq_diff, columns='mean', fc_threshold=args.fc_hm, k=args.k)


###############################################################################
#Volcano plot and gene extraction

#Volcano plot for Pax3mut_BMP vs. HM1_BMP
volcano_plot(RNAseq_filt_fpkm,condition_1='Pax3mut_BMP',condition_2='HM1_BMP',fc_threshold=args.fc_vp)

#Extract gene differentially express between two conditions with a fold change threshold
comparaison_1 = gene_extraction(RNAseq_filt_fpkm, condition_1='Pax3mut_BMP', 
                                condition_2='HM1_BMP', fc_threshold=args.fc_vp)


###############################################################################
#Comparison of disregulated genes with human factor risk genes of Neural Tube Deffect (NTD)

#risk_factor_data structure
print(risk_factor_data.head())

#Comparison of genes from comparaison_1 to risk_factor_data
matches = comparaison_1.isin(risk_factor_data['Ortholog.symbol'])
print('The number of genes matching is: '+ str(len(comparaison_1[matches])))
print(comparaison_1[matches])






