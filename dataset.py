#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:31:59 2024

@author: lucasdenis
"""

import pandas as pd
import numpy as np

#import datasets
RNAseq_data = pd.read_csv("RNAseq_data.csv", sep=';', decimal=",")
risk_factor_data = pd.read_csv("risk_factors.csv", sep=',', decimal=".")


###############################################################################
#function to filter on fpkm value

def filter_fpkm(data, fpkm_threshold):

    # Identify columns containing "fpkm" in their names
    columns_to_check = data.filter(like='fpkm').columns
    
    #filter rows
    filtered_data = data[data[columns_to_check].gt(fpkm_threshold).all(axis=1)]

    return filtered_data


###############################################################################
#function to log-transform the data

def log_trans(data):
    # Extract columns containing 'fpkm'
    fpkm_columns = data.filter(like='fpkm').columns
    
    # Create a new dataframe with the log-transformed values
    log_fpkm_data = data.copy()
    log_fpkm_data[fpkm_columns] = np.log1p(data[fpkm_columns])
    
    return log_fpkm_data


###############################################################################
#Average fpkm per condition

def mean_fpkm(data):
    
    #initialization
    conditions = ['DKO_BMP','DKO_no','HM1_BMP','HM1_no',
                  'Px3m_BMP','Px3m_no','Px7m_BMP','Px7m_no']
    conditions_mean = [condition + '_mean' for condition in conditions]
    nb_replicats = 3
    i=1

    #interation of averaging for each replicat
    for k in range(0,len(conditions)):
    
        j=i+nb_replicats
        data[conditions_mean[k]] = data.iloc[:,i:j].mean(axis=1)
        i=i+3

    return data


###############################################################################
#function to filter on qvalue

def filter_qvalue(data, qvalue_threshold):

    # Identify columns containing "qvalue" in their names
    columns_to_check = data.filter(like='qvalue').columns
    
    #filter rows
    filtered_data = data[data[columns_to_check].lt(qvalue_threshold).any(axis=1)]

    return filtered_data


###############################################################################
#function to filter on fold-change

def filter_fc(data, fc_threshold):

    # Identify columns containing "log2fc" in their names
    columns_to_check = data.filter(like='log2fc').columns
    
    #filter rows > threshold or < -threshold
    filtered_data = data[(data[columns_to_check] > np.log2(fc_threshold)).any(axis=1)| 
                         (data[columns_to_check] < np.log2(1/fc_threshold)).any(axis=1)]

    return filtered_data


###############################################################################
#function to select two conditions for comparison

def select_comp(data, condition_1, condition_2):
    
    selected_data = pd.concat([data.filter(like='log2fc'),
                           data.filter(like='qvalue')],axis=1)
    selected_data = selected_data.filter(like=condition_1).filter(like=condition_2)
    selected_data['genes'] = list(data['gene_symbol'])
    selected_data.columns.values[0] = 'log2fc'
    selected_data.columns.values[1] = 'qvalue'
    
    return selected_data


###############################################################################
#Extract differentially expressed genes from a comparison

def gene_extraction(data, condition_1, condition_2, fc_threshold, qvalue_threshold=0.05):
    
    selected_data = pd.concat([data.filter(like='log2fc'),
                           data.filter(like='qvalue')],axis=1)
    selected_data = selected_data.filter(like=condition_1).filter(like=condition_2)
    selected_data['genes'] = list(data['gene_symbol'])
    selected_data.columns.values[0] = 'log2fc'
    selected_data.columns.values[1] = 'qvalue'
    
    fc_threshold_log = np.log2(fc_threshold)
    
    #Select the genes from the new dataset
    mask = (selected_data['qvalue'] < qvalue_threshold) & (np.abs(selected_data['log2fc']) > fc_threshold_log)
    diff_expr_genes = selected_data['genes'][mask]
    
    return diff_expr_genes
   
    
   
    