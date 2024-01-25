#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 00:01:28 2024

@author: lucasdenis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataset import filter_fc, select_comp


###############################################################################
#Scree plot function

def scree_plot(data, columns):
    
    fpkm_columns = data.filter(like=columns).columns
    fpkm_data = data[fpkm_columns]

    # Transpose the DataFrame
    fpkm_data_transposed = fpkm_data.T

    # Perform PCA
    pca = PCA(n_components=12)
    pca.fit_transform(fpkm_data_transposed)

    #percentage of variation
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = [str(x) for x in range(1,len(per_var)+1)]

    #Plot features
    plt.figure(figsize=(10, 6))
    plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Components')
    plt.title('Scree Plot')
    plt.axhline(y=5, color='red', linestyle='--', label='5% Explained Variance')
    plt.legend()
    
    #annotation
    for i in range(3):
        plt.annotate(f'PC{i + 1}: {per_var[i]}%', (i + 1, per_var[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center')
    
    plt.show(block=True)


###############################################################################
#PCA function

def plot_PCA(data, columns, conditions, n_components=2, nb_conditions=8, nb_replicats=3):

    fpkm_columns = data.filter(like=columns).columns
    fpkm_data = data[fpkm_columns]
        
    # Transpose the DataFrame
    fpkm_data_transposed = fpkm_data.T
    
    # set the colors for the plot
    np.random.seed(42)
    
    if conditions is None:
        conditions = [f'Condition {i+1}' for i in range(nb_conditions)]
    all_colors = sns.color_palette('colorblind', int(nb_conditions/2))
    colors = [color for pair in zip(all_colors, all_colors) for color in pair]
    markers = ['o','x']*int(nb_conditions/2)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(fpkm_data_transposed)

    # Create a DataFrame for the PCA results
    pca_RNAseq = pd.DataFrame(data=pca_result, 
                              columns=[f'PC{i+1}' for i in range(n_components)])

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    
    # Cosmetic PCA plot
    for i, condition in enumerate(conditions):
        condition_mask = np.arange(nb_replicats) + i * nb_replicats
        plt.scatter(pca_RNAseq['PC1'][condition_mask], 
                    pca_RNAseq['PC2'][condition_mask], 
                    color=colors[i], marker=markers[i], s=60, label=condition)

    plt.title('PCA of RNAseq Data', fontsize=14)
    
    if n_components == 2:
        plt.xlabel('PC 1', fontsize=14)
        plt.ylabel('PC 2', fontsize=14)

    else:
        print(f"PCA Results for {n_components} components:\n", pca_RNAseq)

    # Add legend
    plt.legend()
    plt.show(block=True)


###############################################################################
def log2_plot(data):
    
    fc_values = np.arange(1, 190.1, 0.5)
    
    # Calculate the number of genes for each fold change threshold
    nb_genes_list = [len(filter_fc(data, fc)) for fc in fc_values]

    # Set seaborn style
    sns.set(style="whitegrid", font_scale=1.2)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=fc_values, y=nb_genes_list, linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Fold change')
    plt.ylabel('Number of genes')
    plt.title('Number of Genes vs. Fold Change Threshold')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.show(block=True)
    

###############################################################################
#heatmap with hierachical clustering

def heatmap_hc(data, fc_threshold, columns, method='average', metric='euclidean', 
               row_cluster=True, col_cluster=True):
    
    fc_filt_data = filter_fc(data, fc_threshold)
    
    #select columns containing a specified string
    fpkm_columns = fc_filt_data.filter(like=columns).columns
    fpkm_data = fc_filt_data[fpkm_columns]
    
    # Create a clustered heatmap
    cluster_hm = sns.clustermap(fpkm_data, method=method, metric=metric, 
                    row_cluster=row_cluster, col_cluster=col_cluster, 
                    cmap='viridis')
    plt.ylabel('Gene expressions'+'\nnb genes= '+str(len(fc_filt_data)))
    cluster_hm.ax_heatmap.set(yticklabels=[])
    
    plt.show(block=True)


###############################################################################
#Elbow plot function to choose "k" in k-mean clustering

def elbow_plot(data, columns, fc_threshold):
    
    k_values = range(1, 15)
    fc_filt_data = filter_fc(data, fc_threshold)

    #select relevant columns
    fpkm_columns = fc_filt_data.filter(like=columns).columns
    fpkm_data = fc_filt_data[fpkm_columns]

    #Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(fpkm_data)

    # Calculate inertia for each k
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia_values.append(kmeans.inertia_)
    
    # Plot the elbow plot
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Plot for Optimal K')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    
    plt.show(block=True)


###############################################################################
#k-mean cluster heatmap plot
def heatmap_km(data, columns, fc_threshold, k):
    
    #filter on fold change
    fc_filt_data = filter_fc(data, fc_threshold)
    
    #select relevant columns
    gene_symbol = fc_filt_data['gene_symbol']
    fpkm_columns = fc_filt_data.filter(like=columns).columns
    fpkm_data = fc_filt_data[fpkm_columns]

    #Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(fpkm_data)
    
    #number of cluster
    kmeans = KMeans(n_clusters=k, random_state=42)
    fpkm_data['cluster'] = kmeans.fit_predict(scaled_data)
    fpkm_data['gene_symbol'] = gene_symbol
    
    # Clustered data
    clustered_data = fpkm_data.set_index('gene_symbol').sort_values('cluster')

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    heatmap_km = sns.heatmap(clustered_data.iloc[:, :-1], cmap='viridis', 
                             cbar_kws={'label': 'Expression Level'})
    plt.title('Gene Expression Heatmap - '+str(k)+' clusters')
    plt.ylabel('Genes (n='+str(len(gene_symbol))+')')
    plt.xlabel('Conditions')
    heatmap_km.set_yticklabels([])
    
    plt.show(block=True)
    

###############################################################################
#Volcano plot

def volcano_plot(data, condition_1, condition_2, fc_threshold, qvalue_threshold=0.05):
    
    #select data for the comparison (wiith the select_comp function)
    comp = select_comp(data, condition_1, condition_2)
    
    #transform the qvalue
    comp['qvalue'] = comp['qvalue'].apply(lambda x: -np.log(x))
    
    # Thresholds for significance and fold change
    qvalue_lim_log = -np.log(qvalue_threshold)
    fc_lim_log2 = np.log2(fc_threshold)
    
    #Create a volcano plot
    plt.figure(figsize=(10, 6))
    plt.scatter(comp['log2fc'], comp['qvalue'], color='grey', 
                alpha=0.7, edgecolors='none')

    # Highlight significant points
    mask = (comp['qvalue'] > qvalue_lim_log) & (np.abs(comp['log2fc']) > fc_lim_log2)
    plt.scatter(comp['log2fc'][mask], comp['qvalue'][mask], 
                color='red', alpha=0.5)

    # Set labels and title
    plt.xlabel('log2(fold change)')
    plt.ylabel('-log(q-value)')
    plt.title(str(condition_1)+' vs. '+ str(condition_2))
    plt.axhline(y=qvalue_lim_log, color='black', linestyle='--', linewidth=1, 
                label='Significance Threshold: '+str((qvalue_threshold)))
    plt.axvline(x=fc_lim_log2, color='blue', linestyle='--', linewidth=1, 
                label='Fold Change Threshold: '+str((fc_threshold)))
    plt.axvline(x=-fc_lim_log2, color='blue', linestyle='--', linewidth=1)

    # Add legend
    plt.legend()
    plt.show(block=True)







        
        
        