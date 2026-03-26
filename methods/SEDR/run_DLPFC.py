import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable. 

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('/home/lytq/Spatial-Transcriptomics-Benchmark/utils')
from evaluate import evaluate_clustering

import time
import tracemalloc

import SEDR

data_path = '/home/lytq/Spatial-Transcriptomics-Benchmark/data/DLPFC'
output_path = '/home/lytq/Spatial-Transcriptomics-Benchmark/Results'

data_names = os.listdir(data_path)
data_names = [i for i in data_names if i.isdigit()]

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

SEEDS = [42, 123, 456, 789, 2024]


for seed in SEEDS:
    print(f"================ RUNNING SEED {seed} ======================")
    SEDR.fix_seed(seed)
    for section_id in data_names:
        print(f"    => Start Processing {section_id}")
        
        n_clusters = 5 if section_id in ['151669','151670','151671','151672'] else 7 
        
        dir_out = f'{output_path}/{seed}/DLPFC/SEDR/{section_id}'
        os.makedirs(dir_out, exist_ok=True)
        
        tracemalloc.start()
        start_time = time.time()
        
        # Load data
        adata = sc.read_visium(os.path.join(data_path, section_id))
        adata.var_names_make_unique()

        df_meta = pd.read_csv(os.path.join(data_path, section_id, 'metadata.tsv'), sep='\t')
        adata.obs['layer_guess'] = df_meta['layer_guess'].values

        adata.layers['count'] = adata.X.toarray()
        sc.pp.filter_genes(adata, min_cells=50)
        sc.pp.filter_genes(adata, min_counts=10)
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=2000)
        adata = adata[:, adata.var['highly_variable'] == True]
        sc.pp.scale(adata)

        adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
        adata.obsm['X_pca'] = adata_X

        graph_dict = SEDR.graph_construction(adata, 12)
        
        sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
        using_dec = True
        if using_dec:
            sedr_net.train_with_dec(N=1)
        else:
            sedr_net.train_without_dec(N=1)
        sedr_feat, _, _, _ = sedr_net.process()
        adata.obsm['SEDR'] = sedr_feat
        
        SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR', random_seed=seed)
        print('Clustering finished')
        
        # Evaluate clustering
        time_taken = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_used = peak / (1024 ** 2)
        tracemalloc.stop()

        results = evaluate_clustering(adata, df_meta, time_taken, memory_used, dir_out, pred_key='SEDR')
        print(f'ARI = {results["ARI"]:.4f}')

        # Plot clustering
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        sc.pl.spatial(adata, color='SEDR', ax=axes, show=False)
        axes.set_title(f'SEDR (ARI={results["ARI"]:.4f})')
        axes.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dir_out, 'clustering.pdf'), dpi=300, bbox_inches='tight')

        # Plot UMAP
        sc.pp.neighbors(adata, use_rep='SEDR', metric='cosine')
        sc.tl.umap(adata)

        
        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        sc.pl.umap(adata, color='layer_guess', ax=ax[0], show=False)
        sc.pl.umap(adata, color='SEDR', ax=ax[1], show=False)
        ax[0].set_title('Manual Annotation')
        ax[1].set_title('SEDR')
        for a in ax:
            a.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_out, 'umap.pdf'), format='pdf', dpi=300, bbox_inches='tight')    

        low_dim_data = pd.DataFrame(adata.obsm['SEDR'], index=adata.obs.index)
        low_dim_data.to_csv(f'{dir_out}/low_dim_data.csv')
        adata.obs.to_csv(f'{dir_out}/cell_metadata.csv')
        umap_coords = adata.obsm["X_umap"]
        spot_ids = adata.obs_names
        umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        umap_df["spot_id"] = spot_ids
        umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
        umap_df.to_csv(f'{dir_out}/spatial_umap_coords.csv')    
        
        print(f"    => Finished Processing {section_id}")