import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

import torch

import STAGATE.STAGATE_torch.STAGATE_pyG as STAGATE
import random
import time
import tracemalloc
import sys
sys.path.append('/home/lytq/Spatial-Transcriptomics-Benchmark/utils')
from evaluate import evaluate_clustering
    

SEEDS = [42, 123, 456, 789, 2024]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # the location of R (used for the mclust clustering)
    # os.environ['R_HOME'] = '/home/lytq/.conda/envs/stagate/lib/R'
    # os.environ['R_USER'] = '/.conda/envs/stagate/lib/python3.10/site-packages/rpy2'

    data_path = '/home/lytq/Spatial-Transcriptomics-Benchmark/data/BRCA1'
    output_path = '/home/lytq/Spatial-Transcriptomics-Benchmark/Results/'
    data_names = ['V1_Human_Breast_Cancer_Block_A_Section_1']
        
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')

    for seed in SEEDS:
        print("\n==============================")
        print(f"RUNNING SEED: {seed}")
        print("==============================")
        set_seed(seed)
        
        for section_id in data_names:
            print(f'Processing {section_id}...')
            n_clusters = 20

            dir_out = f'{output_path}/{seed}/BRCA1/STAGATE/'
            os.makedirs(dir_out, exist_ok=True)
            
            time_start = time.time()
            tracemalloc.start()

            # Load data
            input_dir = os.path.join(data_path, section_id)
            adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
            adata.var_names_make_unique()

            #Normalization
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # read the annotation
            Ann_df = pd.read_csv(os.path.join(data_path, section_id, 'metadata.tsv'), sep='\t', index_col=0)
            Ann_df['Ground Truth'] = Ann_df['fine_annot_type'].values
            adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
            adata.layers['count'] = adata.X.toarray()

            # plt.rcParams["figure.figsize"] = (3, 3)
            # sc.pl.spatial(adata, img_key="hires", color=["Ground Truth"])

            ## Constructing the spatial network
            STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
            STAGATE.Stats_Spatial_Net(adata)

            ## Runing STAGATE
            adata = STAGATE.train_STAGATE(adata, device=device)
            print(f'Finish training STAGATE for {section_id}')
            
            ## Perform clustering
            sc.pp.neighbors(adata, use_rep='STAGATE')
            sc.tl.umap(adata)
            adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)

            ## Calculate clustering metrics
            # obs_df = adata.obs.dropna()
            # clustering_results = calculate_clustering_matrix(obs_df['mclust'], obs_df['Ground Truth'], section_id)
            # clustering_results.to_csv(f'{dir_out}/clustering_results.tsv', sep='\t', index=False)
            # ARI = clustering_results[clustering_results['Metric']=='ARI']['Score'].values[0]
            # print('Adjusted rand index = %.2f' %ARI)

            time_end = time.time()
            time_taken = time_end - time_start
            current, peak = tracemalloc.get_traced_memory()
            memory_used = peak / (1024 ** 2)
            tracemalloc.stop()
            
            clustering_results = evaluate_clustering(adata, Ann_df, time_taken, memory_used, dir_out, 
                                                    pred_key='mclust', gt_df_key='fine_annot_type')
            ARI = clustering_results["ARI"]
            print('Adjusted rand index = %.2f' %ARI)

            ## Save UMAP
            # plt.rcParams["figure.figsize"] = (4, 3)
            # sc.pl.umap(adata, color=["mclust"], title=['STAGATE'])

            fig, axes = plt.subplots(1, 2, figsize=(10, 3))
            sc.pl.umap(adata, color='Ground Truth', ax=axes[0], show=False)
            sc.pl.umap(adata, color='mclust', ax=axes[1], show=False)
            axes[0].set_title('Manual Annotation')
            axes[1].set_title('STAGATE')

            # for ax in axes:
            #     ax.set_aspect(1)

            plt.tight_layout()
            plt.savefig(os.path.join(dir_out, 'umap.pdf'), format='pdf', dpi=300, bbox_inches='tight')
            # plt.close()
            
            # Plot spatial clustering
            plt.rcParams["figure.figsize"] = (6, 6)
            sc.pl.spatial(adata, 
                        color=["mclust"], 
                        title=['STAGATE (ARI=%.4f)'%ARI],     
                        frameon=False, spot_size=150)
            plt.savefig(os.path.join(dir_out, f'clustering.pdf'), bbox_inches='tight', dpi=300)

            ## Save results
            adata.obs['STAGATE'] = adata.obs['mclust']
            # adata.write(f'{dir_out}/result.h5ad')
            # adata.obs.to_csv(f'{dir_out}/metadata.tsv', sep='\t')

            
            # used_adata = adata[adata.obs['Ground Truth']!='nan',]
            # used_adata = used_adata[~used_adata.obs['Ground Truth'].isna()]
            # sc.tl.paga(used_adata, groups='Ground Truth')
            # plt.rcParams["figure.figsize"] = (4, 3)
            # sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
            #                 title=section_id+'_STAGATE', legend_fontoutline=2, show=False)
            # plt.savefig(os.path.join(dir_out, f'{section_id}_trajectory.png'), bbox_inches='tight', dpi=300)        

            low_dim_data = pd.DataFrame(adata.obsm['STAGATE'], index=adata.obs.index)
            # expression_data = pd.DataFrame(adata.layers['count'], index=adata.obs.index, columns=adata.var.index)
            cell_metadata = adata.obs

            low_dim_data.to_csv(f"{dir_out}/low_dim_data.csv")
            # expression_data.T.to_csv(f"{dir_out}/expression_matrix.csv")
            cell_metadata.to_csv(f"{dir_out}/cell_metadata.csv")
            
            umap_coords = adata.obsm["X_umap"]
            spot_ids = adata.obs_names
            umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
            umap_df["spot_id"] = spot_ids
            umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
            umap_df.to_csv(os.path.join(dir_out, "spatial_umap_coords.csv"), index=False)



if __name__ == '__main__':
    main()