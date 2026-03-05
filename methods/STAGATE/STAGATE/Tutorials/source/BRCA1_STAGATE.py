import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure
import torch

import STAGATE_pyG as STAGATE

def calculate_clustering_matrix(pred, gt, sample):
    cols = ['Sample', 'Score', "Metric"]
    df = pd.DataFrame(columns=cols)
    
    pca_ari = adjusted_rand_score(pred, gt)
    df = df._append(pd.Series([sample, pca_ari, "ARI"],
                             index=cols), ignore_index=True)
    
    pca_ami = adjusted_mutual_info_score(pred, gt)
    df = df._append(pd.Series([sample, pca_ami, "AMI"],
                             index=cols), ignore_index=True)
    
    pca_hcv = homogeneity_completeness_v_measure(gt, pred)
    df = df._append(pd.Series([sample, pca_hcv[0], "Homogeneity"],
                             index=cols), ignore_index=True)
    
    df = df._append(pd.Series([sample, pca_hcv[1], "Completeness"],
                             index=cols), ignore_index=True)
    
    df = df._append(pd.Series([sample, pca_hcv[2], "V_measure"],
                            index=cols), ignore_index=True)
    
    return df
    
def main():
    # the location of R (used for the mclust clustering)
    # os.environ['R_HOME'] = '/home/lytq/.conda/envs/stagate/lib/R'
    # os.environ['R_USER'] = '/.conda/envs/stagate/lib/python3.10/site-packages/rpy2'

    data_path = '/home/lytq/STAGATE/data/BRCA1'
    data_names = ['V1_Human_Breast_Cancer_Block_A_Section_1']
        
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    for section_id in data_names:
        print(f'Processing {section_id}...')
        n_clusters = 20

        dir_out = f'/home/lytq/STAGATE/results/BRCA1'
        os.makedirs(dir_out, exist_ok=True)

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
        # Ann_df['Ground Truth'] = Ann_df['layer_guess']
        # adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
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

        ## Save UMAP
        plt.rcParams["figure.figsize"] = (4, 3)
        sc.pl.umap(adata, color=["mclust"], title=['STAGATE'])

        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # sc.pl.umap(adata, color='Ground Truth', ax=axes[0], show=False)
        # sc.pl.umap(adata, color='mclust', ax=axes[1], show=False)
        # axes[0].set_title('Ground Truth')
        # axes[1].set_title('STAGATE Clustering')

        # for ax in axes:
        #     ax.set_aspect(1)

        # plt.tight_layout()
        plt.savefig(os.path.join(dir_out, f'umap.png'), bbox_inches='tight', dpi=300)
        # plt.close()
        

        plt.rcParams["figure.figsize"] = (4, 3)
        sc.pl.spatial(adata, color=["mclust"], title=['STAGATE'], frameon=False, spot_size=150)
        plt.savefig(os.path.join(dir_out, f'domains.png'), bbox_inches='tight', dpi=300)

        ## Save results
        adata.obs['STAGATE'] = adata.obs['mclust']
        adata.write(f'{dir_out}/result.h5ad')
        adata.obs.to_csv(f'{dir_out}/metadata.tsv', sep='\t')

        df = pd.DataFrame(data=adata.obsm['STAGATE'], index=adata.obs.index)
        df.to_csv(f'{dir_out}/PCs.tsv', sep='\t')
        
        # used_adata = adata[adata.obs['Ground Truth']!='nan',]
        # used_adata = used_adata[~used_adata.obs['Ground Truth'].isna()]
        # sc.tl.paga(used_adata, groups='Ground Truth')
        # plt.rcParams["figure.figsize"] = (4, 3)
        # sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=False, size=20,
        #                 title=section_id+'_STAGATE', legend_fontoutline=2, show=False)
        # plt.savefig(os.path.join(dir_out, f'{section_id}_trajectory.png'), bbox_inches='tight', dpi=300)        

        # if section_id == '151673':
        #     continue
        low_dim_data = pd.DataFrame(adata.obsm['STAGATE'], index=adata.obs.index)
        expression_data = pd.DataFrame(adata.layers['count'], index=adata.obs.index, columns=adata.var.index)
        cell_metadata = adata.obs

        low_dim_data.to_csv(f"{dir_out}/low_dim_data.csv")
        expression_data.T.to_csv(f"{dir_out}/expression_matrix.csv")
        cell_metadata.to_csv(f"{dir_out}/cell_metadata.csv")
        
        umap_coords = adata.obsm["X_umap"]
        spot_ids = adata.obs_names
        umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        umap_df["spot_id"] = spot_ids
        umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
        umap_df.to_csv(os.path.join(dir_out, "spatial_umap_coords.csv"), index=False)



if __name__ == '__main__':
    main()