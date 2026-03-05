import os,csv,re,sys
import pandas as pd
import numpy as np
import scanpy as sc
import math
import SpaGCN as spg
import random, torch
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('/home/lytq/Spatial-Transcriptomics-Benchmark/utils')
from sdmbench import compute_ARI, compute_NMI, compute_CHAOS, compute_PAS, compute_ASW, compute_HOM, compute_COM

import time
import psutil
import tracemalloc


seeds = [42, 123, 456, 789, 2024]

def evaluate_clustering(adata: sc.AnnData, df_meta, time_taken: float, memory_used: float, output_dir: str) -> dict:
    """Evaluate clustering using sdmbench"""
    gt_key = 'ground_truth'
    pred_key = 'refined_pred'
    adata.obs['ground_truth'] = df_meta['layer_guess'].values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    
    results = {
        "ARI": compute_ARI(adata, gt_key, pred_key),
        "AMI": compute_NMI(adata, gt_key, pred_key),
        "Homogeneity": compute_HOM(adata, gt_key, pred_key),
        "Completeness": compute_COM(adata, gt_key, pred_key),
        "ASW": compute_ASW(adata, pred_key),
        "CHAOS": compute_CHAOS(adata, pred_key),
        "PAS": compute_PAS(adata, pred_key),
        "Time": time_taken,
        "Memory": memory_used
    }
    
    df_results = pd.DataFrame([results])
    df_results.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    return results

BASE_PATH = Path('/home/lytq/Spatial-Transcriptomics-Benchmark/data/DLPFC')
output_path = Path('/home/lytq/Spatial-Transcriptomics-Benchmark/Results/')

sample_list = ['151507', '151508', '151509', '151510', 
                '151669', '151670', '151671', '151672', 
                '151673', '151674', '151675', '151676']

for seed in seeds:
    print("\n==============================")
    print(f"RUNNING SEED: {seed}")
    print("==============================")

    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ARI_list = []
    for sample_name in sample_list:
        print(f"================ Start Processing {sample_name} ======================")

        dir_input = Path(f'{BASE_PATH}/{sample_name}/')
        dir_output = Path(f'{output_path}/{seed}/DLPFC/SpaGCN/{sample_name}/')
        dir_output.mkdir(parents=True, exist_ok=True)

        if sample_name in ['151669', '151670', '151671', '151672']:
            n_clusters = 5
        else:
            n_clusters = 7
            
        time_start = time.time()
        tracemalloc.start()
        
        ##### read data
        adata = sc.read_visium(dir_input)
        adata.var_names_make_unique()

        spatial=pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)

        adata.obs["x1"]=spatial[1]
        adata.obs["x2"]=spatial[2]
        adata.obs["x3"]=spatial[3]
        adata.obs["x4"]=spatial[4]
        adata.obs["x5"]=spatial[5]

        adata=adata[adata.obs["x1"]==1]
        adata.var_names=[i.upper() for i in list(adata.var_names)]
        adata.var["genename"]=adata.var.index.astype("str")
        # adata.write_h5ad(f"{dir_output}/sample_data.h5ad")

        #Read in hitology image
        img=cv2.imread(f"{dir_input}/spatial/{sample_name}_full_image.tif")

        #Set coordinates
        adata.obs["x_array"]=adata.obs["x2"]
        adata.obs["y_array"]=adata.obs["x3"]
        adata.obs["x_pixel"]=adata.obs["x4"]
        adata.obs["y_pixel"]=adata.obs["x5"]
        x_array=adata.obs["x_array"].tolist()
        y_array=adata.obs["y_array"].tolist()
        x_pixel=adata.obs["x_pixel"].tolist()
        y_pixel=adata.obs["y_pixel"].tolist()

        #Test coordinates on the image
        img_new=img.copy()
        for i in range(len(x_pixel)):
            x=x_pixel[i]
            y=y_pixel[i]
            img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

        # cv2.imwrite(f'{dir_output}/sample_map.jpg', img_new)

        #Calculate adjacent matrix
        b=49
        a=1
        adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=a, histology=True)
        # np.savetxt(f'{dir_output}/adj.csv', adj, delimiter=',')


        ##### Spatial domain detection using SpaGCN
        spg.prefilter_genes(adata, min_cells=3) # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        #Normalize and take log for UMI
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)

        ### 4.2 Set hyper-parameters
        p=0.5 
        spg.test_l(adj,[1, 10, 100, 500, 1000])
        l=spg.find_l(p=p,adj=adj,start=100, end=500,sep=1, tol=0.01)
        n_clusters=n_clusters
        r_seed=t_seed=n_seed=seed
        res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, 
                            t_seed=t_seed, n_seed=n_seed)

        ### 4.3 Run SpaGCN
        clf=spg.SpaGCN()
        clf.set_l(l)
        #Set seed
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        #Run
        clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
        y_pred, prob=clf.predict()
        adata.obs["pred"]= y_pred
        adata.obs["pred"]=adata.obs["pred"].astype('category')
        #Do cluster refinement(optional)
        adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
        refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
        adata.obs["refined_pred"]=refined_pred
        adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
        
        df_meta = pd.read_csv(dir_input / 'metadata.tsv', sep='\t')
        adata.obs['layer_guess'] = df_meta['layer_guess']

        sc.pp.neighbors(adata, n_neighbors=10)
        sc.tl.umap(adata)
        
        time_taken = time.time() - time_start
        size, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_used = peak / (1024 ** 2) # MB
        
        clustering_results = evaluate_clustering(adata, df_meta, time_taken, memory_used, dir_output)
        
        #Save results
        # adata.write_h5ad(f"{dir_output}/results.h5ad")
        # adata.obs.to_csv(f'{dir_output}/cell_metadata.csv')
        
        #Set colors used
        # adata=sc.read(f"{dir_output}/results.h5ad")
        # plot_color=["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#997273","#787878","#DB4C6C","#9E7A7A","#554236","#AF5F3C","#93796C","#F9BD3F","#DAB370","#877F6C","#268785"]
        #Plot spatial domains
        
        # domains="pred"
        # num_celltype=len(adata.obs[domains].unique())
        # adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
        # ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
        # ax.set_aspect('equal', 'box')
        # ax.axes.invert_yaxis()
        # plt.savefig(f"{dir_output}/clustering.pdf", dpi=300, bbox_inches='tight')
        # plt.close()

        # #Plot refined spatial domains
        # domains="refined_pred"
        # num_celltype=len(adata.obs[domains].unique())
        # adata.uns[domains+"_colors"]=list(plot_color[:num_celltype])
        # ax=sc.pl.scatter(adata,alpha=1,x="y_pixel",y="x_pixel",color=domains,title=domains,color_map=plot_color,show=False,size=100000/adata.shape[0])
        # ax.set_aspect('equal', 'box')
        # ax.axes.invert_yaxis()
        # plt.savefig(f"{dir_output}/refined_clustering.pdf", dpi=300, bbox_inches='tight')
        # plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        sc.pl.spatial(adata, color='layer_guess', ax=axes[0], show=False)
        sc.pl.spatial(adata, color='refined_pred', ax=axes[1], spot_size=100, show=False)
        axes[0].set_title('Manual Annotation')
        axes[1].set_title(f'SpaGCN (ARI = {clustering_results["ARI"]:.4f})')
        handles, labels = axes[1].get_legend_handles_labels()
        new_labels = [str(int(label) + 1) if label.isdigit() else label for label in labels]
        axes[1].legend(handles, new_labels, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        for ax in axes:
            ax.axis('off')
        plt.savefig(f'{dir_output}/clustering.pdf', dpi=300, bbox_inches='tight')
        
        
        fig, axes = plt.subplots(1,2,figsize=(4*2, 3))
        sc.pl.umap(adata, color='layer_guess', ax=axes[0], show=False)
        sc.pl.umap(adata, color='refined_pred', ax=axes[1], show=False)
        axes[0].set_title('Manual Annotation')
        axes[1].set_title('SpaGCN')

        handles, labels = axes[1].get_legend_handles_labels()
        new_labels = [str(int(label) + 1) if label.isdigit() else label for label in labels]
        axes[1].legend(handles, new_labels, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))

        for ax in axes:
            ax.set_aspect(1)

        plt.tight_layout()
        plt.savefig(f'{dir_output}/umap.pdf', dpi=300, bbox_inches='tight')
        
        low_dim_data = pd.DataFrame(adata.obsm['X_pca'], index=adata.obs.index)
        cell_metadata = adata.obs
        low_dim_data.to_csv(f"{dir_output}/low_dim_data.csv", index=False)
        cell_metadata.to_csv(f"{dir_output}/cell_metadata.csv", index=False)
        
        umap_coords = adata.obsm["X_umap"]
        spot_ids = adata.obs_names
        umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        umap_df["spot_id"] = spot_ids
        umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
        umap_df.to_csv(os.path.join(dir_output, "spatial_umap_coords.csv"), index=False)
        
        # df_meta = df_meta[~pd.isnull(df_meta['layer_guess'])]
        ARI = clustering_results['ARI']
        print('===== Project: {} ARI score: {:.3f}'.format(sample_name, ARI))

    print('===== Project: AVG ARI score: {:.3f}'.format(np.mean(ARI_list)))

