import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, \
                            homogeneity_completeness_v_measure
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scanpy as sc
import stlearn as st
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('/home/lytq/Spatial-Transcriptomics-Benchmark/utils')
from sdmbench import compute_ARI, compute_NMI, compute_CHAOS, compute_PAS, compute_ASW, compute_HOM, compute_COM

import time
import psutil
import tracemalloc

import warnings
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 456, 789, 2024]

BASE_PATH = Path('/home/lytq/Spatial-Transcriptomics-Benchmark/data/DLPFC')
sample_list = ['151507', '151508', '151509', '151510', 
               '151669', '151670', '151671', '151672', 
               '151673', '151674', '151675', '151676']

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def evaluate_clustering(adata: sc.AnnData, df_meta, time_taken: float, memory_used: float, output_dir: str) -> dict:
    """Evaluate clustering using sdmbench"""
    gt_key = 'ground_truth'
    pred_key = 'X_pca_kmeans'
    adata.obs['ground_truth'] = df_meta['ground_truth_le'].values
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

# def calculate_clustering_matrix(pred, gt, sample, methods_):
#     df = pd.DataFrame(columns=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"])

#     pca_ari = adjusted_rand_score(pred, gt)
#     df = df.append(pd.Series([sample, pca_ari, "pca", methods_, "Adjusted_Rand_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

#     pca_nmi = normalized_mutual_info_score(pred, gt)
#     df = df.append(pd.Series([sample, pca_nmi, "pca", methods_, "Normalized_Mutual_Info_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

#     pca_purity = purity_score(pred, gt)
#     df = df.append(pd.Series([sample, pca_purity, "pca", methods_, "Purity_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

#     pca_homogeneity, pca_completeness, pca_v_measure = homogeneity_completeness_v_measure(pred, gt)

#     df = df.append(pd.Series([sample, pca_homogeneity, "pca", methods_, "Homogeneity_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)


#     df = df.append(pd.Series([sample, pca_completeness, "pca", methods_, "Completeness_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)

#     df = df.append(pd.Series([sample, pca_v_measure, "pca", methods_, "V_Measure_Score"],
#                              index=['Sample', 'Score', 'PCA_or_UMAP', 'Method', "test"]), ignore_index=True)
#     return df


# def purity_score(y_true, y_pred):
#     # compute contingency matrix (also called confusion matrix)
#     cm = contingency_matrix(y_true, y_pred)
#     # return purity
#     return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

for seed in SEEDS:
    print("\n==============================")
    print(f"RUNNING SEED: {seed}")
    print("==============================")
    set_seed(seed)
    
    for sample in sample_list:
        print(f"================ Start Processing {sample} ======================")

        OUTPUT_PATH = Path(f"/home/lytq/Spatial-Transcriptomics-Benchmark/Results/{seed}/DLPFC/stLearn/{sample}")
        if OUTPUT_PATH.exists():
            print(f"Output for sample {sample} already exists. Skipping...")
            continue
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        TILE_PATH = Path(f'{OUTPUT_PATH}/tiles/')
        TILE_PATH.mkdir(parents=True, exist_ok=True)

        # Start time and memory usage tracking
        start_time = time.time()
        tracemalloc.start()
        
        # Load data
        data = st.Read10X(os.path.join(BASE_PATH, sample))
        ground_truth_df = pd.read_csv( BASE_PATH / sample / 'metadata.tsv', sep='\t')
        ground_truth_df['ground_truth'] = ground_truth_df['layer_guess']

        # Pre-processing for ground truth
        le = LabelEncoder()
        ground_truth_le = le.fit_transform(list(ground_truth_df["ground_truth"].values))
        n_cluster = len((set(ground_truth_df["ground_truth"]))) - 1
        data.obs['ground_truth'] = ground_truth_df["ground_truth"]
        ground_truth_df["ground_truth_le"] = ground_truth_le 
        
        # pre-processing for gene count table
        st.pp.filter_genes(data,min_cells=1)
        st.pp.normalize_total(data)
        st.pp.log1p(data)
        st.em.run_pca(data,n_comps=15)
        st.pp.tiling(data, TILE_PATH)
        st.pp.extract_feature(data)
        
        # stSME
        st.spatial.SME.SME_normalize(data, use_data="raw", weights="physical_distance")
        data_ = data.copy()
        data_.X = data_.obsm['raw_SME_normalized']
        st.pp.scale(data_)
        st.em.run_pca(data_,n_comps=30)
        st.tl.clustering.kmeans(data_, n_clusters=n_cluster, use_data="X_pca", key_added="X_pca_kmeans", random_state=seed)
        
        st.pp.neighbors(data_, n_neighbors=10, use_rep="X_pca")
        st.em.run_umap(data_)

        
        # End time and memory usage tracking
        end_time = time.time()
        time_taken = end_time - start_time
        memory_used = tracemalloc.get_traced_memory()[1] / (1024 ** 2) # in MB
        tracemalloc.stop()
        
        os.system(f"rm -rf {TILE_PATH}")
        # print(data_)
        # Evaluate clustering
        metrics = evaluate_clustering(data_, ground_truth_df, time_taken, memory_used, OUTPUT_PATH)
        
        # Plot clusters
        fig, ax = plt.subplots(figsize=(6, 6))
        st.pl.cluster_plot(data_, use_label="X_pca_kmeans", ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [str(int(label) + 1) if label.isdigit() else label for label in labels]
        ax.legend(handles, new_labels, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5), markerscale=3)
        plt.title(f"stLearn (ARI = {metrics['ARI']:.4f})")
        plt.savefig(OUTPUT_PATH / 'clustering.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        # methods_ = "stSME_disk"
        # results_df = calculate_clustering_matrix(data_.obs["X_pca_kmeans"], ground_truth_le, sample, methods_)
        data_.obs.to_csv(OUTPUT_PATH / 'cell_metadata.csv')
        df_PCA = pd.DataFrame(data = data_.obsm['X_pca'], index = data_.obs.index)
        df_PCA.to_csv(OUTPUT_PATH / 'low_dim_data.csv', index=False)


        umap_coords = data_.obsm["X_umap"]
        spot_ids = data_.obs_names
        umap_df = pd.DataFrame(umap_coords, columns=["UMAP1", "UMAP2"])
        umap_df["spot_id"] = spot_ids
        umap_df = umap_df[["spot_id", "UMAP1", "UMAP2"]]
        umap_df.to_csv(OUTPUT_PATH / "spatial_umap_coords.csv", index=False)
        
        data_.obs['X_pca_kmeans_shift'] = (data_.obs['X_pca_kmeans'].astype(int) + 1).astype(str)
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(data_, color='X_pca_kmeans_shift', title='stLearn', ax=ax) 
        plt.savefig(OUTPUT_PATH / 'umap.pdf', format='pdf', dpi=300, bbox_inches='tight')
        
        print("================ End ======================")













