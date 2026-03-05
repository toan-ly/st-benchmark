# Import necessary libraries
library(Seurat)
library(SeuratData)
library(ggplot2)
library(patchwork)
library(dplyr)
library(mclust)
library(aricode)
library(clevr)  # For homogeneity, completeness, v-measure
options(bitmapType = 'cairo')
options(future.globals.maxSize = 1024 * 1024 * 1024) # 1 GB
library(cluster)  # ASW
# library(pryr)  # Memory usage
# library(microbenchmark)  # Timing


seeds <- c(42, 123, 456, 789, 2024)

# Define batch-specific clustering parameters
batch_cluster_map <- list(
  'V1_Human_Breast_Cancer_Block_A_Section_1' = 20
)

# Paths
data_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/data/BRCA1")
save_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/Results/")

#' Calculate clustering evaluation metrics
#' @param ground_truth Vector of ground truth labels
#' @param clusters Vector of predicted cluster labels
#' @param data_matrix Matrix of data points
#' @return List of evaluation metrics
calculate_metrics <- function(ground_truth, clusters, data_matrix) {
  tryCatch({
    # Remove NA values
    valid_indices <- !is.na(clusters) & !is.na(ground_truth)
    clusters <- clusters[valid_indices]
    ground_truth <- ground_truth[valid_indices]
    data_matrix <- data_matrix[valid_indices, ]

    if (length(ground_truth) != length(clusters)) {
      stop("Mismatch in length between ground_truth and clusters.")
    }
    if (any(is.na(ground_truth)) || any(is.na(clusters))) {
      stop("Missing values found in ground_truth or clusters.")
    }

    # Convert factors to numeric for clustering metrics
    if (is.factor(clusters)) {
      clusters <- as.numeric(as.character(clusters))
    }
    if (is.factor(ground_truth)) {
      ground_truth <- as.numeric(as.character(ground_truth))
    }

    # # Convert to factors
    # ground_truth <- as.factor(ground_truth)
    # clusters <- as.factor(clusters)

    ARI <- adjustedRandIndex(ground_truth, clusters)
    AMI <- AMI(ground_truth, clusters)
    homogeneity <- clevr::homogeneity(ground_truth, clusters)
    completeness <- clevr::completeness(ground_truth, clusters)
    v_measure <- clevr::v_measure(ground_truth, clusters)

    # Calculate Silhouette Score (ASW)
    dist_matrix <- dist(data_matrix)
    sil <- silhouette(clusters, dist_matrix)
    ASW <- mean(sil[, 3])

    return(list(ARI = ARI, 
                AMI = AMI, 
                Homogeneity = homogeneity, 
                Completeness = completeness, 
                V_Measure = v_measure, 
                ASW = ASW))
  }, error = function(e) {
    warning("Error calculating metrics: ", e$message)
    return(list(ARI = NA, 
                AMI = NA, 
                Homogeneity = NA, 
                Completeness = NA, 
                V_Measure = NA, 
                ASW = NA))
  })
}

# Main loop
for (seed in seeds) {
  cat("\n==============================\n")
  cat("RUNNING SEED:", seed, "\n")
  cat("==============================\n")

  set.seed(seed)
  
  # Loop through all batches
  for (sample.name in names(batch_cluster_map)) {
    cat("Processing batch:", sample.name, "\n")
    n_clusters <- batch_cluster_map[[sample.name]]

    dir.input <- file.path(data_path, sample.name)
    dir.output <- file.path(save_path, as.character(seed), "BRCA1", "Seurat")
    if (!dir.exists(dir.output)) {
      dir.create(dir.output, recursive = TRUE)
    }

    start_time <- Sys.time()
    gc(reset = T)

    # Load spatial transcriptomics data
    sp_data <- tryCatch({
      Load10X_Spatial(dir.input, filename = "filtered_feature_bc_matrix.h5")
    }, error = function(e) {
      stop("Error loading spatial data for batch ", sample.name, ": ", e$message)
    })

    # Load metadata and add to the Seurat object
    df_meta <- read.table(file.path(dir.input, 'metadata.tsv'), sep = '\t', header=TRUE)
    sp_data <- AddMetaData(sp_data, metadata = df_meta$fine_annot_type, col.name = 'fine_annot_type')

    # Data processing: Visualization and QC plots
    plot1 <- VlnPlot(sp_data, features = "nCount_Spatial", pt.size = 0.1) + NoLegend()
    plot2 <- SpatialFeaturePlot(sp_data, features = "nCount_Spatial") + theme(legend.position = "right")
    qc_plot = wrap_plots(plot1, plot2)
    # ggsave(file.path(dir.output, 'QC.png'), plot = qc_plot, width = 10, height = 5)

    # Data normalization using SCTransform
    sp_data <- SCTransform(sp_data, assay = "Spatial", verbose = FALSE)

    # Dimensionality reduction and clustering
    sp_data <- RunPCA(sp_data, assay = "SCT", verbose = FALSE, npcs = 50)
    sp_data <- FindNeighbors(sp_data, reduction = "pca", dims = 1:30)

    # Find optimal resolution for clustering
    sp_data <- tryCatch({
      for (resolution in seq(1.2, 0.4, by = -0.01)) {
        sp_data <- FindClusters(sp_data, verbose = FALSE, resolution = resolution)
        if (length(levels(sp_data@meta.data$seurat_clusters)) == n_clusters) {
          cat("Optimal resolution found for batch", sample.name, ": ", resolution, "\n")
          break
        }
      }
      sp_data
    }, error = function(e) {
      stop("Error during clustering for batch ", sample.name, ": ", e$message)
    })


    # Run UMAP for visualization
    sp_data <- RunUMAP(sp_data, reduction = "pca", dims = 1:30)

    # End recording time and memory usage
    end_time <- Sys.time()
    elapsed_time <- as.numeric(difftime(end_time, start_time, units="secs"))
 
    memInfo1 <- gc()
    # memInfo1[11]
    # memInfo1[12]

    gc(reset = TRUE)
    memInfo2 <- gc()
    # memInfo2[11]
    # memInfo2[12]
    memory_usage <- memInfo1["Vcells", ncol(memInfo1)] - memInfo2["Vcells", ncol(memInfo2)]  # Memory usage in MB
    print(memory_usage)


    labels <- sp_data@meta.data$seurat_clusters
    gt <- sp_data@meta.data$fine_annot_type
    pca_data <- sp_data@reductions$pca@cell.embeddings 
    metrics <- calculate_metrics(gt, labels, pca_data)
    metrics$Time <- elapsed_time
    metrics$Memory <- memory_usage
    write.csv(as.data.frame(metrics), file = file.path(dir.output, 'metrics.csv'), row.names = FALSE)

    # Visualization of clustering results
    p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE) +
      ggtitle(paste("Seurat")) + 
      theme(plot.title = element_text(hjust = 0.5, size = 16),
            legend.title = element_blank())  

    p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3) +
      ggtitle(paste("Seurat (ARI =", round(metrics$ARI[1], 2), ")")) + 
      theme(plot.title = element_text(hjust = 0.5, size = 16),
            legend.title = element_blank())

    ggsave(file.path(dir.output, 'umap.pdf'), 
          plot = p1, width = 6, height = 6, dpi = 300)
    ggsave(file.path(dir.output, 'clustering.pdf'), 
          plot = p2, width = 6, height = 6, dpi = 300)
    
    # Save Seurat object and results
    # saveRDS(sp_data, file.path(dir.output, 'Seurat_final.rds'))
    write.csv(sp_data@reductions$pca@cell.embeddings,
                file = file.path(dir.output, 'low_dim_data.csv'),
                row.names = TRUE)
    write.csv(sp_data@meta.data,
                file = file.path(dir.output, 'cell_metadata.csv'),
                row.names = TRUE)

    # expression_data <- as.data.frame(as.matrix(GetAssayData(sp_data, assay = "Spatial", slot = "counts")))
    # write.table(t(expression_data), file = file.path(dir.output, "expression_matrix.tsv"), sep = "\t", quote = FALSE, col.names = NA)

    umap_coords <- as.data.frame(sp_data@reductions$umap@cell.embeddings)
    umap_coords$spot_id <- rownames(umap_coords)
    write.csv(umap_coords, file = file.path(dir.output, "spatial_umap_coords.csv"), row.names = FALSE)

  }
  cat("Done Breast Cancer.\n")
}