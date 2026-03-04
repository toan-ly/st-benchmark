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
library(cluster) # For silhouette score


seeds <- c(42, 123, 456, 789, 2024)

# Define batch-specific clustering parameters
batch_cluster_map <- list(
  '151669' = 5, '151670' = 5, '151671' = 5, '151672' = 5,
  '151673' = 7, '151674' = 7, '151675' = 7, '151676' = 7,
  '151507' = 7, '151508' = 7, '151509' = 7, '151510' = 7
)

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

    # Placeholder for CHAOS and PAS
    CHAOS <- NA  
    PAS <- NA    

    return(list(ARI = ARI, 
                AMI = AMI, 
                Homogeneity = homogeneity, 
                Completeness = completeness, 
                V_Measure = v_measure, 
                ASW = ASW, 
                CHAOS = CHAOS, 
                PAS = PAS))
  }, error = function(e) {
    warning("Error calculating metrics: ", e$message)
    return(list(ARI = NA, 
                AMI = NA, 
                Homogeneity = NA, 
                Completeness = NA, 
                V_Measure = NA, 
                ASW = NA, 
                CHAOS = NA, 
                PAS = NA))
  })
}

#' Save analysis results and visualizations
#' @param sp_data Processed Seurat object
#' @param metrics_df Metrics data frame
#' @param dir.output Output directory
save_results <- function(sp_data, metrics_df, dir.output) {
  # Save metrics
  write.csv(metrics_df, file = file.path(dir.output, 'metrics.csv'), row.names = FALSE)
  
  # Generate and save clustering plots separately
  p1 <- DimPlot(sp_data, reduction = "umap", label = TRUE) + 
    ggtitle(paste("Seurat")) + 
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          legend.title = element_blank())  
          
  p2 <- SpatialDimPlot(sp_data, label = TRUE, label.size = 3) + 
    ggtitle(paste("Seurat (ARI =", round(metrics_df$ARI[1], 2), ")")) + 
    theme(plot.title = element_text(hjust = 0.5, size = 16),
          legend.title = element_blank())
  
  
  ggsave(file.path(dir.output, 'umap.pdf'), 
         plot = p1, width = 6, height = 6, dpi = 300)
  ggsave(file.path(dir.output, 'clustering.pdf'), 
         plot = p2, width = 6, height = 6, dpi = 300)
  
  # Save reduced dimension data
  write.csv(sp_data@reductions$pca@cell.embeddings,
            file = file.path(dir.output, 'low_dim_data.csv'),
            row.names = TRUE)
  
  # Save metadata
  write.csv(sp_data@meta.data,
            file = file.path(dir.output, 'cell_metadata.csv'),
            row.names = TRUE)
  
  # Save UMAP coordinates
  umap_coords <- as.data.frame(sp_data@reductions$umap@cell.embeddings)
  umap_coords$spot_id <- rownames(umap_coords)
  write.csv(umap_coords,
            file = file.path(dir.output, "spatial_umap_coords.csv"),
            row.names = FALSE)
  
}

data_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/data/DLPFC")
save_root <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/Results")

metrics_list <- list()

for (seed in seeds) {
  cat("\n==============================\n")
  cat("RUNNING SEED:", seed, "\n")
  cat("==============================\n")

  set.seed(seed)

  # Loop through all batches
  for (sample.name in names(batch_cluster_map)) {
    cat("Processing batch:", sample.name, "\n")
    n_clusters <- batch_cluster_map[[sample.name]]

    # Define input and output directories
    dir.input <- file.path(data_path, sample.name)
    dir.output <- file.path(save_root, as.character(seed), "DLPFC", "Seurat", sample.name)
    if (!dir.exists(dir.output)) {
      dir.create(dir.output, recursive = TRUE)
    }

    # Start timing and memory tracking
    start_time <- Sys.time()
    gc(reset = T)

    # Load spatial transcriptomics data
    sp_data <- tryCatch({
      Load10X_Spatial(dir.input, filename = "filtered_feature_bc_matrix.h5")
    }, error = function(e) {
      stop("Error loading spatial data for batch ", sample.name, ": ", e$message)
    })

    # Load metadata and add to the Seurat object
    df_meta <- read.table(file.path(dir.input, 'metadata.tsv'))
    sp_data <- AddMetaData(sp_data, metadata = df_meta$layer_guess, col.name = 'layer_guess')

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
      for (resolution in seq(1, 0.1, by = -0.01)) {
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


    execution_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))  # Time in seconds

    memInfo1 <- gc()
    # memInfo1[11]
    # memInfo1[12]

    gc(reset = TRUE)
    memInfo2 <- gc()
    # memInfo2[11]
    # memInfo2[12]
    memory_usage <- memInfo1["Vcells", ncol(memInfo1)] - memInfo2["Vcells", ncol(memInfo2)]  # Memory usage in MB
    print(memory_usage)


    # Evaluate clustering performance
    gt <- sp_data@meta.data$layer_guess
    pred <- sp_data@meta.data$seurat_clusters
    pca_data <- sp_data@reductions$pca@cell.embeddings
    metrics <- calculate_metrics(gt, pred, pca_data)
    cat("ARI for batch", sample.name, ":", metrics$ARI, "\n")

    # Create metrics dataframe
    metrics_df <- data.frame(
      Sample = sample.name,
      ARI = metrics$ARI,
      AMI = metrics$AMI,
      Homogeneity = metrics$Homogeneity,
      Completeness = metrics$Completeness,
      V_Measure = metrics$V_Measure,
      ASW = metrics$ASW,
      Time = execution_time,
      Memory = memory_usage
    )

    

    # Save results
    save_results(sp_data, metrics_df, dir.output)
    metrics_list[[sample.name]] <- metrics_df
  }

  metrics_df <- do.call(rbind, metrics_list)
  write.csv(metrics_df, file = file.path(save_root, as.character(seed), "DLPFC", "Seurat", "metrics.csv"), row.names = TRUE)
  cat("All batches processed successfully.\n")
}