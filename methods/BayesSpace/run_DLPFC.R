library(BayesSpace)
library(ggplot2)
library(Seurat)
library(SingleCellExperiment)
library(mclust)
library(aricode)
library(clevr)  # For homogeneity, completeness, v-measure
library(cluster) # For silhouette score
library(sceasy)
library(reticulate)
library(scater)  # For runUMAP

use_python("~/venvs/sceasy/bin/python", required = TRUE)
py_config()


seeds <- c(42, 123, 456, 789, 2024)

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
#' @param dlpfc Processed SingleCellExperiment object
#' @param labels Cluster labels
#' @param metrics_df Metrics data frame
#' @param dir.output Output directory
save_results <- function(dlpfc, labels, metrics_df, dir.output) {
  # Save metrics
  write.csv(metrics_df, file = file.path(dir.output, 'metrics.csv'), row.names = FALSE)
  
  # Save spatial clustering plot
  # cluster_plot <- clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
  #   scale_fill_viridis_d(option = "A", labels = 1:7, name=NULL) +
  #   labs(title=paste("BayesSpace (ARI =", round(metrics_df$ARI, 2), ")", sep="")) +
  #   theme(plot.title = element_text(hjust = 0.5, size = 16))
  
  # ggsave(file.path(dir.output, 'clustering.pdf'), plot = cluster_plot,
  #        width = 6, height = 6, dpi = 300, device = "pdf")
  
  # Save reduced dimension data
  write.csv(reducedDim(dlpfc, "PCA"),
            file = file.path(dir.output, 'low_dim_data.csv'),
            row.names = TRUE)
  
  # Save metadata
  write.csv(colData(dlpfc),
            file=file.path(dir.output, 'cell_metadata.csv'),
            row.names = TRUE)
  
  # Save UMAP coordinates and plot
  if (!"UMAP_neighbors15" %in% names(reducedDimNames(dlpfc))) {
    set.seed(seed)
    dlpfc <- runUMAP(dlpfc, dimred="PCA", name="UMAP_neighbors15")
  }
  umap_coords <- as.data.frame(reducedDim(dlpfc, "UMAP_neighbors15"))
  umap_coords$spot_id <- rownames(umap_coords)
  write.csv(umap_coords,
            file = file.path(dir.output, "spatial_umap_coords.csv"),
            row.names = FALSE)
  
  # umap_plot <- ggplot(umap_coords, aes(x = V1, y = V2, color = as.factor(labels))) +
  #   geom_point(size = 1.5, alpha = 0.8) +
  #   scale_color_brewer(palette = "Set1") +
  #   labs(title = "BayesSpace", x = "UMAP 1", y = "UMAP 2", color = 'Cluster') +
  #   theme(plot.title = element_text(hjust = 0.5, size = 16),
  #         panel.grid = element_blank(),
  #         panel.background = element_blank(),
  #         axis.line = element_line(color = "black"))
  
  # ggsave(file.path(dir.output, 'umap.pdf'), plot = umap_plot,
  #        width = 6, height = 6, dpi = 300, device = "pdf")

  # expression_data <- as.data.frame(as.matrix(assay(dlpfc, "counts")))
  # write.table(t(expression_data), 
  #           file = file.path(dir.output, "expression_matrix.tsv"), 
  #           sep = "\t", quote = FALSE, col.names = NA)
}


data_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/data/DLPFC")
save_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/Results")
h5ad_path <- file.path("/Users/toanly/Downloads/Spatial-Transcriptomics-Benchmark/data/DLPFC12")

metrics_list <- list()

# dlpfc_all <- spatialLIBD::fetch_data(type="sce")

for (seed in seeds) {
  cat("\n==============================\n")
  cat("RUNNING SEED:", seed, "\n")
  cat("==============================\n")

  set.seed(seed)

  for (sample.name in names(batch_cluster_map)) {
    cat("Processing batch:", sample.name, "\n")
    n_clusters <- batch_cluster_map[[sample.name]]

    dir.input <- file.path(data_path, sample.name)
    dir.output <- file.path(save_path, as.character(seed), "DLPFC", "BayesSpace", sample.name)

    if (!dir.exists(file.path(dir.output))) {
      dir.create(file.path(dir.output), recursive = TRUE)
    }

    start_time <- Sys.time()
    gc(reset = TRUE)

    # dlpfc <- getRDS("2020_maynard_prefrontal-cortex", sample.name) # getRDS not working anymore
    # dlpfc_temp <- read10Xh5(dir.input)
    # dlpfc_temp <- dlpfc_temp[, match(colnames(dlpfc), colnames(dlpfc_temp))]

    # match_idx <- match(dlpfc$barcode, dlpfc_temp$barcode)
    # dlpfc$pxl_col_in_fullres <- dlpfc_temp$pxl_col_in_fullres[match_idx]
    # dlpfc$pxl_row_in_fullres <- dlpfc_temp$pxl_row_in_fullres[match_idx]


    # Load data
    h5ad_file <- file.path(h5ad_path, paste0(sample.name, ".h5ad"))
    rds_file <- file.path(h5ad_path, paste0(sample.name, ".rds"))

    if (!file.exists(rds_file)) {
      sceasy::convertFormat(
        h5ad_file, 
        from="anndata", 
        to="seurat",
        outFile = rds_file,
      )
    }
    dlpfc_temp <- readRDS(rds_file)
    dlpfc <- as.SingleCellExperiment(dlpfc_temp)
    

    set.seed(seed)
    dec <- scran::modelGeneVar(dlpfc)
    top <- scran::getTopHVGs(dec, n = 2000)

    set.seed(seed)
    dlpfc <- scater::runPCA(dlpfc, subset_row=top)

    dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

    q <- n_clusters  
    d <- 15  

    set.seed(seed)
    dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium', 
                            nrep=10000, gamma=3, save.chain=FALSE)

    labels <- dlpfc$spatial.cluster
    # gt <- dlpfc$layer_guess
    gt <- dlpfc$cluster.init
    pca_data <- reducedDim(dlpfc, "PCA")

    metrics <- calculate_metrics(gt, labels, pca_data)
    cat("ARI for batch", sample.name, ":", metrics$ARI, "\n")    

    cat('Calculated metrics for', sample.name, '\n')

    # Extract execution time and memory
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

    save_results(dlpfc, labels, metrics_df, dir.output)
    metrics_list[[sample.name]] <- metrics_df
  }

  metrics_df <- do.call(rbind, metrics_list)
  write.csv(metrics_df, file = file.path(save_path, "metrics.csv"), row.names = TRUE)

}