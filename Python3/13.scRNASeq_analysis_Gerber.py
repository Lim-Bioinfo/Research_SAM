import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from kneed import KneeLocator  # For detecting the "elbow point"


def find_optimal_neighbors_resolution(
    adata,
    pcs,
    neighbor_range,
    resolution_range,
    neighbor_step,
    resolution_step
):
    """
    Systematically tries different n_neighbors and Leiden resolutions.
    Returns (best_n_neighbors, best_resolution, best_silhouette).
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData with .obsm['X_pca'] computed.
    pcs : int
        Number of PCs to use in neighbors.
    neighbor_range : tuple
        (min_n, max_n) for n_neighbors
    resolution_range : tuple
        (min_r, max_r) for resolution
    neighbor_step : int
        Step size for n_neighbors search
    resolution_step : float
        Step size for resolution search
    
    Returns
    -------
    best_combo : (int, float)
        The best (n_neighbors, resolution) found.
    best_sil : float
        The silhouette score associated with that best combo.
    """
    best_combo = None
    best_sil = -1.0

    # Convert resolution_range to np.arange
    res_values = np.arange(resolution_range[0], resolution_range[1] + resolution_step, resolution_step)
    
    # We'll iterate over neighbors and resolution, compute silhouette, track best
    for n_nb in tqdm(range(neighbor_range[0], neighbor_range[1]+1, neighbor_step),
                     desc="Searching n_neighbors", leave=True):
        sc.pp.neighbors(adata, n_neighbors=n_nb, n_pcs=pcs)  # build KNN graph
        for res in res_values:
            sc.tl.leiden(adata, resolution=res, flavor='igraph', key_added='leiden_temp', n_iterations=10, directed=False, random_state = 452456)
            labels = adata.obs['leiden_temp'].values
            
            # Silhouette on PCA coords
            X_pca = adata.obsm['X_pca'][:, :pcs]
            sil = silhouette_score(X_pca, labels)
            
            # Track best
            if sil > best_sil:
                best_sil = sil
                best_combo = (n_nb, res)
    
    return best_combo, best_sil


def annotate_clusters_by_marker_genes(adata, marker_genes_dict, cluster_key="leiden", layer=None):
    """
    Annotate clusters based on marker gene expression.

    Parameters
    ----------
    adata : AnnData
    marker_genes_dict : dict
        Dictionary of {celltype: [marker1, marker2, ...]}
    cluster_key : str
        Key in adata.obs where cluster labels are stored (e.g., "leiden")
    layer : str or None
        If specified, use adata.layers[layer] instead of adata.X

    Returns
    -------
    Adds adata.obs["annotated_celltype"] with assigned cell type per cluster.
    Also returns the cluster → celltype mapping.
    """
    expr = adata.to_df() if layer is None else pd.DataFrame(
        adata.layers[layer], index=adata.obs_names, columns=adata.var_names)

    # Create a DataFrame for cluster-wise average expression
    cluster_ids = adata.obs[cluster_key].unique()
    cluster_means = {}

    for cluster in cluster_ids:
        cells_in_cluster = adata.obs[adata.obs[cluster_key] == cluster].index
        cluster_expr = expr.loc[cells_in_cluster]
        cluster_means[cluster] = cluster_expr.mean(axis=0)

    cluster_means_df = pd.DataFrame(cluster_means).T  # shape: clusters x genes

    # Score marker expression per cluster per cell type
    scores = {}
    for celltype, markers in marker_genes_dict.items():
        valid_markers = [g for g in markers if g in cluster_means_df.columns]
        if len(valid_markers) == 0:
            continue
        scores[celltype] = cluster_means_df[valid_markers].mean(axis=1)

    scores_df = pd.DataFrame(scores)  # shape: clusters x cell types
    cluster_to_celltype = scores_df.idxmax(axis=1)

    # Map back to cells
    adata.obs["annotated_celltype"] = adata.obs[cluster_key].map(cluster_to_celltype)

    return cluster_to_celltype

if __name__=="__main__":
    # Input data
    df = pd.read_csv("../Input/GSE81383_data_melanoma_scRNAseq_BT_2015-07-02.txt", sep="\t", index_col=0) # scRNA-Seq of 92 samples from GSE81383
    metadata_df = pd.read_csv("../Input/GSE81383_92_clinical.txt", sep = "\t") # Clinical information of 92 samples from GSE81383 in Supplementary Data S7, SPecially SAM group and Group annotated by Gerber et al.


    # Create an AnnData object
    adata = sc.AnnData(df.T)
    adata.var_names = df.index
    adata.obs_names = df.columns
    metadata_df["Sample ID"] = metadata_df["Sample ID"].astype(str)
    adata.obs = adata.obs.merge(metadata_df, left_index=True, right_on="Sample ID", how="left")
    adata.obs_names = adata.obs_names.astype(str)  # or int


    # Preprocessing data
    sc.pl.highest_expr_genes(adata, n_top=20)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=20)

    adata.var["mt"] = adata.var_names.str.startswith("MT")# mitochondrial genes, "MT-" for human, "Mt-" for mouse

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=False, percent_top=None)

    adata = adata[adata.obs.n_genes_by_counts < 9000, :]
    adata = adata[adata.obs.pct_counts_mt < 1, :].copy()


    # Normalization
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.highly_variable_genes(adata)
    adata = adata[:, adata.var["highly_variable"]].copy()

    adata.raw = adata.copy()
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(adata, max_value=10)


    # Identifying optimal parameters

    sc.tl.pca(adata)
    variance_ratios = adata.uns["pca"]["variance_ratio"]
    cumulative_variance = np.cumsum(variance_ratios)
    elbow_point = KneeLocator(range(1, len(variance_ratios) + 1), variance_ratios, curve="convex", direction="decreasing").elbow

    print(f"Optimal number of PCs (elbow method): {elbow_point}")
    optimal_pcs = elbow_point

    pcs = optimal_pcs
    best_combo, best_sil = find_optimal_neighbors_resolution(
    adata,
    pcs=pcs,
    neighbor_range=(2, 31), # 2, 21
    resolution_range=(2, 20.1), # 0.3, 50.1
    neighbor_step=1,
    resolution_step=0.05)
    print(f"Best combo: n_neighbors={best_combo[0]}, resolution={best_combo[1]}, silhouette={best_sil:.3f}")


    # Final clustering with best parameters
    n_nb_best, res_best = best_combo
    sc.pp.neighbors(adata, n_neighbors=n_nb_best, n_pcs=pcs)
    sc.tl.leiden(adata, resolution=res_best, flavor="igraph")
    sc.tl.umap(adata)

    sc.pp.neighbors(adata, n_neighbors=best_combo[0], n_pcs=optimal_pcs)
    chosen_res = best_combo[1]  # e.g. after seeing cluster counts
    sc.tl.leiden(adata, resolution=chosen_res, flavor="igraph", n_iterations=100, directed=False)

    color_dict = palette={"3": "#005493", "7": "#76D6FF", "5": "#FF8AD8", "1": "#941751", "4": "#7A81FF", "0": "#009193", "2": "#FF2600", '6': "#FF7E79"}
    sc.pl.umap(adata, color=["leiden"], palette=color_dict, title="Leiden Clusters", size = 100, save="GSE81383_clustered.svg")

    # Annotation
    marker_dict = {
        "CD14+ Mono": ["FCN1", "CD14"],
        "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
        "ID2-hi myeloid prog": ["CD14", "ID2", "VCAN", "S100A9", "CLEC12A", "KLF4", "PLAUR"],
        "cDC1": ["CLEC9A", "CADM1"],
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],  # Note: DMXL2 should be negative
    "Normoblast": ["SLC4A1", "SLC25A37", "HBB", "HBA2", "HBA1", "TFRC"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],  # Note HBM and GYPA are negative markers
    "NK": ["GNLY", "NKG7", "CD247", "GRIK4", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Lymph prog": ["VPREB1", "MME", "EBF1", "SSBP2", "BACH2", "CD79B", "IGHM", "PAX5", "PRKCE", "DNTT", "IGLL1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    "B1 B": ["MS4A1", "SSPN", "ITGB1", "EPHA4", "COL4A4", "PRDM1", "IRF4", "CD38", "XBP1", "PAX5", "BCL11A", "BLK", "IGHD", "IGHM", "ZNF215"],  # Note IGHD and IGHM are negative markers
    "Transitional B": ["MME", "CD38", "CD24", "ACSM3", "MSI2"],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    "Plasmablast": ["XBP1", "RF4", "PRDM1", "PAX5"],  # Note PAX5 is a negative marker
    "CD4+ T activated": ["CD4", "IL7R", "TRBC2", "ITGB1"],
    "CD4+ T naive": ["CD4", "IL7R", "TRBC2", "CCR7"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T activation": ["CD69", "CD38"],  # CD69 much better marker!
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
    "G/M prog": ["MPO", "BCL2", "KCNQ5", "CSF3R"],
    "HSC": ["NRIP1", "MECOM", "PROM1", "NKAIN2", "CD34"],
    "MK/E prog": ["ZNF385D", "ITGA2B", "RYR3", "PLCB1"],  # Note PLCB1 is a negative marker
    } # From https://www.sc-best-practices.org/cellular_structure/annotation.html

    marker_genes_in_data = {}
    for ct, markers in marker_dict.items():
        markers_found = []
        for marker in markers:
            if marker in adata.var.index:
                markers_found.append(marker)
        marker_genes_in_data[ct] = markers_found

    marker_genes_in_data = {k: v for k, v in marker_genes_in_data.items() if len(v) > 0}
    sc.pl.dotplot(adata, marker_genes_in_data, groupby="leiden", standard_scale="var", cmap="Greens", save="GSE81383_dotplot.svg")

    cluster_mapping = annotate_clusters_by_marker_genes(adata, marker_genes_in_data, cluster_key="leiden") # Annotate cell information to each cluster

    # Save annotated with cell type information figure
    celltype_color_dict = palette={"B1 B": "#005493", "CD16+ Mono": "#76D6FF", "Erythroblast": "#FF2600", "G/M prog": "#941751", "Lymph prog": "#7A81FF", "NK": "#009193", "pDC": "#0096FF"}
    sc.pl.umap(adata, color="annotated_celltype", palette= celltype_color_dict, size = 100, save="GSE81383_annotated.svg")

    # Save annotated with group information figure
    group_color_dict = palette={"Group1": "#941751", "Group2": "#7A81FF", "Group3": "#009193"}
    sc.pl.umap(adata, color="Group", palette=group_color_dict, title="Predicted samples in GSE81383", size = 100, save="GSE81383_grouped.svg")

    # Save annotated with predicted tumor sample information figure
    sc.pl.umap(adata, color="SAM group", palette={"SAM-H": "#0070C0", "SAM-L": "#C00000"},  title="Samples classified SAM groups in GSE81383", size = 100, save="GSE81383_predicted.svg")
