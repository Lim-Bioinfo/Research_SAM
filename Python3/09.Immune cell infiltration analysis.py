import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import scipy.cluster.hierarchy as sch

from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, ttest_ind, shapiro
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import ListedColormap

def compare_cell_types_boxplot(dataframe, dictionary):
    """
    Plots boxplots for each cell type (index of df1_T), comparing 'Primary' vs 'Metastases' or 'SAM-H' vs 'SAM-L'
    Also performs either t-test or Mann-Whitney test for each cell type.
    
    Parameters
    ----------
    dataframe: shape (n_cell_types, n_samples) and index = cell_type, columns = sample IDs
    dictionary : {"SAM-H": [... sample IDs ...], "SAM-L": [... sample IDs ...]}
    """

    cols = list(dictionary.keys())
    col1 = dictionary[cols[0]]
    col2 = dictionary[cols[1]]
    
    # We will store results in a list of dicts to convert to DataFrame for easy plotting
    plot_data = []  # Each row => { "cell_type", "group", "value" }
    p_values = {}   # store p-value per cell type
    
    for cell_type in dataframe.index:
        # get values for SAM-H
        val_1 = dataframe.loc[cell_type, col1].dropna()
        # get values for SAM-L
        val_2 = dataframe.loc[cell_type, col2].dropna()
        
        # Prepare for plotting
        for v in val_1:
            plot_data.append({"cell_type": cell_type, "group": cols[0], "value": v})
        for v in val_2:
            plot_data.append({"cell_type": cell_type, "group": cols[1], "value": v})

    stat, pval = ttest_ind(val_1, val_2, equal_var=False, alternative="two-sided")
    p_values[cell_type] = pval
    # Convert plot_data to DataFrame
    df_plot = pd.DataFrame(plot_data)
    
    # Plot with seaborn
    plt.figure(figsize=(20, 8))
    ax = sns.boxplot(data=df_plot, x="cell_type", y="value", hue="group")
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Relative Proportion")
    ax.set_title("Comparison of Cell Types in %s vs. %s" % (cols[0], cols[1]))
    plt.tight_layout()
    plt.legend(title="Group", loc="best")
    plt.show()
    
    # Print p-values
    print("=== P-values ===")
    for ct in dataframe.index:
        print(f"{ct:25s} p-value = {p_values[ct]:.4g}")
    
    return p_values

if __name__=="__main__":
    # Load datasets
    df1 = pd.read_csv("xCell_Meta_SKCM_PM_expression_revised_xCell.txt", sep = "\t")
    df2 = pd.read_csv("xCell_Meta_SKCM_SV_expression_revised_xCell.txt", sep = "\t")
    clinical_df1 = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S4
    clinical_df2 = pd.read_excel("../Input/Clinical_Meta_SV.xlsx") #Data from Supplementary Data S5

    df1 = df1.set_index(keys='Unnamed: 0')
    df2 = df2.set_index(keys='Unnamed: 0')


    df1_sam_dict = {"SAM-H":list(clinical_df1[clinical_df1['SAM group'] == 'SAM_H']['Sample ID']), "SAM-L":list(clinical_df1[clinical_df1['SAM group'] == 'SAM_L']['Sample ID'])}
    df2_sam_dict = {"SAM-H":list(clinical_df2[clinical_df2['SAM group'] == 'SAM_H']['Sample ID']), "SAM-L":list(clinical_df2[clinical_df2['SAM group'] == 'SAM_L']['Sample ID'])}
    df1_tumor_dict = {"Primary":list(clinical_df1[clinical_df1['Sample type'] == 'Primary']['Sample ID']), "Metastases":list(clinical_df1[clinical_df1['Sample type'] == 'Metastases']['Sample ID'])}


    df1_pm_pvals = compare_cell_types_boxplot(df1, df1_tumor_dict)
    plt.savefig("Meta_PM_xCell_PvsM.svg", dpi = 500)
    df1_sam_pvals = compare_cell_types_boxplot(df1, df1_sam_dict)
    plt.savefig("Meta_PM_xCell_HvsL.svg", dpi = 500)
    df2_sam_pvals = compare_cell_types_boxplot(df2, df2_sam_dict)
    plt.savefig("Meta_SV_xCell_HvsL.svg", dpi = 500)
