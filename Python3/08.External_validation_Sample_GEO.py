import pandas as pd
import numpyn as np
import seaborn as sns

from collections import Counter
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, accuracy_score, average_precision_score, roc_curve
from functools import reduce
from combat.pycombat import pycombat
from scipy.stats import ttest_ind


def sam_score_transcriptomic(expr_df, pairs, bottom=1/3): #Compute SAM scores from a expression matrix of meta-cohort using SAM gene pairs
    """
    For each sample, we define:
    (1) A gene is "low expressed" if its expression lies in the bottom quantile (per-gene) given by `bottom` (e.g. bottom 1/3).
    (2) A pair (a, b) is "co-impaired" in a sample if both a and b are low expressed in that sample.

    Metrics returned per sample:
    (1) SAM score: Number of SAM pairs that are co-impaired in that sample.
          
    expr_df : DataFrame, Gene x sample expression matrix (rows = genes, columns = samples).
    pairs : iterable of (str, str), List of SAM gene pairs (gene symbols).
    bottom : float, default 1/3, Per-gene lower quantile used to define low expression.
    """
    expr = expr_df.copy()
    expr.index = expr.index.str.upper()

    # Keep only pairs where both genes are present in the expression data
    P = [(a, b) for (a, b) in pairs if a in expr.index and b in expr.index]

    # Degree of each gene across all SAM pairs (for down-weighting hubs)
    deg = Counter([g for p in P for g in p])

    # Per-gene expression threshold for "low" expression
    thr = expr.quantile(bottom, axis=1)

    sam_cnt = np.zeros(expr.shape[1], dtype=int)   # Co-impaired pair count

    # Pair-based scores
    for (a, b) in P:
        # Mask where both genes a and b are in the low-expression tail
        mask = (expr.loc[a] <= thr[a]) & (expr.loc[b] <= thr[b])
        sam_cnt += mask.values.astype(int)

    return pd.DataFrame({"Sample_ID": expr.columns,"SAM score": sam_cnt})
    

if __name__=="__main__":
    # Load datasets
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S5
    df1 = pd.read_csv("GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df5 = pd.read_csv("GSE65904_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    
    ## Get SKCM SAM pairs
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 


    # Assuming preprocessing and normalization are done
    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION'];del df5['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME');df5 = df5.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns);df5_list = list(df5.columns)           
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy();df5_copied = df5.copy()


    df1_copied = np.log2(df1_copied + 1)
    df2_copied = np.log2(df2_copied + 1)
    df4_copied = np.log2(df4_copied + 1)
    df5_copied = np.log2(df5_copied + 1)

    # Merge dataframes
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied, df5_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)

    batches = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns) + ['GSE65904']*len(df5.columns)
    df_corrected = pycombat(df_merged, batches)

    primary = label['GSE7553']['Primary'] + label['GSE8401']['Primary'] + label['GSE15605']['Primary'] + label['GSE46517']['Primary']
    metastases = label['GSE7553']['Metastases'] + label['GSE8401']['Metastases'] + label['GSE15605']['Metastases'] + label['GSE46517']['Metastases']
  
    label_dict = {'Primary' : primary, 'Metastases' : metastases}
    df_corrected = df_corrected[sorted(primary + metastases)]

    
    ## Calculate SAM score and survival analysis
    improved = sam_score_transcriptomic(df_corrected, input_pair, 1/3)
    thresh_mean = np.mean(scores)
    patient_labels = []
    
    for i in range(0, len(improved)):
        if improved.iloc[i, j] > thresh_mean:#thresh_mean:
            label_type = 'SAM-H'
        if improved.iloc[i, j] < thresh_mean:#thresh_mean:
            label_type = 'SAM-L'
        patient_labels.append((improved.iloc[i, 0], improved.iloc[i, j], label_type))

    classification_df = pd.DataFrame(patient_labels, columns=['Sample ID', 'SAM score', 'SAM group'])
    result_df = pd.merge(clinical_df, classification_df, how='left', on='Sample ID')

    # Compare SAM score between primary and metastases samples
    sam_score_merged_dict = {'Primary':dict(), 'Metastases' : dict()}

    for i in range(0, len(result_df)):
        if result_df.iloc[i, 3] == 'Primary':
            sam_score_merged_dict['Primary'][result_df.iloc[i, 2]] = result_df.iloc[i, 4]
        if result_df.iloc[i, 3] == 'Metastases':
            sam_score_merged_dict['Metastases'][result_df.iloc[i, 2]] = result_df.iloc[i, 4]
            
    print(mannwhitneyu(list(sam_score_merged_dict['Primary'].values()), list(sam_score_merged_dict['Metastases'].values())))
