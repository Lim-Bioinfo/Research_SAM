import pandas as pd
import numpy as np
import gzip
import time

from lifelines.plotting import add_at_risk_counts
from collections import Counter
from combat.pycombat import pycombat
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt
from sklearn import set_config
from tqdm import tqdm
from functools import reduce

def cox_regression_analysis(clinical_data):
    results = {}

    #cox_data = clinical_data.drop(columns=['gender'])
    clinical_data = clinical_data.copy()
    cox_data_dummies = pd.get_dummies(clinical_data[['SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'event', 'SAM group']])
        
    control_group_name = 'SAM group_SAM-L'
    
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)        
        
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data_dummies, duration_col='OS (Weeks)', event_col='event')
    return cph.summary


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
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_SV.xlsx") # Supplementary Data S4
    df1 = pd.read_csv("../Input/GSE22153_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE22154_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE54467_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE59455_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df5 = pd.read_csv("../Input/TCGA_SKCM_median_expression_improved.txt", sep = "\t")

    ## Get SKCM SAM pairs
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 


    ## Preprocessing data
    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME');df5 = df5.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)             
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy();df5_list = list(df5.columns)

    shift_amount = 1 - df1_copied.min().min()
    df1_copied_shifted = df1_copied + shift_amount
    df1_copied = np.log2(df1_copied_shifted+1)
    df2_copied = np.log2(df2_copied+1)
    df4_copied = np.log2(df4_copied+1)
                                 
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied, df5_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)
    batches = ['GSE22153']*len(df1_copied.columns) + ['GSE22154']*len(df2_copied.columns) + ['GSE54467']*len(df3_copied.columns) + ['GSE59455']*len(df4_copied.columns)+ ['TCGA']*len(df5_copied.columns)
    df_corrected = pycombat(df_merged, batches)

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

    result_df = pd.merge(clinical_df, classification_df, how='inner', on='Sample ID')
    result_cancer_df = result_df[['Sample ID', 'DATASET', 'SAM score', 'SAM group', 'SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'STATUS']]
    result_cancer_df['event'] = result_cancer_df['STATUS'].apply(lambda x: True if x == 'Deceased' else False)
    result_cancer_df = result_cancer_df.dropna()

    result_cox_df = cox_regression_analysis(result_cancer_df)
    hazard_ratio = result_df.loc['SAM group_SAM-H']['exp(coef)']
    p_value = result_df.loc['SAM group_SAM-H']['p']
    
# Draw KM-plot
    time = 'OS (Weeks)'
    temp_df = cancer_clinical_data
    order = ["SAM-L", "SAM-H"]
    tmp = temp_df[temp_df["SAM group"].isin(order)].dropna(subset=[time, 'event', "SAM group"]).copy()
    tmp["SAM group"] = pd.Categorical(tmp["SAM group"], categories=order, ordered=True)

    # Colors
    color_map = {"SAM-L": "#C00000", "SAM-H": "#106ab2"}

    # Split data
    d_L = tmp[tmp["SAM group"] == "SAM-L"]
    d_H = tmp[tmp["SAM group"] == "SAM-H"]

    # Fitters
    km_L = KaplanMeierFitter()
    km_H = KaplanMeierFitter()

    km_L.fit(durations=d_L[time], event_observed=d_L['event'], label=f"SAM-L ({int((d_L['event']==0).sum())}/{len(d_L)})")
    km_H.fit(durations=d_H[time], event_observed=d_H['event'], label=f"SAM-H ({int((d_H['event']==0).sum())}/{len(d_H)})")

    # Plot
    fig, ax = plt.subplots(figsize=(6,7))
    km_L.plot(ax=ax, ci_show=False, color=color_map["SAM-L"])
    km_H.plot(ax=ax, ci_show=False, color=color_map["SAM-H"])

    #ax.set_title(f"Kaplan–Meier Survival by SAM score, {cancer_type}")
    ax.set_xlabel(f"{time} (Days)")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1.02)
    #ax.grid(True, axis="y", alpha=0.25)

    # Add number-at-risk table
    add_at_risk_counts(km_L, km_H, ax=ax)
    plt.savefig("Meta-SV_Survival_OS.svg", dpi = 600)
    plt.tight_layout()
    plt.show()
