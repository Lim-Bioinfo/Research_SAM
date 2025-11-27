import pandas as pd
import numpy as np
import gzip
import time

from lifelines.plotting import add_at_risk_counts
from collections import Counter
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt
from tqdm import tqdm

def sam_score_genomic(gvb_df, pairs, T=0.2):
    """
    Compute SAM scores from a GVB matrix using SAM gene pairs.

    For each sample, we define:
    (1) A gene is "impaired" if its GVB is below the genomic threshold T.
    (2) A pair (a, b) is "co-impaired" in a sample if both a and b are impaired (GVB < T) in that sample.

    Metrics returned per sample: SAM_score
    (1) Number of SAM pairs that are co-impaired in that patient.
    """
    gvb = gvb_df.copy()
    gvb.index = gvb.index.str.upper()

    # Keep only pairs where both genes are present in the GVB matrix
    P = [(a, b) for (a, b) in pairs if a in gvb.index and b in gvb.index]

    # Degree of each gene across all SAM pairs (for down-weighting hubs)
    deg = Counter([g for p in P for g in p])

    sam_cnt = np.zeros(gvb.shape[1], dtype=int)   # Unweighted co-impaired pair count
    
    # Pair-based scores
    for (a, b) in P:
        # Mask where both genes a and b are impaired (GVB < T)
        mask = (gvb.loc[a] < T) & (gvb.loc[b] < T)

        sam_cnt += mask.values.astype(int)

    return pd.DataFrame({"Patient ID": gvb.columns,"SAM score": sam_cnt})

def calculate_gvb_per_patient(df):
    # Create a copy of the dataframe
    df = df.copy()
    lof_mutations = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']

    # Filter the dataframe for genes with loss of function mutations and set their GVB score to 0
    lof_df = df[df['Variant_Classification'].isin(lof_mutations)]
    lof_df = lof_df[['Hugo_Symbol', 'case_submitter_id']].drop_duplicates()
    lof_df['GVB_score'] = 1e-8

    # Filter the dataframe to include only missense mutations and the required columns
    df = df[df['Variant_Classification'] == 'Missense_Mutation']
    filtered_df = df[['Hugo_Symbol', 'SIFT_score', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2', 'case_submitter_id']].copy()
    filtered_df['SIFT_score'] = pd.to_numeric(filtered_df['SIFT_score'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    filtered_df.dropna(subset=['SIFT_score'], inplace=True)
    filtered_df.loc[(filtered_df['SIFT_score'] == 0), 'SIFT_score'] = 1e-8

    filtered_df['homozygous'] = (filtered_df['Tumor_Seq_Allele1'] == filtered_df['Tumor_Seq_Allele2']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele1']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele2'])

    deleterious_df = filtered_df[filtered_df['SIFT_score'] < 0.7].copy()
    deleterious_df.loc[:, 'adjusted_SIFT_score'] = deleterious_df.apply(lambda row: np.power(row['SIFT_score'], 2) if row['homozygous'] else row['SIFT_score'], axis=1)
    gvb_scores_df = deleterious_df.groupby(['case_submitter_id', 'Hugo_Symbol'], group_keys=True)['adjusted_SIFT_score'].apply(lambda x: np.power(np.prod(x), 1/len(x)) if len(x) > 0 else 1).reset_index()
    gvb_scores_df.rename(columns={'adjusted_SIFT_score':'GVB_score'}, inplace=True)
    combined_df = pd.concat([gvb_scores_df, lof_df], ignore_index=True)
    gvb_scores_pivot = combined_df.pivot_table(index='Hugo_Symbol', columns='case_submitter_id', values='GVB_score').fillna(1)

    missing_genes = set(df['Hugo_Symbol']).difference(gvb_scores_pivot.index)
    for gene in missing_genes:
        gvb_scores_pivot.loc[gene] = 1

    gvb_scores_pivot = gvb_scores_pivot.sort_index()
    return gvb_scores_pivot

def count_low_gvb_per_gene(gvb_scores_pivot, gvb_threshold):
    low_gvb_counts = {}

    for gene in gvb_scores_pivot.index:
        low_gvb_count = (gvb_scores_pivot.loc[gene] < gvb_threshold).sum()
        low_gvb_counts[gene] = low_gvb_count

    return low_gvb_counts


def cox_regression_analysis(clinical_data, time, event):
    results = {}
    cox_data_dummies = pd.get_dummies(clinical_data[['SEX', time, event, 'SAM group']])

    control_group_name = 'SAM group_SAM-L'
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)     
    
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data_dummies, duration_col=time, event_col=event)
    return cph.summary



if __name__=="__main__":
    ## Input data
    print("Open data")
    start_time = time.time()
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    maf_df = pd.read_csv('.../Input/Liu_cBioPortal_annotated_mutation.tsv.gz', compression = "gzip", sep = '\t', low_memory=False)
    clinical_patient_df = pd.read_csv("../Input/Liu_data_clinical_patient.txt", sep = "\t")
    clinical_sample_df = pd.read_csv("../Input/Liu_data_clinical_sample.txt", sep = "\t")

    ## Get SKCM SAM pairs
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 
    
    ## Merge clinical sampe and clinical patient data
    cancer_type = 'SKCM'
    gvb_threshold = 0.2
    
    time = 'OS_MONTHS' #OS_MONTHS	PFS_MONTHS
    event = 'OS_STATUS' #OS_STATUS PFS_STATUS

    clinical_df = pd.merge(clinical_patient_df, clinical_sample_df, on='PATIENT_ID', how = 'left')
    clinical_df = clinical_df.drop_duplicates()
    clinical_df = clinical_df[clinical_df['SOMATIC_STATUS'] == 'Matched']
    clinical_df = clinical_df[clinical_df['PRIMARY_TYPE'] == 'skin']
    
    ## Data preprocessing
    maf_df['case_submitter_id'] = maf_df['Tumor_Sample_Barcode']
    merged_df = pd.merge(maf_df, clinical_df, how='left', left_on='case_submitter_id', right_on='SAMPLE_ID')

    ## Filter variants for PASS and LOF mutation

    merged_df = merged_df[((merged_df['Variant_Classification']=="Missense_Mutation") | (merged_df['Variant_Classification']=="Nonsense_Mutation")
            | (merged_df['Variant_Classification']=="Frame_Shift_Del") | (merged_df['Variant_Classification']=="Frame_Shift_Ins") 
            | (merged_df['Variant_Classification']=="Splice_Site"))]

    temp_cancer_df = merged_df.reset_index(drop=True)
    cancer_clinical_data = temp_cancer_df[['case_submitter_id', 'SEX', time, event]].drop_duplicates(ignore_index = True)
    result_gvb_df = calculate_gvb_per_patient(temp_cancer_df)
    improved = sam_score_genomic(result_gvb_df, input_pair)

    scores = list(improved['SAM score'])
    thresh_mean = np.mean(scores)
    patient_labels = []
    for i in range(0, len(improved)):
        if improved.iloc[i, 1] > thresh_mean:#thresh_mean:
            label_type = 'SAM-H'
        if improved.iloc[i, 1] < thresh_mean:#thresh_mean:
            label_type = 'SAM-L'
        patient_labels.append((improved.iloc[i, 0], improved.iloc[i, 1], label_type))
            
    classification_df = pd.DataFrame(patient_labels, columns=['Patient_ID', 'SAM score', 'SAM group'])
    result_df = pd.merge(classification_df, cancer_clinical_data, how='inner', left_on='Patient_ID', right_on = 'case_submitter_id')    

    result_clinical_df = result_df[['Patient_ID', 'SEX', time, event, 'SAM group']]
    result_clinical_df = result_clinical_df[(result_clinical_df['SEX'] == 'Female') | (result_clinical_df'SEX'] == 'Male')]
    result_clinical_df = result_clinical_df[(result_clinical_df[time] >= 0)]

    if event == 'OS_STATUS':
        result_clinical_df = result_clinical_df[(result_clinical_df[event] == '0:LIVING') | (result_clinical_df[event] == '1:DECEASED')]
        result_clinical_df['event'] = cancer_df[event].apply(lambda x: True if x == '1:DECEASED' else False)
    if event == 'PFS_STATUS':
        result_clinical_df = result_clinical_df[(result_clinical_df[event] == '0:CENSORED') | (result_clinical_df[event] == '1:PROGRESSION')]
        result_clinical_df['event'] = result_clinical_df[event].apply(lambda x: True if x == '1:PROGRESSION' else False)

    result_cox_df = cox_regression_analysis(result_clinical_df, time, 'event')
    hazard_ratio = result_cox_df.loc['SAM group_SAM-H']['exp(coef)']
    p_value = result_cox_df.loc['SAM group_SAM-H']['p']

    ## Draw kaplan-meier curve
    temp_df = result_clinical_df
    order = ["SAM-L", "SAM-H"]
    tmp = temp_df[temp_df["SAM group"].isin(order)].dropna(subset=[time, 'event', "SAM group"]).copy()
    tmp["SAM group"] = pd.Categorical(tmp["SAM group"], categories=order, ordered=True)

    # Colors
    color_map = {"SAM-L": "#E0898D", "SAM-H": "#7FA5DD"}

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
    plt.savefig("Liu_et_al_dfci_2019_%s.svg" % (event), dpi = 600)
    plt.tight_layout()
    plt.show()



