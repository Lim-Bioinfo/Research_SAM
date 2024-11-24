import pandas as pd
import numpy as np
import gzip
import time

from sklearn.decomposition import PCA
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, RobustScaler
from tqdm import tqdm
from functools import reduce

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


def cox_regression_analysis(patient_groups, clinical_data):
    results = {}

    #cox_data = clinical_data.drop(columns=['gender'])
    clinical_data = clinical_data.copy()

    cox_data = pd.merge(clinical_data, patient_groups, how = 'left', on='case_submitter_id')
    cox_data_dummies = pd.get_dummies(cox_data[['SEX', 'OS_MONTHS', 'event', 'sam score']])
        
    control_group_name = 'sam score_sam_l'
    
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)        
        
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(cox_data_dummies, duration_col='OS_MONTHS', event_col='event')
    return cph.summary

def sam_group(pairs, result_gvb_df):
    df_dict = dict()
    group_dict = dict() 
    
    for gene1, gene2 in pairs:
        group_dict[(gene1, gene2)] = {'Existed' : [], 'No-existed' : []}
        for sample in result_gvb_df.columns:
        
            # Subset gene expression data for current sample
            sample_data = result_gvb_df[sample]

            is_gene1_inactive = sample_data[gene1] < 0.3
            is_gene2_inactive = sample_data[gene2] < 0.3
            
            # Count as inactive if either of the sam pair genes is inactive
            if is_gene1_inactive and is_gene2_inactive:
                group_dict[(gene1, gene2)]['Existed'].append(sample)
                continue
            else:
                group_dict[(gene1, gene2)]['No-existed'].append(sample)
                continue
                    
    for gene1, gene2 in pairs:
        df_dict[(gene1, gene2)] = pd.DataFrame(list(group_dict[(gene1, gene2)].items()), columns=['Sample', 'Pair existed'])
        
    return df_dict

if __name__=="__main__":
    ## Input data
    print("Open data")
    start_time = time.time()
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    maf_df = pd.read_csv('mel_dfci_2019_cBioPortal_annotated_mutation_20231031.tsv.gz', compression = "gzip", sep = '\t', low_memory=False)
    clinical_patient_df = pd.read_csv("../Input/mel_dfci_2019/data_clinical_patient.txt", sep = "\t")
    clinical_sample_df = pd.read_csv("../Input/mel_dfci_2019/data_clinical_sample.txt", sep = "\t")

    ## Get SKCM SAM pairs
    sam_dict = dict()
    sam_hr_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            sam_hr_dict[sam_df.iloc[i, 0]] = dict()
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 
        
    for i in range(0, len(sam_df.index)):
        sam_hr_dict[sam_df.iloc[i, 0]][tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))] = sam_df.iloc[i, 3]
    
    ## Merge clinical sampe and clinical patient data
    cancer_type = 'SKCM'
    clinical_df = pd.merge(clinical_patient_df, clinical_sample_df, on='PATIENT_ID', how = 'left')
    clinical_df = clinical_df.drop_duplicates()
    clinical_df = clinical_df[clinical_df['SOMATIC_STATUS'] == 'Matched']
    clinical_df = clinical_df[clinical_df['PRIMARY_TYPE'] == 'skin']
    
    clinical_df = clinical_df[['PATIENT_ID','SAMPLE_ID', 'SEX', 'OS_STATUS', 'OS_MONTHS']]
    
    ## Data preprocessing
    maf_df['case_submitter_id'] = maf_df['Tumor_Sample_Barcode']
    merged_df = pd.merge(maf_df, clinical_df, how='left', left_on='case_submitter_id', right_on='SAMPLE_ID')

    ## Filter variants for PASS and LOF mutation

    merged_df = merged_df[((merged_df['Variant_Classification']=="Missense_Mutation") | (merged_df['Variant_Classification']=="Nonsense_Mutation")
            | (merged_df['Variant_Classification']=="Frame_Shift_Del") | (merged_df['Variant_Classification']=="Frame_Shift_Ins") 
            | (merged_df['Variant_Classification']=="Splice_Site"))]

    temp_cancer_df = merged_df.reset_index(drop=True)
    ## Include patient that has all clinical information
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['SEX'] == 'Female') | (temp_cancer_df['SEX'] == 'Male')]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['OS_STATUS'] == '0:LIVING') | (temp_cancer_df['OS_STATUS'] == '1:DECEASED')]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['OS_MONTHS'] >= 0)]
    temp_cancer_df = temp_cancer_df.reset_index(drop=True)
        
    temp_cancer_df['event'] = temp_cancer_df['OS_STATUS'].apply(lambda x: True if x == '1:DECEASED' else False)
    print('Patient', len(set(temp_cancer_df['case_submitter_id'])))

    cancer_df = temp_cancer_df.reset_index(drop=True)
    result_gvb_df = calculate_gvb_per_patient(cancer_df)

    survival_group = sam_group(sam_dict['SKCM'], result_gvb_df)


    ## Calculate SAM scores and do survival analysis
    sam_score_dict = dict()

    for cancer_type in tqdm(['SKCM'], desc = "Cancer type"):
        if len(survival_group) > 0:
            sam_score_dict[cancer_type] = dict()
            cancer_patients = set(result_gvb_df.columns)
        
            for patient in cancer_patients:
                for gene_pair in survival_group.keys():
                    hr = sam_hr_dict[cancer_type][gene_pair]
                    if patient in survival_group[gene_pair].iloc[0, 1]:
                        if patient not in sam_score_dict[cancer_type].keys():
                            sam_score_dict[cancer_type][patient] = 1/hr
                            continue
                        else:
                            sam_score_dict[cancer_type][patient] += 1/hr           
                            continue   
                        
            for patient in cancer_patients:
                if patient not in sam_score_dict[cancer_type].keys():
                    sam_score_dict[cancer_type][patient] = 0
                    continue

    result_dict = dict()
    for cancer_type in tqdm(sam_score_dict.keys(), desc='Cancer type'):
    
        scores = list(sam_score_dict[cancer_type].values())
        thres_mean = np.mean(scores)

        patient_labels = []
    
        for patient, score in sam_score_dict[cancer_type].items():
            if score > thres_mean:
                label = 'sam_h'
            if score < thres_mean:
                label = 'sam_l'

            patient_labels.append((patient, label))

    classification_df = pd.DataFrame(patient_labels, columns=['case_submitter_id', 'sam score'])
    cancer_clinical_data = cancer_df[['case_submitter_id', 'SEX', 'OS_MONTHS', 'event']]
    cancer_clinical_data = cancer_clinical_data.drop_duplicates(ignore_index = True)
    result_df = cox_regression_analysis(classification_df, cancer_clinical_data)
    hazard_ratio = result_df.loc['sam score_sam_h']['exp(coef)']
    p_value = result_df.loc['sam score_sam_h']['p']

    ## Save kaplan-meier curve
    cancer_clinical_data = cancer_df[['case_submitter_id','SEX', 'OS_MONTHS', 'event']]
    cancer_clinical_data = cancer_clinical_data.drop_duplicates(ignore_index = True)
    temp_df = pd.merge(cancer_clinical_data, result_dict[cancer_type], how='right', on='case_submitter_id')
    temp_df = temp_df.drop_duplicates()
    kmf = KaplanMeierFitter()

    unique_gene_groups = temp_df['sam score'].unique()
    color_map = { "sam_l": "#C00000","sam_h": "#0070C0"}

    fig, ax = plt.subplots(figsize=(6, 6))

    for group in unique_gene_groups:
        group_data = temp_df[temp_df['sam score'] == group]
        n_all = len(group_data)
        n_alive = len(group_data[group_data['event'] == 0])  # Assuming 'event' is 1 for death and 0 for alive
        kmf.fit(group_data['OS_MONTHS'], event_observed=group_data['event'], label=f'Group {group} ({n_alive}/{n_all})')
        kmf.plot(ax=ax, ci_show=False, color=color_map[group])

    ax.set_title('Kaplan-Meier Survival Curves by SAM score : mel_dfci_2019')
    ax.set_xlabel('Month')
    ax.set_ylabel('Survival Probability')
    plt.legend(loc="best")
    plt.savefig("Kaplan-Meier Survival Curves in dfci 2019 by SAM score_HR_%s_P-value_%s.svg" % (hazard_ratio, p_value), dpi = 600)
    plt.show()