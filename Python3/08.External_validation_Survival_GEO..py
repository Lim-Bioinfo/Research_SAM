import pandas as pd
import numpy as np
import gzip
import time

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
    cox_data_dummies = pd.get_dummies(clinical_data[['SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'event', 'sam score']])
        
    control_group_name = 'sam score_sam_l'
    
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)        
        
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data_dummies, duration_col='OS (Weeks)', event_col='event')
    return cph.summary


def sam_group(pairs, expression_df, threshold):
    df_dict = dict()
    group_dict = dict() 
    
    for gene1, gene2 in pairs:
        if (gene1 in expression_df.index) and (gene2 in expression_df.index):
            group_dict[(gene1, gene2)] = {'Existed' : [], 'No-existed' : []}
            for sample in expression_df.columns:
        
                # Subset gene expression data for current sample
                sample_data = expression_df[sample]
            
                is_gene1_inactive = sample_data[gene1] < sample_data.quantile(threshold)
                is_gene2_inactive = sample_data[gene2] < sample_data.quantile(threshold)
            
                # Count as inactive if either of the sam pair genes is inactive
                if is_gene1_inactive and is_gene2_inactive:
                    group_dict[(gene1, gene2)]['Existed'].append(sample)
                    continue
                else:
                    group_dict[(gene1, gene2)]['No-existed'].append(sample)
                    continue    

    for gene1, gene2 in group_dict.keys():
        df_dict[(gene1, gene2)] = pd.DataFrame(list(group_dict[(gene1, gene2)].items()), columns=['Sample', 'Pair existed'])
        
    return df_dict

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_S.xlsx") # Supplementary Data S3
    df1 = pd.read_csv("../Input/GSE22153_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE22154_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE54467_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE59455_series_matrix_collapsed_to_symbols.gct", sep = "\t")

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
    clinical_df = clinical_df[(clinical_df['AGE (Initial diagnosis)'] >= 0)]
    clinical_df = clinical_df[(clinical_df['OS (Weeks)'] >= 0)]
    clinical_df = clinical_df[(clinical_df['SEX'] == 'Female') | (clinical_df['SEX'] == 'Male')]
    clinical_df = clinical_df[(clinical_df['STATUS'] == 'Deceased') | (clinical_df['STATUS'] == 'Alive')]

    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)             
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy()

    shift_amount = 1 - df1_copied.min().min()
    df1_copied_shifted = df1_copied + shift_amount
    df1_copied = np.log2(df1_copied_shifted+1)
    df2_copied = np.log2(df2_copied+1)
    df4_copied = np.log2(df4_copied+1)
                                 
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)
    df_merged = df_merged[list(clinical_df['SAMPLE_ID'])]

    batches = ['GSE22153']*len(clinical_df[clinical_df['GSE'] == 'GSE22153']) + ['GSE22154']*len(clinical_df[clinical_df['GSE'] == 'GSE22154']) + ['GSE54467']*len(clinical_df[clinical_df['GSE'] == 'GSE54467']) + ['GSE59455']*len(clinical_df[clinical_df['GSE'] == 'GSE59455'])
    df_corrected = pycombat(df_merged, batches)

    ## Calculate SAM score and survival analysis
    result_dict = sam_group(sam_dict['SKCM'], df_corrected , 0.3)
    sam_score_dict = dict()

    for patient in df_corrected.columns:
        for sam_pair in result_dict.keys():
                
            if patient in result_dict[sam_pair].iloc[0, 1]:
                if patient not in sam_score_dict.keys():
                    sam_score_dict[patient] = 1
                    continue
                else:
                    sam_score_dict[patient] += 1
                    continue      
                        

    for patient in df_corrected.columns:
        if patient not in sam_score_dict.keys():
            sam_score_dict[patient] = 0
            continue

    scores = list(sam_score_dict.values())
    thresh_mean = np.mean(scores)
    patient_labels = []
    
    for patient, score in sam_score_dict.items():
        if score > thresh_mean:
            label_type = 'sam_h'
        if score < thresh_mean:
            label_type = 'sam_l'

        patient_labels.append((patient, label_type))

    classification_df = pd.DataFrame(patient_labels, columns=['case_submitter_id', 'sam score'])
    classification_df = pd.merge(clinical_df, classification_df, how='left', left_on='SAMPLE_ID', right_on='case_submitter_id')
    classification_df.dropna(subset=['sam score'], inplace=True)
    classification_df['event'] = classification_df['STATUS'].apply(lambda x: True if x == 'Deceased' else False)
    
    cancer_clinical_data = classification_df[['case_submitter_id', 'SEX', 'AGE (Initial diagnosis)', 'OS (Weeks)', 'event', 'sam score']]
    cancer_clinical_datax = cancer_clinical_data.drop_duplicates(ignore_index = True)
    
    result_df = cox_regression_analysis(cancer_clinical_data)
    hazard_ratio = result_df.loc['sam score_sam_h']['exp(coef)']
    p_value = result_df.loc['sam score_sam_h']['p']

    ## Draw kaplan-meier plot
    temp_df = cancer_clinical_data.drop_duplicates(ignore_index = True)
    kmf = KaplanMeierFitter()
    unique_gene_groups = temp_df['sam score'].unique()
    color_map = {"sam_l": "#C00000", "sam_h": "#106ab2"}

    fig, ax = plt.subplots(figsize=(6, 6))

    for group in unique_gene_groups:
        group_data = temp_df[temp_df['sam score'] == group]
        n_all = len(group_data)
        n_alive = len(group_data[group_data['event'] == 0])  # Assuming 'event' is 1 for death and 0 for alive
        kmf.fit(group_data['OS (Weeks)'], event_observed=group_data['event'], label=f'Group {group} ({n_alive}/{n_all})')
        kmf.plot(ax=ax, ci_show=False, color=color_map[group])


    ax.set_title('Meta cohort - SAM-H and SAM-L')
    ax.set_xlabel('OS (Weeks)')
    ax.set_ylabel('Survival Probability')
    plt.legend(loc="best")
    plt.savefig("Kaplan-Meier Survival Curves in GEO metacohort by SAM score_HR_%s_P-value_%s.svg" % (hazard_ratio, p_value), dpi = 600)
    plt.show()
