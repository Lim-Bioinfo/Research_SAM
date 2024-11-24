import pandas as pd
import numpy as np
import gzip

from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt
from tqdm import tqdm

def calculate_gvb_per_patient(df):
    # Create a copy of the dataframe
    df = df.copy()

    # Define the loss of function mutation types
    lof_mutations = ['Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'Splice_Site']

    # Filter the dataframe for genes with loss of function mutations and set their GVB score to 0
    lof_df = df[df['Variant_Classification'].isin(lof_mutations)]
    lof_df = lof_df[['Hugo_Symbol', 'case_submitter_id']].drop_duplicates()
    lof_df['GVB_score'] = 1e-8

    # Filter the dataframe to include only missense mutations and the required columns
    df = df[df['Variant_Classification'] == 'Missense_Mutation']
    filtered_df = df[['Hugo_Symbol', 'SIFT', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2', 'case_submitter_id']].copy()

    # Convert SIFT scores to numeric and add a small value (1e-8)
    filtered_df['SIFT'] = pd.to_numeric(filtered_df['SIFT'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    filtered_df.dropna(subset=['SIFT'], inplace=True)
    filtered_df.loc[(filtered_df['SIFT'] == 0), 'SIFT'] = 1e-8

    # Calculate zygosity
    filtered_df['homozygous'] = (filtered_df['Tumor_Seq_Allele1'] == filtered_df['Tumor_Seq_Allele2']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele1']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele2'])

    # Filter variants with SIFT score less than 0.7
    deleterious_df = filtered_df[filtered_df['SIFT'] < 0.7].copy()

    # Adjust SIFT scores based on zygosity
    deleterious_df.loc[:, 'adjusted_SIFT'] = deleterious_df.apply(lambda row: np.power(row['SIFT'], 2) if row['homozygous'] else row['SIFT'], axis=1)

    # Calculate GVB score for each gene per patient
    gvb_scores_df = deleterious_df.groupby(['case_submitter_id', 'Hugo_Symbol'], group_keys=True)['adjusted_SIFT'].apply(lambda x: np.power(np.prod(x), 1/len(x)) if len(x) > 0 else 1).reset_index()
    gvb_scores_df.rename(columns={'adjusted_SIFT':'GVB_score'}, inplace=True)

    # Combine the dataframes for loss of function mutations and missense mutations
    combined_df = pd.concat([gvb_scores_df, lof_df], ignore_index=True)

    # Pivot the dataframe to have genes as rows and patients as columns
    gvb_scores_pivot = combined_df.pivot_table(index='Hugo_Symbol', columns='case_submitter_id', values='GVB_score').fillna(1)

    # Include genes that were excluded because they had no variants with SIFT scores < 0.7 or missing SIFT scores
    missing_genes = set(df['Hugo_Symbol']).difference(gvb_scores_pivot.index)
    for gene in missing_genes:
        gvb_scores_pivot.loc[gene] = 1

    # Sort index
    gvb_scores_pivot = gvb_scores_pivot.sort_index()
    return gvb_scores_pivot

def count_low_gvb_per_gene(gvb_scores_pivot, gvb_threshold):
    low_gvb_counts = {}

    for gene in gvb_scores_pivot.index:
        low_gvb_count = (gvb_scores_pivot.loc[gene] < gvb_threshold).sum()
        low_gvb_counts[gene] = low_gvb_count

    return low_gvb_counts


def group_patients_by_gvb(gvb_scores_pivot, gene_pairs, gvb_threshold):
    result_groups = {}
    
    gvb_scores_np = gvb_scores_pivot.to_numpy()
    index_to_case_id = dict(enumerate(gvb_scores_pivot.columns))

    for pair in gene_pairs:
    #for pair in gene_pairs:
        gene1, gene2 = pair
        gene1_index = gvb_scores_pivot.index.get_loc(gene1)
        gene2_index = gvb_scores_pivot.index.get_loc(gene2)

        gene1_low_gvb = gvb_scores_np[gene1_index] < gvb_threshold
        gene2_low_gvb = gvb_scores_np[gene2_index] < gvb_threshold

        group1 = np.logical_and(gene1_low_gvb, gene2_low_gvb)
        group2 = np.logical_and(gene1_low_gvb, np.logical_not(gene2_low_gvb))
        group3 = np.logical_and(gene2_low_gvb, np.logical_not(gene1_low_gvb))
        group4 = np.logical_not(np.logical_or(group1, np.logical_or(group2, group3)))

        existing_groups = []
        if np.any(group1):
            existing_groups.append(("%s and %s" % (gene1, gene2), group1))
            if np.any(group2):
                existing_groups.append(("Only %s" % gene1, group2))
            if np.any(group3):
                existing_groups.append(("Only %s" % gene2, group3))
            if np.any(group4):
                existing_groups.append(("Neither %s nor %s" % (gene1, gene2), group4))

            if len(existing_groups) > 1:
                formatted_list = [(index_to_case_id[i], gene_group) for gene_group, mask in existing_groups for i, case_id in enumerate(mask) if case_id]
                result_groups[pair] = pd.DataFrame(formatted_list, columns=['case_submitter_id', 'gene_group'])

    return result_groups


def cox_regression_analysis(patient_groups, clinical_data):
    results = {}

    #cox_data = clinical_data.drop(columns=['gender'])
    clinical_data = clinical_data.copy()

    cox_data = pd.merge(clinical_data, patient_groups, how = 'left', on='case_submitter_id')
    cox_data_dummies = pd.get_dummies(cox_data[['gender', 'race', 'ajcc_pathologic_tumor_stage', 
                         'age_at_initial_pathologic_diagnosis', 'OS.time', 'event', 'sam score']])

    control_group_name = 'sam score_sam_l'
    if control_group_name in cox_data_dummies.columns:
        cox_data_dummies.drop([control_group_name], axis=1, inplace=True)     
    
    cph = CoxPHFitter(penalizer=0.5)
    cph.fit(cox_data_dummies, duration_col='OS.time', event_col='event')
    return cph.summary


if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    maf_df = pd.read_table('../Input/mc3.v0.2.8.PUBLIC.maf.gz', low_memory = False)
    clinical_df = pd.read_excel("../Input/TCGA-Clinical Data Resource (CDR) Outcome.xlsx")
    clinical_df = clinical_df[['bcr_patient_barcode', 'type', 'gender', 'race','ajcc_pathologic_tumor_stage', 'age_at_initial_pathologic_diagnosis', 'vital_status', 'OS', 'OS.time']]

    ## Set parameter and SAM pairs
    gvb_threshold = 0.3
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 


    ## Data preprocessing
    maf_df['case_submitter_id'] = maf_df['Tumor_Sample_Barcode'].str[:12]

    temp_gene_df = maf_df[['HGNC_ID', 'Hugo_Symbol']][maf_df['HGNC_ID'] != "."]
    temp_gene_df['HGNC_ID'] = temp_gene_df['HGNC_ID'].astype(int)
    temp_gene_df = temp_gene_df.drop_duplicates(ignore_index = True)

    merged_df = pd.merge(maf_df, clinical_df, how='left', left_on='case_submitter_id', right_on='bcr_patient_barcode')
    merged_df = merged_df[(merged_df['FILTER']=="PASS") & ((merged_df['Variant_Classification']=="Missense_Mutation") | (merged_df['Variant_Classification']=="Nonsense_Mutation")
            | (merged_df['Variant_Classification']=="Frame_Shift_Del") | (merged_df['Variant_Classification']=="Frame_Shift_Ins") 
            | (merged_df['Variant_Classification']=="Splice_Site"))] 
    merged_df = merged_df.reset_index(drop=True)

    ## Include patient that has all clinical information
    temp_cancer_df = merged_df[merged_df['type'] != 'TGCT']
    temp_cancer_df = temp_cancer_df[temp_cancer_df['age_at_initial_pathologic_diagnosis'] >= 0]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['vital_status'] == 'Alive') | (temp_cancer_df['vital_status'] == 'Dead')]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['OS.time'] >= 0)]
    temp_cancer_df = temp_cancer_df.reset_index(drop=True)

    ## Make attributes : overall survival time, death event, gender
    temp_cancer_df.loc[temp_cancer_df['race'].isin(['[Not Applicable]', '[Not Evaluated]', '[Unknown]']), 'race'] = 'UNKNOWN'
    temp_cancer_df.loc[~temp_cancer_df['race'].isin(['WHITE', 'BLACK OR AFRICAN AMERICAN', 'UNKNOWN']), 'race'] = 'OTHER'
    temp_cancer_df.loc[temp_cancer_df['ajcc_pathologic_tumor_stage'].isin(['Stage I','Stage IA','Stage IB']), 'ajcc_pathologic_tumor_stage'] = 'Stage I'
    temp_cancer_df.loc[temp_cancer_df['ajcc_pathologic_tumor_stage'].isin(['Stage II','Stage IIA','Stage IIB','Stage IIC']), 'ajcc_pathologic_tumor_stage'] = 'Stage II'
    temp_cancer_df.loc[temp_cancer_df['ajcc_pathologic_tumor_stage'].isin(['Stage III','Stage IIIA','Stage IIIB','Stage IIIC']), 'ajcc_pathologic_tumor_stage'] = 'Stage III'
    temp_cancer_df.loc[temp_cancer_df['ajcc_pathologic_tumor_stage'].isin(['Stage IV','Stage IVA','Stage IVB','Stage IVC']), 'ajcc_pathologic_tumor_stage'] = 'Stage IV'
    temp_cancer_df.loc[temp_cancer_df['ajcc_pathologic_tumor_stage'].isin(['[Not Available]','[Discrepancy]','[Not Applicable]','[Unknown]']), 'ajcc_pathologic_tumor_stage'] = 'UNKNOWN'
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['ajcc_pathologic_tumor_stage'] == 'Stage I') | (temp_cancer_df['ajcc_pathologic_tumor_stage'] == 'Stage II') | 
                                       (temp_cancer_df['ajcc_pathologic_tumor_stage'] == 'Stage III') | (temp_cancer_df['ajcc_pathologic_tumor_stage'] == 'Stage IV') | 
                                       (temp_cancer_df['ajcc_pathologic_tumor_stage'] == 'UNKNOWN')]
    temp_cancer_df['event'] = temp_cancer_df['vital_status'].apply(lambda x: True if x == 'Dead' else False)


    ## Calculate SAM score and survival analysis based on overall survival
    group_dict = dict()
    sam_score_dict = dict()
    sam_input_pair_dict = dict()

    for cancer_type in tqdm(['LUAD', 'LUSC', 'STAD', 'SKCM'], desc = "Cancer type"):
        cancer_df = temp_cancer_df[temp_cancer_df['type'] == cancer_type]
        cancer_df = cancer_df.reset_index(drop=True)
        cancer_clinical_data = cancer_df[['case_submitter_id', 'gender', 'race', 'ajcc_pathologic_tumor_stage', 'age_at_initial_pathologic_diagnosis', 'OS.time', 'event']]
        cancer_clinical_data = cancer_clinical_data.drop_duplicates(ignore_index = True)
        cancer_clinical_data = cancer_clinical_data.reset_index(drop=True)

        result_gvb_df = calculate_gvb_per_patient(cancer_df)
        group_dict[cancer_type] = group_patients_by_gvb(result_gvb_df, sam_dict[cancer_type], 0.3)

        cancer_df = temp_cancer_df[temp_cancer_df['type'] == cancer_type]
        input_pair = sam_dict[cancer_type]
        sam_score_dict[cancer_type] = dict()
    
        temp_sam_df = sam_df[sam_df['Cancer type'] == cancer_type].copy()
    
        for pair in group_dict[cancer_type].keys():
            temp_patients_df = group_dict[cancer_type][pair]
            if 'and' in temp_patients_df.iloc[i, 1]:
                genes = temp_patients_df.iloc[i, 1].split(" and ")
                if tuple(sorted(genes)) in input_pair:
                    for j in range(0, len(temp_sam_df.index)):
                        if set(genes) == set([temp_sam_df.iloc[j, 1], temp_sam_df.iloc[j, 2]]):
                            hr = temp_sam_df.iloc[j, 3]
            
                    if temp_patients_df.iloc[i, 1] not in sam_score_dict[cancer_type].keys():
                        sam_score_dict[cancer_type][temp_patients_df.iloc[i, 1]] = 1/hr
                        continue
                    else:
                        sam_score_dict[cancer_type][temp_patients_df.iloc[i, 1]] += 1/hr
                        continue

        for patient in set(cancer_df['case_submitter_id']):
            if patient not in sam_score_dict[cancer_type].keys():
                sam_score_dict[cancer_type][patient] = 0
        

        scores = list(sam_score_dict[cancer_type].values())
        thresh_median = np.median(scores)
        thresh_mean = np.mean(scores)

        patient_labels = []
    
        for patient, score in sam_score_dict[cancer_type].items():
            if score > thresh_mean:
                label = 'sam_h'
            if score < thresh_mean:
                label = 'sam_l'
            patient_labels.append((patient, label))

        classification_df = pd.DataFrame(patient_labels, columns=['case_submitter_id', 'sam score'])
        result_df = cox_regression_analysis(classification_df, cancer_clinical_data)
        hazard_ratio = result_df.loc['sam score_sam_h']['exp(coef)']
        p_value = result_df.loc['sam score_sam_h']['p']


        ## Draw survival plot
        temp_clinical_df = pd.merge(cancer_clinical_data, classification_df, how='left', on='case_submitter_id')
        kmf = KaplanMeierFitter()
        unique_gene_groups = temp_clinical_df['sam score'].unique()
        color_map = {"sam_l": "#C00000","sam_h": "#0070C0"}
        fig, ax = plt.subplots(figsize=(6, 6))

        # Iterate through the gene groups and fit the data for each group
        for group in unique_gene_groups:
            group_data = temp_clinical_df[temp_clinical_df['sam score'] == group]
            n_all = len(group_data)
            n_alive = len(group_data[group_data['event'] == 0])  # Assuming 'event' is 1 for death and 0 for alive
            kmf.fit(group_data['OS.time'], event_observed=group_data['event'], label=f'Group {group} ({n_alive}/{n_all})')
            kmf.plot(ax=ax, ci_show=False, color=color_map[group])

        # Customize the plot
        ax.set_title('Kaplan-Meier Survival Curves by SAM score, %s' % (cancer_type))
        ax.set_xlabel('Overall survival (Days)')
        ax.set_ylabel('Survival Probability')
        plt.legend(loc="best")
        plt.savefig("../Result/'Kaplan-Meier Survival Curves in %s by SAM score_HR_%s_P-value_%s.svg" % (cancer_type, hazard_ratio, p_value), dpi = 600)
        plt.show()


