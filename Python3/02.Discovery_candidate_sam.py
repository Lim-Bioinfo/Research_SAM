import gzip
import time
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from lifelines import CoxPHFitter
from tqdm import tqdm


def calculate_gvb_per_patient(df, lof_type):
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

def finding_sam_pairs(gvb_scores_pivot, gene_pairs, gvb_threshold):
    result_list = []
    
    gvb_scores_np = gvb_scores_pivot.to_numpy()
    index_to_case_id = dict(enumerate(gvb_scores_pivot.columns))

    i, gene_pair_chunk = gene_pairs
    for pair in gene_pair_chunk:
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
                result_list.append(pair)

    return result_list

def finding_sam_pairs_multiprocessing(gvb_scores_pivot, gene_pairs, gvb_threshold):
    try:      
        chunk_size =  len(gene_pairs) // 10
        num_processes = 100
        result_final_list = list()
        for i in tqdm(list(range(0, len(gene_pairs), chunk_size)), total=len(list(range(0, len(gene_pairs), chunk_size))), desc = "Finding candidate SAM pairs"):
            gene_pairs_parient_chunk = gene_pairs[i:i + chunk_size]

            gene_pairs_child_chunk = dict()
            child_chunk_size = max(1, len(gene_pairs_parient_chunk) // num_processes)

            for j in list(range(0, len(gene_pairs_parient_chunk), child_chunk_size)):
                gene_pairs_child_chunk[j] = gene_pairs_parient_chunk[j:j + child_chunk_size]

            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {executor.submit(finding_sam_pairs, gvb_scores_pivot, (key, value), gvb_threshold): (key, value) for key, value in gene_pairs_child_chunk.items()}
                for future in as_completed(futures):
                    result_final_list.extend(future.result())
                
        return result_final_list                

    except Exception as e:
        print("An error occurred: ", e)


def cox_regression_analysis_single(gene_pair_chunk_dict, gvb_scores_pivot, clinical_data, gvb_threshold, penalizer_value, chunk_size):
    result_lines = []
    i, gene_pair_chunk = gene_pair_chunk_dict
    patient_groups_chunk = group_patients_by_gvb(gvb_scores_pivot, gene_pair_chunk, gvb_threshold)

    for pair, existing_groups in patient_groups_chunk.items():
        clinical_data = clinical_data.copy()

        cox_data = pd.merge(clinical_data, existing_groups, how='left', on='case_submitter_id')
        cox_data_dummies = pd.get_dummies(cox_data[['gender', 'race', 'age_at_initial_pathologic_diagnosis', 'OS.time', 'event', 'gene_group']])

        # Drop the reference group (control group) dummy variable
        control_group_name = 'gene_group_Neither %s nor %s' % (pair[0], pair[1])
        if control_group_name in cox_data_dummies.columns:
            cox_data_dummies.drop([control_group_name], axis=1, inplace=True)

        convergence_achieved = False
        
        try:
            cph = CoxPHFitter(penalizer=penalizer_value)
            cph.fit(cox_data_dummies, duration_col='OS.time', event_col='event')
            cph_result = cph.summary
            corrected_p_vals = multipletests(cph_result["p"], method="bonferroni")[1]
            fdr_vals = multipletests(cph_result["p"], method="fdr_bh")[1]
            cph_result["corrected_p (bonferroni)"] = corrected_p_vals
            cph_result["FDR_BH"] = fdr_vals
            convergence_achieved = True
        except:
            continue

        result_line = '%s,%s,' % (pair[0], pair[1])
        for test_group in ['Only %s' % (pair[0]), 'Only %s' % (pair[1]), '%s and %s' % (pair[0], pair[1])]:
            test_group_name = 'gene_group_' + test_group
            if test_group_name in cph_result.index:
                result_line += ','.join(map(str, [cph.hazard_ratios_[test_group_name], cph_result.loc[test_group_name]['p'], cph_result.loc[test_group_name]['corrected_p (bonferroni)'], cph_result.loc[test_group_name]['FDR_BH']]))
                if 'Only' in test_group_name:
                    result_line += ','
                    continue
                else:
                    result_line += '\n'
                    continue

        result_lines.append(result_line)

    return result_lines


def cox_regression_analysis_multiprocessing(gene_pairs, gvb_scores_pivot, clinical_data, num_processes, cancer_type, chunk_size, gvb_threshold, penalizer_value, lof_type):
    try:
        with gzip.open('../Result/penalizer_%s/Result_%s_SAM_%s_%s_LOF_%s.csv.gz' % (penalizer_value, cancer_type, gvb_threshold, penalizer_value, lof_type), 'wt') as f:
            f.write("Gene1,Gene2,HazardRatio(Only Gene1),P-value(Only Gene1),Corrected P-value_Bonferroni(Only Gene1),FDR_BH(Only Gene1),HazardRatio(Only Gene2),P-value(Only Gene2),Corrected P-value_Bonferroni(Only Gene2),FDR_BH(Only Gene2),HazardRatio(Both genes),P-value(Both genes),Corrected P-value_Bonferroni(Both genes),FDR_BH(Both genes)\n")        

            for i in tqdm(list(range(0, len(gene_pairs), chunk_size)), total=len(list(range(0, len(gene_pairs), chunk_size))), desc = "Filtering candidate SAM pairs"):
                gene_pairs_parent_chunk = gene_pairs[i:i + chunk_size]

                gene_pairs_child_chunk = dict()
                child_chunk_size = max(1, len(gene_pairs_parent_chunk) // num_processes)

                for j in list(range(0, len(gene_pairs_parent_chunk), child_chunk_size)):
                    gene_pairs_child_chunk[j] = gene_pairs_parent_chunk[j:j + child_chunk_size]

                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    futures = {executor.submit(cox_regression_analysis_single, (key, value), gvb_scores_pivot, clinical_data, gvb_threshold, penalizer_value, child_chunk_size): (key, value) for key, value in gene_pairs_child_chunk.items()}
                    result_lines_lists = [future.result() for future in list(futures)]
                    for result_lines in result_lines_lists:
                        f.write(''.join(result_lines))    

    except Exception as e:
        print("An error occurred: ", e)

def read_compressed_results(cancer_type, gvb_threshold, penalizer_value, lof_type):
    data = pd.read_csv('../Result/penalizer_%s/Result_%s_SAM_%s_%s_LOF_%s.csv.gz' % (penalizer_value, cancer_type, gvb_threshold, penalizer_value, lof_type), compression='gzip')
    return data         

if __name__=="__main__":
    # Input data
    print("Open data")
    start_time = time.time()
    maf_df = pd.read_table('../Input/mc3.v0.2.8.PUBLIC.maf.gz', low_memory = False)
    clinical_df = pd.read_excel("../Input/TCGA-Clinical Data Resource (CDR) Outcome.xlsx")
    clinical_df = clinical_df[['bcr_patient_barcode', 'type', 'gender', 'race', 'age_at_initial_pathologic_diagnosis', 'vital_status', 'OS', 'OS.time']]
    cancer_type = str(input("Cancer type : "))
    max_process_num = int(input("The number of process : "))
    gvb_threshold = float(input("GVB threshold : "))
    penalizer_value = float(input("Penalizer value for Cox regression : "))
    lof_type = input("Loss-of-function definition, (1). GDC, (2). BMC, (3). NotBMC : ")

    # Data preprocessing
    maf_df['case_submitter_id'] = maf_df['Tumor_Sample_Barcode'].str[:12]

    temp_gene_df = maf_df[['HGNC_ID', 'Hugo_Symbol']][maf_df['HGNC_ID'] != "."]
    temp_gene_df['HGNC_ID'] = temp_gene_df['HGNC_ID'].astype(int)
    temp_gene_df = temp_gene_df.drop_duplicates(ignore_index = True)

    merged_df = pd.merge(maf_df, clinical_df, how='left', left_on='case_submitter_id', right_on='bcr_patient_barcode')

    # Filter variants for PASS and LOF mutation
    merged_df = merged_df[(merged_df['FILTER']=="PASS") & ((merged_df['Variant_Classification']=="Missense_Mutation") | (merged_df['Variant_Classification']=="Nonsense_Mutation")
        | (merged_df['Variant_Classification']=="Frame_Shift_Del") | (merged_df['Variant_Classification']=="Frame_Shift_Ins") 
        | (merged_df['Variant_Classification']=="Splice_Site"))] 

    merged_df = merged_df.reset_index(drop=True)

    # Include patient that has all clinical information
    temp_cancer_df = merged_df[merged_df['type'] != 'TGCT']
    temp_cancer_df = temp_cancer_df[temp_cancer_df['age_at_initial_pathologic_diagnosis'] >= 0]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['vital_status'] == 'Alive') | (temp_cancer_df['vital_status'] == 'Dead')]
    temp_cancer_df = temp_cancer_df[(temp_cancer_df['OS.time'] >= 0)]
    temp_cancer_df = temp_cancer_df.reset_index(drop=True)

    # Make attributes : overall survival time, death event, gender
    temp_cancer_df.loc[temp_cancer_df['race'].isin(['[Not Applicable]', '[Not Evaluated]', '[Unknown]']), 'race'] = 'UNKNOWN'
    temp_cancer_df.loc[~temp_cancer_df['race'].isin(['WHITE', 'BLACK OR AFRICAN AMERICAN', 'UNKNOWN']), 'race'] = 'OTHER' 
    temp_cancer_df['event'] = temp_cancer_df['vital_status'].apply(lambda x: True if x == 'Dead' else False)

    # Specific cancer
    chunk_split_size = 25
    print("Data for %s" % (cancer_type))
    if cancer_type in list(temp_cancer_df['type']):
        cancer_df = temp_cancer_df[temp_cancer_df['type'] == cancer_type]
    else:
        if cancer_type in ['COADREAD']:
            cancer_df = temp_cancer_df[(temp_cancer_df['type'] == 'COAD') | (temp_cancer_df['type'] == 'READ')]

    cancer_df = cancer_df.reset_index(drop=True)

    # Calculate GVB per patients 
    print("Calculate GVB score")
    result_gvb_df = calculate_gvb_per_patient(cancer_df, lof_type)

    # Finding candiate genes from GVB dataframe
    low_gvb_counts = count_low_gvb_per_gene(result_gvb_df, gvb_threshold)
    low_gvb_counts_df = pd.DataFrame.from_dict(low_gvb_counts, orient='index', columns=['Low_GVB_Count'])
    candidate_genes = list(low_gvb_counts_df[low_gvb_counts_df['Low_GVB_Count'] >= 1].index)
    candidate_gene_pairs = list(combinations(candidate_genes, 2))
    print("The number of all candidates : %s" % (len(candidate_gene_pairs)))

    # Grouping patients by candidate gene pair
    cancer_clinical_data = cancer_df[['case_submitter_id', 'gender', 'race', 'age_at_initial_pathologic_diagnosis', 'OS.time', 'event']]
    cancer_clinical_data = cancer_clinical_data.drop_duplicates(ignore_index = True)
    
	#Start multiprocessing
    print('Starting multiprocessing // ', time.ctime())

    candidate_gene_pairs = finding_sam_pairs_multiprocessing(result_gvb_df, candidate_gene_pairs, gvb_threshold)
    print("The number of real candidate : %s" % (len(candidate_gene_pairs)))
    cox_regression_analysis_multiprocessing(candidate_gene_pairs, result_gvb_df, cancer_clinical_data, max_process_num, cancer_type, len(candidate_gene_pairs)//chunk_split_size, gvb_threshold, penalizer_value, lof_type)

    test_df = read_compressed_results(cancer_type, gvb_threshold, penalizer_value, lof_type)

    print("The volume of dataframe : %s" % (len(test_df.index)))
    print('Process complete! // ', time.ctime())
    print("Elapsed time: %s seconds" % (time.time() - start_time))
    

