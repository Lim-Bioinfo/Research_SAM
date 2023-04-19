import concurrent
import gzip
import numpy as np
import os
import pandas as pd
import seaborn as sns
import shutil
import sys
import time

from itertools import combinations
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt
from multiprocessing import Pool, Process, Manager, Value, Lock, Queue, JoinableQueue
from scipy.stats import pearsonr, spearmanr
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sksurv.preprocessing import OneHotEncoder
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.svm import NaiveSurvivalSVM, FastSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.compare import compare_survival
from tqdm import tqdm

max_process_num = 30

def calculate_gvb(df):
    # Filter the dataframe to include only the required columns
    filtered_df = df[['Hugo_Symbol', 'SIFT', 'Tumor_Seq_Allele1', 'Tumor_Seq_Allele2', 'Match_Norm_Seq_Allele1', 'Match_Norm_Seq_Allele2']].copy()
    
    # Convert SIFT scores to numeric
    filtered_df['SIFT'] = pd.to_numeric(filtered_df['SIFT'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce') + 1e-8

    # Calculate zygosity
    filtered_df['homozygous'] = (filtered_df['Tumor_Seq_Allele1'] == filtered_df['Tumor_Seq_Allele2']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele1']) & \
                            (filtered_df['Tumor_Seq_Allele1'] != filtered_df['Match_Norm_Seq_Allele2'])

    # Filter variants with SIFT score less than 0.7
    deleterious_df = filtered_df[filtered_df['SIFT'] < 0.7].copy()

    # Adjust SIFT scores based on zygosity
    deleterious_df.loc[:, 'adjusted_SIFT'] = deleterious_df.apply(lambda row: row['SIFT']*row['SIFT'] if row['homozygous'] else row['SIFT'], axis=1)

    # Calculate GVB score for each gene
    gvb_scores_df = deleterious_df.groupby('Hugo_Symbol', group_keys=True)['adjusted_SIFT'].apply(lambda x: np.power(np.prod(x), 1/len(x)) if len(x) > 0 else 1).reset_index()
    gvb_scores_df.rename(columns={'adjusted_SIFT':'GVB score'}, inplace=True)
    return gvb_scores_df

def gene_group(row, gene_pair):
    gene_A, gene_B = gene_pair
    if row[gene_A] and row[gene_B]:
        return '%s_and_%s' % (gene_A, gene_B)
    elif row[gene_A]:
        return '%s_only' % (gene_A)
    elif row[gene_B]:
        return '%s_only' % (gene_B)
    else:
        return 'No_gene'

def analyze_gene_pair(genes, temp_cancer_df, patient_gene_df, output_lock):
    patient_gene_df['gene_group'] = patient_gene_df.apply(lambda row: gene_group(row, genes), axis=1)

    if '%s_and_%s' % (genes[0], genes[1]) in set(patient_gene_df['gene_group']):
        temp_survival_df = pd.merge(temp_cancer_df[['case_submitter_id', 'age_at_index', 'gender', 'overall_survival_time', 'event']], patient_gene_df[['case_submitter_id', 'gene_group']], how='right', on='case_submitter_id')
        temp_survival_df = temp_survival_df.drop_duplicates(ignore_index = True)

        temp_survival_df_dummies = pd.get_dummies(temp_survival_df[['age_at_index', 'gender', 'gene_group', 'overall_survival_time', 'event']])

        cph = CoxPHFitter(penalizer=0.2)
        cph.fit(temp_survival_df_dummies, duration_col='overall_survival_time', event_col='event')
        #print('\t'.join(map(str, [genes[0], genes[1], cph.hazard_ratios_['gene_group_%s_and_%s' % (genes[0], genes[1])], cph.summary.loc['gene_group_%s_and_%s'% (genes[0], genes[1])]['p']])) + '\r')
        result_line = ','.join(map(str, [genes[0], genes[1], cph.hazard_ratios_['gene_group_%s_and_%s' % (genes[0], genes[1])], cph.summary.loc['gene_group_%s_and_%s' % (genes[0], genes[1])]['p']])) + '\n'
        if result_line:
            with output_lock:
                with gzip.open('Result_COAD_SAM_0.3_20230418.csv.gz', 'at') as f:
                    f.write(result_line)

if __name__=="__main__":
    # Input data
    print("Open data")
    start_time = time.time()
    maf_df = pd.read_table('input/mc3.v0.2.8.PUBLIC.maf.gz', low_memory = False)
    clinical_df = pd.read_csv("input/clinical.tsv", sep = "\t", low_memory = False)

    # Merge MAF file, clinical file and exposure file
    maf_df['case_submitter_id'] = maf_df['Tumor_Sample_Barcode'].str[:12]
    clinical_list = list(clinical_df.columns)
    clinical_list.remove('treatment_type');clinical_list.remove('treatment_or_therapy')
    temp_clinical_df = clinical_df[clinical_list]
    temp_clinical_df = temp_clinical_df.drop_duplicates(ignore_index = True)
    merged_df = pd.merge(maf_df, temp_clinical_df, how='inner', on='case_submitter_id')

    # Filter variants for PASS and non-synonymous
    merged_df = merged_df[(merged_df['FILTER']=="PASS") & ((merged_df['Variant_Classification']=="Missense_Mutation") | (merged_df['Variant_Classification']=="Nonsense_Mutation"))]
    merged_df = merged_df.reset_index(drop=True)

    # Specific cancer : COAD
    print("Data for COAD")
    temp_cancer_df = merged_df[merged_df['project_id'] == 'TCGA-COAD']
    temp_cancer_df = temp_cancer_df.reset_index(drop=True)
    temp_cancer_df = temp_cancer_df[temp_cancer_df['age_at_index'] != "'--"]

    # Convert string into float
    temp_cancer_df['age_at_diagnosis'] = pd.to_numeric(temp_cancer_df.loc[:,'age_at_diagnosis'], errors = 'coerce')
    temp_cancer_df['age_at_index'] = pd.to_numeric(temp_cancer_df.loc[:,'age_at_index'], errors = 'coerce')
    temp_cancer_df['days_to_death'] = pd.to_numeric(temp_cancer_df.loc[:,'days_to_death'], errors = 'coerce')
    temp_cancer_df['days_to_last_follow_up'] = pd.to_numeric(temp_cancer_df.loc[:,'days_to_last_follow_up'], errors = 'coerce')

    # Make attributes : overall survival time, death event, gender
    temp_cancer_df['overall_survival_time'] = temp_cancer_df['days_to_death'].fillna(temp_cancer_df['days_to_last_follow_up'])
    temp_cancer_df['event'] = temp_cancer_df['vital_status'].apply(lambda x: True if x == 'Dead' else False)
    temp_cancer_df['gender'] = temp_cancer_df['gender'].apply(lambda x: 1 if x == 'male' else 0)

    # Make pivot table to group patients
    patient_gene_df = temp_cancer_df[['case_submitter_id','Hugo_Symbol']].pivot_table(index='case_submitter_id', columns='Hugo_Symbol', aggfunc=len, fill_value=0).reset_index()

    # Replace 'your_dataframe' with your actual dataframe variable
    gvb_scores = calculate_gvb(temp_cancer_df)

    # Get candidate gene pair which genes' gvb score < 0.3
    print("Calculate GVB score")
    candidates = list(combinations(list(gvb_scores[gvb_scores['GVB score'] < 0.3]['Hugo_Symbol']), 2))
    print("The number of all candidates : %s" % (len(candidates)))

    # Create a lock for output file writing
    output_lock = Lock()
    
	#Start multiprocessing
    print('Starting multiprocessing // ', time.ctime())
    fo = gzip.open('Result_COAD_SAM_0.3_20230418.csv.gz', 'wt')
    fo.write("GeneA,GeneB,Hazared ratio(exp for coef),p-value\n")

    num = 0
    if len(candidates) <= max_process_num:
        # run multiprocessing
        procs = []
        # allocate processes to different cores within a cluster
        for gene_pair in candidates:
            sys.stdout.write("Doing multiprocessing %s / %s // %s \r" %(num+1, len(candidates), time.ctime()))
            sys.stdout.flush()
            proc = Process(target=analyze_gene_pair, args=(gene_pair, temp_cancer_df, patient_gene_df, output_lock))
            procs.append(proc)
            proc.start()
            num += 1

        for proc in procs:
            proc.join()

    else:
        new_iternum = int(np.ceil(len(candidates)/max_process_num))

        for i in range(0, new_iternum + 1):
            new_candidate_list=[]
            first_index = i*max_process_num
            if i < len(candidates):
                new_index_list = np.arange(first_index, first_index+max_process_num)
            else:
                last_process = len(candidates) - first_index
                new_index_list = np.arange(first_index, first_index+last_process)

            for new_index in new_index_list:
                if not new_index >= len(candidates):
                    new_candidate_list.append(candidates[new_index])

            # run multiprocessing
            procs = []
            # allocate processes to different cores within a cluster
            for gene_pair in new_candidate_list:
                sys.stdout.write("Doing multiprocessing %s / %s // %s \r" %(num+1, len(candidates), time.ctime()))
                sys.stdout.flush()
                proc = Process(target=analyze_gene_pair, args=(gene_pair, temp_cancer_df, patient_gene_df, output_lock))
                procs.append(proc)
                proc.start()
                num += 1
            for proc in procs:
                proc.join()
    fo.close()

    test_df = pd.read_csv("Result_COAD_SAM_0.3_20230418.csv.gz", compression='gzip')
    print("The volume of dataframe :" % (len(test_df.index)))
    print('Process complete! // ', time.ctime())
    
