import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from tqdm import tqdm


if __name__=="__main__":
    print("Open data")
    expression_df = pd.read_csv('../Input/TCGA_public_data/EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv', low_memory = False, sep = "\t")
    clinical_df = pd.read_excel("../Input/TCGA-Clinical Data Resource (CDR) Outcome.xlsx")
    sam_df = pd.read_csv("../Result/Primary_SAM_pairs.csv")

    ## Get primary SAM pairs
    sam_dict = dict()
    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    ## Calculate coexpression for primary tumor samples
    expression_df = expression_df.dropna(axis = 0)
    spearmancorr_result_dict = dict()


    for cancer_type in tqdm(['BLCA', 'LUAD', 'LUSC', 'STAD', 'SKCM'], desc = 'Cancer type'):
        temp_clinical_df = clinical_df[clinical_df['type'] == cancer_type]
        patient_set = set(temp_clinical_df[temp_clinical_df['Analyzed'] == 'Yes']['bcr_patient_barcode'])
        
        expression_df_copied = expression_df.copy()
        barcode_cols = [col for col in expression_df_copied.columns if 'TCGA' in col]
        temp_result = []
        trimmed_barcode_cols = []
        
        for col in expression_df_copied.columns:
            if (col in barcode_cols) & (col[13:15] in ['01', '02', '03', '04', '05', '06', '07', '08', '09']):
                if col[:12] in patient_set:
                    temp_result.append(col[:16])
                    trimmed_barcode_cols.append(col[:16])
                    continue
                else:
                    temp_result.append(col)
                    continue                    
            else:
                temp_result.append(col)
                continue
        
        expression_df_copied.columns = temp_result
        
        other_cols = ['gene_id']
        keep_columns = other_cols + trimmed_barcode_cols
        
        # Assuming expression_df is your TCGA expression data
        expression_df_filtered = expression_df_copied[keep_columns].copy()

        # Process 'gene_id' column
        expression_df_filtered['gene_id'] = expression_df_filtered['gene_id'].apply(lambda x: np.nan if x.split('|')[0] == '?' else x.split('|')[0])

        # Drop NaN rows and set index
        expression_df_filtered = expression_df_filtered.dropna(axis=0)
        expression_df_filtered = expression_df_filtered.set_index(expression_df_filtered['gene_id'])
        del expression_df_filtered['gene_id']
        expression_df_filtered_list = list(expression_df_filtered.columns)
        expression_df_filtered_copied = expression_df_filtered.copy()

        spearmancorr_result_dict[cancer_type] = dict()
        for pair in sam_dict[cancer_type]:
            if pair[0] in expression_df_filtered_copied.index and pair[1] in expression_df_filtered_copied.index:
                spearmancorr_result_dict[cancer_type][pair] = spearmanr(expression_df_filtered_copied.loc[pair[0]], expression_df_filtered_copied.loc[pair[1]])
            else:
                if len(set(pair) & set(expression_df_filtered_copied.index)) != 2:
                    print(cancer_type, pair)


    with open('../Result/TCGA_SAM_co-expression_spearman.csv', 'w') as f:
        f.write("Cancer type,Gene1,Gene2,Corr,P-value\n")
        for cancer_type in spearmancorr_result_dict.keys():
            for pair in spearmancorr_result_dict[cancer_type].keys():
                f.write("%s,%s,%s,%s,%s\n" % (cancer_type, pair[0], pair[1],spearmancorr_result_dict[cancer_type][pair][0], spearmancorr_result_dict[cancer_type][pair][1]))
