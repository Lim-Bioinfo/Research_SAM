import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn import set_config
from tqdm import tqdm


if __name__=="__main__":
    result_dict = dict()
    for cancer_type in tqdm(['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 
                    'COADREAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 
                    'KIRC', 'KIRP', 'LGG',  'LIHC', 'LUAD', 'LUSC', 
                    'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 
                    'SARC', 'SKCM', 'STAD',  'THCA', 'THYM', 'UCEC', 'UCS', 'UVM'], desc = 'Cancer type, finding for real SAM pairs'):
        file = '../Result/penalizer_%s/Result_%s_SAM_%s_%s_LOF.csv.gz' % (0.5, cancer_type, 0.3, 0.5)
        if os.path.isfile(file):
            result_df = pd.read_csv(file, compression='gzip')
            result_df = result_df[(result_df['P-value(Both genes)'] < 0.05) & (result_df['HazardRatio(Both genes)'] < 1)]
            result_df = result_df[((result_df['P-value(Only Gene1)'] >= 0.05) | (result_df['HazardRatio(Only Gene1)'] >= 1))]
            result_df = result_df[((result_df['P-value(Only Gene2)'] >= 0.05) | (result_df['HazardRatio(Only Gene2)'] >= 1))]
            result_dict[cancer_type] = result_df

    ## Primary SAM pairs
    with open("../Result/Primary_SAM_pairs.csv", "wt") as f:
        f.write("Cancer type,Gene1,Gene2,HazardRatio,P-value,Corrected P-value_Bonferroni, FDR_BH\n")
        for cancer_type in result_dict.keys():
            if len(result_dict[cancer_type].index) >= 1:
                for i in range(0, len(result_dict[cancer_type].index)):
                    f.write('%s,%s,%s,%s,%s,%s,%s\n' % (cancer_type, result_dict[cancer_type].iloc[i, 0], result_dict[cancer_type].iloc[i, 1], result_dict[cancer_type].iloc[i, 10], result_dict[cancer_type].iloc[i, 11], result_dict[cancer_type].iloc[i, 12], result_dict[cancer_type].iloc[i, 13]))
                    



