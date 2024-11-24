import os
import numpy as np
import pandas as pd

from sklearn import set_config
from tqdm import tqdm


if __name__=="__main__":
    print("Open file")
    lee_sl_df = pd.read_csv("../Input/JS Lee_NatComm_Experimentally identified gold standard SL interactions.csv")
    synleth_sl_df = pd.read_csv("../Input/Human_SL_SynLethDB.csv")
    sam_df = pd.read_csv("../Result/Primary_SAM_pairs.csv")
    similarity_df = pd.read_csv("../Result/Result_phylogenetic_similarity.csv.gz", compression ="gzip")
    coexpression_df = pd.read_csv("../Input/TCGA_SAM_co-expression_spearman.csv")
    essential_df1 = pd.read_csv("../Input/PlosGen_Georgi et al_essesntial_profile.csv")
    essential_df2 = pd.read_excel("../Input/NatComm_Nichols et al_essesntial_profile.xlsx")


    ## Primary SAM pairs
    primary_sam_dict = dict()
    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in primary_sam_dict.keys():
            primary_sam_dict[sam_df.iloc[i, 0]] = dict()
            primary_sam_dict[sam_df.iloc[i, 0]][tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))] = [sam_df.iloc[i, 3], sam_df.iloc[i, 4]]
            continue
        else:
            primary_sam_dict[sam_df.iloc[i, 0]][tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))] = [sam_df.iloc[i, 3], sam_df.iloc[i, 4]]
            continue            


    ## Essential genes
    essential_genes_Benjamin = set(essential_df1[essential_df1["Essential"] == 'Y']['Gene_symbol'])
    essential_genes_Nichols = set(essential_df2[(essential_df2['Final essential'] == 1)]['Symbol'])


    ## Phylogenetic similar pair with different threshold
    mean_distance = np.mean(similarity_df['Distance'])
    median_distance = np.median(similarity_df['Distance'])
    sl_distance = 10.5

    mean_df = similarity_df[(similarity_df ['Distance'] < mean_distance)]
    median_df = similarity_df[(similarity_df ['Distance'] < median_distance)]
    sl_df = similarity_df[(similarity_df ['Distance'] < sl_distance)]

    mean_set = set([tuple(sorted((mean_df.iloc[i, 0], mean_df.iloc[i, 1]))) for i in range(0, len(mean_df.index))])
    median_set = set([tuple(sorted((median_df.iloc[i, 0], median_df.iloc[i, 1]))) for i in range(0, len(median_df.index))])
    sl_set = set([tuple(sorted((sl_df.iloc[i, 0], sl_df.iloc[i, 1]))) for i in range(0, len(sl_df.index))])


    ## SL pairs from previous studies
    lee_sl_df = lee_sl_df[lee_sl_df['SL'] == 1]
    lee_sl_set = set([tuple(sorted((lee_sl_df.iloc[i, 0], lee_sl_df.iloc[i, 1]))) for i in range(0, len(lee_sl_df.index))])
    synleth_sl_set = set([tuple(sorted((synleth_sl_df.iloc[i, 0], synleth_sl_df.iloc[i, 1]))) for i in range(0, len(synleth_sl_df.index))])


    ## SAM pairs with coexpressed
    sam_coexpressed_dict = dict()
    for i in range(0, len(coexpression_df.index)):
        if coexpression_df.iloc[i, 3] > 0.1 and coexpression_df.iloc[i, 4] < 0.05:
            if coexpression_df.iloc[i, 0] not in sam_coexpressed_dict.keys():
                sam_coexpressed_dict[coexpression_df.iloc[i, 0]] = [tuple(sorted((coexpression_df.iloc[i, 1], coexpression_df.iloc[i, 2])))]
                continue
            else:
                sam_coexpressed_dict[coexpression_df.iloc[i, 0]].append(tuple(sorted((coexpression_df.iloc[i, 1], coexpression_df.iloc[i, 2]))))
                continue 


    ## Save SAM pairs
    with open("../Result/All_SAM_pairs.csv", "wt") as f:
        f.write("Cancer type,Gene1,Gene2,HR,P-value\n")
        for cancer_type in primary_sam_dict.keys():
            input_pairs = set(sam_coexpressed_dict[cancer_type]) & mean_set - lee_sl_set - synleth_sl_set
            for pair in input_pairs:
                if len(set(pair) & essential_genes_Benjamin) == 0 and len(set(pair) & essential_genes_Nichols) == 0:
                    f.write('%s,%s,%s,%s,%s\n' % (cancer_type, pair[0], pair[1], primary_sam_dict[cancer_type][pair][0], primary_sam_dict[cancer_type][pair][1]))




