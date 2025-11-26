import os
import numpy as np
import pandas as pd

from sklearn import set_config
from tqdm import tqdm


if __name__=="__main__":
    print("Open file")
    lee_sl_df = pd.read_csv("../Input/JS Lee_NatComm_Experimentally identified gold standard SL interactions.csv")
    synleth_sl_df = pd.read_csv("../Input/gene_sl_gene.tsv", sep = "\t")
    sam_df = pd.read_csv("../Result/Primary_SAM_pairs.csv")
    similarity_df = pd.read_csv("../Result/Result_phylogenetic_similarity.csv.gz", compression ="gzip")
    essential_df = pd.read_excel("../Input/NatComm_Nichols et al_essesntial_profile.xlsx")


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
    non_essential_genes_Nichols = set(essential_df[(essential_df['Final essential'] == 0)]['Symbol'])

    ## SL pairs from previous studies
    lee_sl_df = lee_sl_df[lee_sl_df['SL'] == 1]
    lee_sl_set = set([tuple(sorted((lee_sl_df.iloc[i, 0], lee_sl_df.iloc[i, 1]))) for i in range(0, len(lee_sl_df.index))])
    synleth_sl_set = set([tuple(sorted((synleth_sl_df.iloc[i, 2], synleth_sl_df.iloc[i, 6]))) for i in range(0, len(synleth_sl_df.index))])



    ## Save SAM pairs
    with open("../Result/All_SAM_pairs.csv", "wt") as f:
        f.write("Cancer type,Gene1,Gene2,HR,P-value\n")
        for cancer_type in primary_sam_dict.keys():
            input_pairs = set(primary_sam_dict['SKCM']) - lee_sl_set - synleth_sl_set
            for pair in input_pairs:
                if len(set(pair) & non_essential_genes_Nichols) == 2:
                    f.write('%s,%s,%s,%s,%s\n' % (cancer_type, pair[0], pair[1], primary_sam_dict[cancer_type][pair][0], primary_sam_dict[cancer_type][pair][1]))





