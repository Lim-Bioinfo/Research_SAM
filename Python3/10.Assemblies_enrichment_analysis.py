import pandas as pd
import numpy as np
import scipy.stats as stats
import gseapy as gp

from NeST import load_hierarchy

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    nest = pd.read_csv(../Input/NeST_node.csv")
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue

    nest_dict = dict()
    sam_set = set([pair[0] for pair in sam_dict['SKCM']) | set([pair[1] for pair in sam_dict['SKCM']])

    for i in range(0, len(nest)):
        if nest.iloc[i, 2] not in nest_dict.keys():
            nest_dict[nest.iloc[i, 2]] = set(nest.iloc[i, 3].split(' '))

    enr_bg = gp.enrichr(gene_list=list(gene_set),
                        gene_sets=nest_dict,
                        organism='human',
                        outdir=None,
                        #background= all_genes, #all_genes #'hsapiens_gene_ensembl', 'nest_dict['NeST']'
                        verbose=False)
    enr_bg.results.to_csv("Functional_enrichment_NeST_EnrichR_GSEAPY.csv",index = False)
   


