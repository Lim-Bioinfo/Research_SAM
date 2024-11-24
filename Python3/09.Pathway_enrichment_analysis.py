import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import gseapy
from tqdm import tqdm

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    ## Functional enrichment analysis for each cancer types' SAM pairs
    for cancer_type in tqdm(['LUAD', 'LUSC', 'STAD', 'SKCM'], desc = "Cancer type"):
        selected_genes = set()
        for pair in sam_dict[cancer_type]:
            selected_genes.add(pair[0])
            selected_genes.add(pair[1])
    
            enr = gseapy.enrichr(gene_list=list(selected_genes), # or "./tests/data/gene_list.txt",
                                gene_sets=['MSigDB_Hallmark_2020', 'KEGG_2021_Human', 
                            'GO_Biological_Process_2023', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2023',
                            'Reactome_2022', 'WikiPathways_2019_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                 )
            result_df = enr.results

            ## Discovery significantly enriched pathways
            result_df = result_df[result_df['P-value'] < 0.05]
            result_df.to_excel("../Result/EnrichR_GSEAPY_%s_SAM_genes.xlsx" % (cancer_type), index = False)

            ## Top 5 significantly enriched pathways bar plot
            ax = gseapy.barplot(enr.results,
              column="P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=5,
              figsize=(10,20),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'MSigDB_Hallmark_2020' : '#70AD47', 'KEGG_2021_Human' : '#00B0F0', 
                            'GO_Biological_Process_2023' : '#7030A0', 'GO_Cellular_Component_2023' : '#002060', 'GO_Molecular_Function_2023' : '#4472C4',
                            'Reactome_2022' : '#FFA034', 'WikiPathways_2019_Human' : '#D90056'})

            plt.savefig("%s enrichr from gseapy p0.05.svg" % (cancer_type), dpi =600)