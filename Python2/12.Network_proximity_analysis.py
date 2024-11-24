import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm
from toolbox import wrappers

if __name__=="__main__":
    print("Open file")
    info = pd.read_csv("../Input/9606.protein.info.v12.0.txt.gz", sep = "\t")
    ppi = pd.read_csv("../Input/9606.protein.links.v12.0.txt.gz", sep = " ")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")

    ## Network preprocessing
    info_dict = info.set_index('#string_protein_id').T.to_dict()
    ppi_700 = ppi[(ppi['combined_score'] > 700)]
    ppi_new_700 = ppi_700.reset_index(drop=True)
    ppi_new_700.loc[ppi_new_700['combined_score'] > 700, 'combined_score'] = 1
    ppi_new_700 = ppi_new_700[['protein1', 'combined_score', 'protein2']]
    ppi_700.to_csv("../Result/Human_STRING_v12.0_all_Fernandez_700_20241104.sif", sep = " ", index = False, header = False)

    for i in tqdm(range(0, len(ppi_700.index)), total=len(range(0, len(ppi_700.index))), desc="Change ENSP to Symbol, score > 700"):
        ppi_700.iloc[i, 0] = info_dict[ppi_700.iloc[i, 0]]['preferred_name']
        ppi_700.iloc[i, 2] = info_dict[ppi_700.iloc[i, 2]]['preferred_name']


    ## Get SAM pairs
    sam_dict = dict()
    network = wrappers.get_network("../Result/Human_STRING_v12.0_all_Fernandez_700.sif", only_lcc = True)
    nodes = network.nodes()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    ## Calculate network proximity
    network = wrappers.get_network("../Input/Human_STRING_v12.0_all_Fernandez_700.sif", only_lcc = True)
    for pair in sorted(sam_dict['SKCM']):
        for target in [{'PIK3CA'}, {'MAPK1','PPP2CA','UGCG'}, {'BRAF'}, {'RAC1', 'TIAM1' ,'TRIO'}, {'SRD5A2'}]:
            result = wrappers.calculate_proximity(network, target, set(pair) & set(network.nodes()))
            if len(set(pair) & network.nodes()) > 0:
                print("%s,%s,%s,%s,%s" % (pair[0], pair[1], result[0], result[1], result[3]))
        print('-----------')