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

    ## Bioplanet, melanoma-associated genes
    bioplanet_melanoma = {'PDGFRB', 'EGF', 'PDGFRA', 'BAD', 'MITF', 'BRAF', 'CDK6', 'PIK3CB', 'PIK3CA', 'CDK4', 'TP53', 'RAF1', 'RB1', 'PIK3R5', 'PIK3R3', 'ARAF', 'PIK3R2', 'EGFR', 'NRAS', 'CCND1', 'AKT3', 'CDH1', 'E2F1', 'AKT1', 'E2F2', 'FGF21', 'AKT2', 'E2F3', 'FGF20', 'FGF23', 'FGF22', 'MAP2K2', 'CDKN2A', 'MAP2K1', 'HGF', 'IGF1', 'FGF18', 'FGF17', 'FGF16', 'PIK3R1', 'FGF19', 'FGF10', 'FGFR1', 'MET', 'KRAS', 'FGF14', 'FGF13', 'FGF12', 'FGF11', 'MDM2', 'CDKN1A', 'PDGFC', 'PDGFB', 'PTEN', 'PDGFA', 'PIK3CD', 'FGF1', 'FGF2', 'FGF3', 'FGF4', 'IGF1R', 'FGF5', 'PIK3CG', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'PDGFD', 'MAPK3', 'HRAS', 'MAPK1'}
    ctd_melanoma_direct = set(['BAP1','BRAF','CDK4','CDKN2A','CNR1','CREB1','JMJD6','MC1R','MITF','POT1','STK11','TERT','XRCC3','PPAR'])
    neoplasm_metastasis = pd.read_csv("../Input/CTD_Neoplasm_Metastasis_D009362_genes_20250328_123515.tsv", sep = "\t")
    neoplasm_metastasis = neoplasm_metastasis.dropna()
    ctd_metastasis_direct = set(neoplasm_metastasis['Gene Symbol'])

    target_biomarkers =  bioplanet_melanoma & ctd_melanoma_direct & ctd_metastasis_direct
    
    
    ## Drug targen information, from Drugbank
    drug_dict = {'5-Azacytidine':{'DNMT1', 'PARP1'}, 'Bortezomib' : {'PSMB5', 'PSMB1', 'PRSS1'}, 'Cisplatin':{'MPG', 'A2M', 'TF', 'ATOX1'}, 
            'Doxorubicin':{'TOP2A', 'TOP2B', 'TOP1', 'TERT', 'NOLC1'}, 'Paclitaxel':{'TUBB1', 'BLC2', 'NR1I2'}, 'Sirolimus':{'MTOR'},
            'Sunitinib':{'PDGFRB', 'FLT1', 'KIT', 'KDR', 'FLT4', 'FLT3', 'CSF1R', 'PDGFRA', 'MET'}, 'Erlotinib':{'NR1I2', 'EGFR'},
            'Vorinostat':{'HDAC1', 'HDAC2', 'HDAC3', 'HDAC6', 'HDAC8'}, 'Dabrafenib':{'BRAF', 'RAF1', 'SIK1', 'NEK11', 'LIMK1'},
            'Sorafenib':{'BRAF', 'RAF1', 'FLT4', 'KDR', 'FLT1', 'FLT3', 'PDGFRB', 'KIT', 'FGFR1', 'RET', 'EGFR'}, 
            'RO4929097':{'PSEN1', 'NCSTN', 'PSENEN', 'APH1B', 'APH1A'}, 'Geldanamycin':{'HSP90AA1', 'HSP90B1', 'HSP90AB1'}, 
            'Gemcitabine':{'RRM1', 'RRM2', 'TYMS', 'CMPK1'}, 'Lapatinib':{'EGFR', 'ERBB2', 'EEF2K'}, 'Topotecan':{'TOP1', 'TOP1MT'}, 'Masitinib':{'SRC'}, 
            'Dasatinib':{'ABL1', 'SRC', 'EPHA2', 'LCK', 'YES1', 'KIT', 'PDGFRB', 'FYN', 'BCR', 'STAT5B', 'ABL2', 'BTK', 'NR4A3', 'CSK', 
                         'EPHA5', 'EPHB4', 'FGR', 'FRK', 'HSPA8' ,'LYN', 'MAP3K20', 'MAPK14', 'PPAT'}}
    
    ## Calculate network proximity between pairs and drug target genes
    network = wrappers.get_network("../Result/Human_STRING_v12.0_all_Fernandez_700.sif", only_lcc = True)
    for pair in sorted(sam_dict['SKCM']):
        for target in [{'PIK3CA'}, {'MAPK1','PPP2CA','UGCG'}, {'BRAF'}, {'RAC1', 'TIAM1' ,'TRIO'}, {'SRD5A2'}]:
            result = wrappers.calculate_proximity(network, target, set(pair) & set(network.nodes()))
            if len(set(pair) & network.nodes()) > 0:
                print("%s,%s,%s,%s,%s" % (pair[0], pair[1], result[0], result[1], result[3]))
        print('-----------')
