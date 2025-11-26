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

    melanoma_cutaneous_malignant = pd.read_csv("CTD_melanoma_cutaneous_malignantD000096142_genes.tsv", sep = "\t", low_memory=False)
    neoplasm_metastasis = pd.read_csv("CTD_Neoplasm_Metastasis_D009362_genes.tsv", sep = "\t", low_memory=False)

    
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

    
    ## Get genes in SAM pairs
    selected_genes = []
    for pair in sam_dict['SKCM']:
        selected_genes.append(pair[0])
        selected_genes.append(pair[1])

    
    ## Get common genes associated with both melanoma and metastasis from CTD and BioPlanet
    bioplanet_melanoma = {'PDGFRB', 'EGF', 'PDGFRA', 'BAD', 'MITF', 'BRAF', 'CDK6', 'PIK3CB', 'PIK3CA', 'CDK4', 'TP53', 'RAF1', 'RB1', 'PIK3R5', 'PIK3R3', 'ARAF', 'PIK3R2', 'EGFR', 'NRAS', 'CCND1', 'AKT3', 'CDH1', 'E2F1', 'AKT1', 'E2F2', 'FGF21', 'AKT2', 'E2F3', 'FGF20', 'FGF23', 'FGF22', 'MAP2K2', 'CDKN2A', 'MAP2K1', 'HGF', 'IGF1', 'FGF18', 'FGF17', 'FGF16', 'PIK3R1', 'FGF19', 'FGF10', 'FGFR1', 'MET', 'KRAS', 'FGF14', 'FGF13', 'FGF12', 'FGF11', 'MDM2', 'CDKN1A', 'PDGFC', 'PDGFB', 'PTEN', 'PDGFA', 'PIK3CD', 'FGF1', 'FGF2', 'FGF3', 'FGF4', 'IGF1R', 'FGF5', 'PIK3CG', 'FGF6', 'FGF7', 'FGF8', 'FGF9', 'PDGFD', 'MAPK3', 'HRAS', 'MAPK1'}
    
    melanoma_cutaneous_malignant = melanoma_cutaneous_malignant[(melanoma_cutaneous_malignant['Direct Evidence'] == 'marker/mechanism') | (melanoma_cutaneous_malignant['Direct Evidence'] == 'therapeutic')]
    melanoma_cutaneous_malignant_genes = set(melanoma_cutaneous_malignant['Gene Symbol'])

    neoplasm_metastasis = neoplasm_metastasis[(neoplasm_metastasis['Direct Evidence'] == 'marker/mechanism') | (neoplasm_metastasis['Direct Evidence'] == 'therapeutic')]
    neoplasm_metastasis_genes = set(neoplasm_metastasis['Gene Symbol'])    
    
    target_biomarkers =  bioplanet_melanoma & melanoma_cutaneous_malignant_genes & neoplasm_metastasis_genes

    
    ## Drug targen information, from CTRP V2
    drug_dict = {'5-Azacytidine':{'DNMT1'}, #PG
                 'Bortezomib' : {'PSMB1', 'PSMB2', 'PSMB5', 'PSMD1', 'PSMD2'}, #PG
                 'Brivanib':{'FLT1', 'KDR'}, #PG
                 'Dasatinib':{'EPHA2', 'KIT', 'LCK', 'SRC', 'YES1'}, #PG
                 'Doxorubicin':{'TOP2A'}, #PG
                 'Erlotinib':{'EGFR', 'ERBB2'}, #PG
                 'Geldanamycin':{'HSP90AA1'},#PG
                 'Gemcitabine':{'CMPK1', 'RRM1', 'TYMS'}, #PG
                 'Lapatinib':{'EGFR', 'ERBB2'}, #PG
                 'Sirolimus':{'MTOR'}, #PG
                 'Sunitinib':{'FLT1', 'FLT3', 'KDR', 'KIT', 'PDGFRA', 'PDGFRB'}, #PG
                 'Sorafenib':{'BRAF', 'FLT3', 'KDR', 'RAF1'}, #PG
                 'Topotecan':{'TOP1'}, #PG
                 'Vorinostat':{'HDAC1', 'HDAC2', 'HDAC3', 'HDAC6', 'HDAC8'}, #PG
                 
                 'Alisertib':{'AURKA', 'AURKB'}, #CTRS
                 'BRD-A05715709':{'IDH1'}, #CTRS
                 'BRD-K11533227':{'HDAC1', 'HDAC2'}, #CTRS
                 'CAY10603':{'HDAC6'}, #CTRS
                 'DBeQ':{'VCP'}, #CTRS
                 'KI8751':{'KDR', 'KIT', 'PDGFRA'}, #CTRS
                 'KW-2449':{'AURKA', 'FLT3'}, #CTRS
                 'MLN 2480':{'TOP1', 'TOP1MT'}, #CTRS
                 'PCI-34051':{'HDAC8'}, #CTRS
                 'PF-03758309':{'PAK4'}, #CTRS
                 'Pifithrin MU':{'HSPA1A', 'HSPA1B', 'HSPA1L', 'TP53'},#CTRS
                 'RO4929097':{'APH1A', 'NCSTN', 'PSEN1', 'PSENEN'}, #CTRS
                 'SGX-523':{'MET'}, #CTRS
                 'SJ 172550':{'MDM2', 'TP53'}, #CTRS
                 'Tacedinaline':{'HDAC1', 'HDAC2', 'HDAC3', 'HDAC6', 'HDAC8'}, #CTRS
                 'Trametinib':{'MAP2K1', 'MAP2K2'} #CTRS
                }


    ## Calculate LCC to identify form module or not
    sam_lcc_result = wrappers.calculate_lcc_significance(network, selected_genes & nodes)

    
    ## Calculate network proximity between genes in SAM pairs and common genes associated with both melanoma and metastasis
    sam_proximity_result = wrappers.calculate_proximity(network, selected_genes & nodes, target_biomarkers & nodes)

    
    ## Calculate network proximity between drug target genes and melanoma and metastasis-related genes
    for drug in drug_dict.keys():
        print(drug, wrappers.calculate_proximity(network, drug_dict[drug] & nodes, target_biomarkers & nodes))
        print('-----------')
    
    ## Calculate network proximity between drug target genes and genes in SAM pairs
    for drug in drug_dict.keys():
        print(drug, wrappers.calculate_proximity(network, drug_dict[drug] & nodes, selected_genes & nodes))
        print('-----------')
