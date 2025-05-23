import pandas as pd
import numpy as np
import scipy.stats as stats

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
    sasr_set = set([pair[0] for pair in input_pair]) | set([pair[1] for pair in sam_dict['SKCM']])

    for i in range(0, len(nest)):
        if nest.iloc[i, 2] not in nest_dict.keys():
            nest_dict[nest.iloc[i, 2]] = set(nest.iloc[i, 3].split(' '))

    results = []
    for ctype_genes in [sasr_set]:
        M = len(set.union(*nest_dict.values()))  # total unique genes across all assemblies
        if M < len(ctype_genes):
            # Or you can set M = big number of known human genes if you prefer,
            # but let's assume union of all NeST genes for demonstration
            M = len(set.union(*nest_dict.values()))
    
        # Alternatively, if you want to define M as e.g. 20k for typical coding genes, do so.
    
        for assembly_name, assembly_genes in nest_dict.items():
            # Overlap
            overlap_genes = ctype_genes.intersection(assembly_genes)
            k = len(overlap_genes)   # number of genes from ctype set that are in this assembly
            n = len(assembly_genes)  # size of the assembly
            N = len(ctype_genes)     # size of the ctype gene set
        
            # Hypergeometric p-value: P(X>=k) for X ~ Hypergeom(M, n, N)
            # But we often do a one-sided test for enrichment => sf(k-1,...)
            pval = stats.hypergeom.sf(k-1, M, n, N)

            results.append({
                'CancerType': cancer_type,
                'Assembly': assembly_name,
                'OverlapCount': k,
                'AssemblySize': n,
                'CancerGeneSetSize': N,
                'TotalGenes': M,
                'pval': pval,
                'OverlapGenes': ' '.join(list(overlap_genes))
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("../Result/NeST_SAM_SKCM_enrichment_result.csv", index = False)
   

