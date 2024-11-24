import numpy as np
import pandas as pd
import itertools
import gzip
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import distance
from tqdm import tqdm


def calculate_similarity(gene_pair_chunks, phylo_df, gene_to_index, feature_weight):
    result_lines = []
    for gene_pair in gene_pair_chunks:
        gene1 = gene_pair[0]
        gene2 = gene_pair[1]

        gene1_phylo_profile = phylo_df.loc[gene_to_index[gene1]][2:] # skip first 2 columns
        gene2_phylo_profile = phylo_df.loc[gene_to_index[gene2]][2:] # skip first 2 columns

        diff_square = (gene1_phylo_profile - gene2_phylo_profile) ** 2
        distance = np.dot(diff_square, feature_weight)
        
        result_line = f"{gene1},{gene2},{distance}\n"
        result_lines.append(result_line)

    return result_lines


def calculate_similarity_multiprocessing(gene_pairs, phylo_df, gene_to_index, feature_weight, num_processes):
    chunk_size = len(gene_pairs)//100
    gene_pair_chunks = [gene_pairs[i:i + chunk_size] for i in range(0, len(gene_pairs), chunk_size)]
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(calculate_similarity, gene_pair_chunk, phylo_df, gene_to_index, feature_weight): gene_pair_chunk for gene_pair_chunk in tqdm(gene_pair_chunks, desc = 'Input chunks to each process')}
        with gzip.open('../Result/Result_Phylogenetic_similarity.csv.gz', 'wt') as file1:
            file1.write("Gene1,Gene2,Distance\n")
            for future in tqdm(list(as_completed(futures)), desc='Writing to file'):
                result_lines = list(future.result())
                file1.write(''.join(result_lines))


if __name__=="__main__":
    start_time = time.time()
    print("Open data")
    phylo_df = pd.read_csv("../Input/JS Lee_Phylo_profile_Natcommm.csv")
    max_process_num = int(input("The number of process : "))

    gene_set = list(phylo_df['genes'])

    feature_weight = pd.read_csv('../Input/JS Lee_Phylo_weight_Naturecomm.txt', sep = "\t").values.flatten()

    gene_to_index = {gene: i for i, gene in enumerate(phylo_df.iloc[:, 1])}
    gene_pairs = list(itertools.combinations(gene_set, 2))

    print('Starting multiprocessing // ', time.ctime())
    ## Using multiprocessing to calculate phylogenetic similarity
    calculate_similarity_multiprocessing(gene_pairs, phylo_df, gene_to_index, feature_weight, max_process_num)
    
    data = pd.read_csv("../Result/Result_Phylogenetic_similarity.csv.gz", compression = 'gzip')
    print("The number of distances between gene pairs  : %s" % (len(data.index)))
    print('Process complete! // ', time.ctime())
    print("Elapsed time: %s seconds" % (time.time() - start_time))