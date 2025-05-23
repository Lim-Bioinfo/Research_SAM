import pandas as pd
import numpyn as np
import seaborn as sns

from functools import reduce
from combat.pycombat import pycombat
from scipy.stats import ttest_ind

def sam_group(pairs, expression_df, threshold):
    df_dict = dict()
    group_dict = dict() 
    
    for gene1, gene2 in pairs:
        if (gene1 in expression_df.index) and (gene2 in expression_df.index):
            group_dict[(gene1, gene2)] = {'Existed' : [], 'No-existed' : []}
            for sample in expression_df.columns:
        
                # Subset gene expression data for current sample
                sample_data = expression_df[sample]
            
                is_gene1_inactive = sample_data[gene1] < sample_data.quantile(threshold)
                is_gene2_inactive = sample_data[gene2] < sample_data.quantile(threshold)
            
                # Count as inactive if either of the sam pair genes is inactive
                if is_gene1_inactive and is_gene2_inactive:
                    group_dict[(gene1, gene2)]['Existed'].append(sample)
                    continue
                else:
                    group_dict[(gene1, gene2)]['No-existed'].append(sample)
                    continue    

    for gene1, gene2 in group_dict.keys():
        df_dict[(gene1, gene2)] = pd.DataFrame(list(group_dict[(gene1, gene2)].items()), columns=['Sample', 'Pair existed'])
        
    return df_dict

if __name__=="__main__":
    # Load datasets
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
  
    df1 = pd.read_csv("GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    clinical_df = pd.read_excel("../Input/Clinical_Meta_PM.xlsx") #Data from Supplementary Data S2
  
    ## Get SKCM SAM pairs
    sam_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

  
    label = {'GSE7553':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE7553']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE7553'])}
            'GSE8401':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE8401']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE8401'])}
            'GSE15605':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE15605']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE15605'])}
            'GSE46517':{'Primary':list(clinical_df[(clinical_df['Sample type'] == 'Primary') & clinical_df['Dataset'] == 'GSE46517']), 
                      'Metastases':list(clinical_df[(clinical_df['Sample type'] == 'Metastases') & clinical_df['Dataset'] == 'GSE46517'])}}

    # Preprocessing
    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)             
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy()

    df1_copied = np.log2(df1_copied + 1)
    df2_copied = np.log2(df2_copied + 1)
    df4_copied = np.log2(df4_copied + 1)

    # Merge dataframes
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)

    batches = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns)
    df_corrected = pycombat(df_merged, batches)

    primary = label['GSE7553']['Primary'] + label['GSE8401']['Primary'] + label['GSE15605']['Primary'] + label['GSE46517']['Primary']
    metastases = label['GSE7553']['Metastases'] + label['GSE8401']['Metastases'] + label['GSE15605']['Metastases'] + label['GSE46517']['Metastases']
  
    label_dict = {'Primary' : primary, 'Metastases' : metastases}
    df_corrected = df_corrected[sorted(primary + metastases)]

    # SAM score
    sam_score_merged_dict = {'Primary':dict(), 'Metastases' : dict()}
    result_dict = sam_group(sam_dict['SKCM'], df_corrected, 0.3)

    for sample_type in label_dict.keys():
        for patient in label_dict[sample_type]:     
            for sam_pair in result_dict.keys():
                if patient in result_dict[sam_pair].iloc[0, 1]:
                    if patient not in sam_score_merged_dict[sample_type].keys():
                        sam_score_merged_dict[sample_type][patient] = 1
                        continue
                    else:
                        sam_score_merged_dict[sample_type][patient] += 1
                        continue
                    
    for sample_type in label_dict.keys():
        for patient in label_dict[sample_type]:
            if patient not in sam_score_merged_dict[sample_type].keys():
                sam_score_merged_dict[sample_type][patient] = 0
                continue


    
    primary_scores = list(sam_score_merged_dict['Primary'].values())
    metastases_scores = list(sam_score_merged_dict['Metastases'].values()
    print(ttest_ind(primary_scores, metastases_scores)                    
    
                             
    # Create a DataFrame for seaborn
    df = pd.DataFrame({
    "SAM score": primary_scores + metastases_scores  # Combine values
    "Sample Type": ["Primary"] * len(primary_scores) + ["Metastases"] * len(metastases_scores)  # Labels})

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(x="Sample Type", y="SAM score", data=df, ax=ax)

    # Labels
    ax.set_xlabel("Sample Type")
    #ax.set_ylabel("SAM score")
    ax.set_title("Distribution of SAM Score by Sample Type")
    #plt.savefig("../../Research_SAM/figures/Compare primary and metastases SAM score in metacohort.svg", dpi = 600)

    # Show the plot
    plt.show()
