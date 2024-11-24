import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gzip

from combat.pycombat import pycombat
from functools import reduce
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.preprocessing import RobustScaler
from scipy.stats import ttest_ind, mannwhitneyu


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
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    df1 = pd.read_csv("../Input/GSE7553_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df2 = pd.read_csv("../Input/GSE8401_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df3 = pd.read_csv("../Input/GSE15605_series_matrix_collapsed_to_symbols.gct", sep = "\t")
    df4 = pd.read_csv("../Input/GSE46517_series_matrix_collapsed_to_symbols.gct", sep = "\t")

    ## Batch correction
    label = {'GSE7553' : {'Primary':['GSM183224','GSM183225','GSM183235','GSM183258','GSM183259','GSM183260','GSM183261','GSM183262','GSM183263','GSM183264','GSM183265','GSM183266','GSM183303','GSM183304'], 
                      'Metastases' : ['GSM183226','GSM183227','GSM183228','GSM183229','GSM183230','GSM183231','GSM183232','GSM183233','GSM183252','GSM183253','GSM183254','GSM183255','GSM183256','GSM183257','GSM183273','GSM183274','GSM183275','GSM183276','GSM183277','GSM183278','GSM183279','GSM183280','GSM183281','GSM183282','GSM183283','GSM183284','GSM183285','GSM183286','GSM183287','GSM183288','GSM183289','GSM183290','GSM183291','GSM183292','GSM183293','GSM183294','GSM183295','GSM183296','GSM183297','GSM183298']},
            'GSE8401' : {'Primary':['GSM207929','GSM207930','GSM207931','GSM207932','GSM207933','GSM207934','GSM207935','GSM207936','GSM207937','GSM207938','GSM207939','GSM207940','GSM207941','GSM207942','GSM207943','GSM207944','GSM207945','GSM207946','GSM207947','GSM207948','GSM207949','GSM207950','GSM207951','GSM207952','GSM207953','GSM207954','GSM207955','GSM207956','GSM207957','GSM207958','GSM207959'], 
                     'Metastases' : ['GSM207960','GSM207961','GSM207962','GSM207963','GSM207964','GSM207965','GSM207966','GSM207967','GSM207968','GSM207969','GSM207970','GSM207971','GSM207972','GSM207973','GSM207974','GSM207975','GSM207976','GSM207977','GSM207978','GSM207979','GSM207980','GSM207981','GSM207982','GSM207983','GSM207984','GSM207985','GSM207986','GSM207987','GSM207988','GSM207989','GSM207990','GSM207991','GSM207992','GSM207993','GSM207994','GSM207995','GSM207996','GSM207997','GSM207998','GSM207999','GSM208000','GSM208001','GSM208002','GSM208003','GSM208004','GSM208005','GSM208006','GSM208007','GSM208008','GSM208009','GSM208010','GSM208011']},
            'GSE15605' : {'Primary':['GSM390224','GSM390225','GSM390226','GSM390227','GSM390228','GSM390229','GSM390230','GSM390231','GSM390232','GSM390233','GSM390234','GSM390235','GSM390236','GSM390237','GSM390238','GSM390239','GSM390240','GSM390241','GSM390242','GSM390243','GSM390244','GSM390245','GSM390246','GSM390247','GSM390248','GSM390249','GSM390250','GSM390251','GSM390252','GSM390253','GSM390254','GSM390255','GSM390256','GSM390257','GSM390258','GSM390259','GSM390260','GSM390261','GSM390262','GSM390263','GSM390264','GSM390265','GSM390266','GSM390267','GSM390268','GSM390269'], 
                      'Metastases' : ['GSM390270','GSM390271','GSM390272','GSM390273','GSM390274','GSM390275','GSM390276','GSM390277','GSM390278','GSM390279','GSM390280','GSM390281']},
            'GSE46517' : {'Primary':['GSM1131639','GSM1131640','GSM1131641','GSM1131642','GSM1131643','GSM1131644','GSM1131645','GSM1131646','GSM1131647','GSM1131648','GSM1131649','GSM1131650','GSM1131651','GSM1131652','GSM1131653','GSM1131654','GSM1131655','GSM1131656','GSM1131657','GSM1131658','GSM1131659','GSM1131660','GSM1131661','GSM1131662','GSM1131663','GSM1131664','GSM1131665','GSM1131666','GSM1131667','GSM1131668','GSM1131669'], 
                      'Metastases' : ['GSM1131566','GSM1131567','GSM1131568','GSM1131569','GSM1131570','GSM1131571','GSM1131572','GSM1131573','GSM1131574','GSM1131575','GSM1131576','GSM1131577','GSM1131578','GSM1131579','GSM1131580','GSM1131581','GSM1131582','GSM1131583','GSM1131584','GSM1131585','GSM1131586','GSM1131587','GSM1131588','GSM1131589','GSM1131590','GSM1131591','GSM1131592','GSM1131593','GSM1131594','GSM1131595','GSM1131596','GSM1131597','GSM1131598','GSM1131599','GSM1131600','GSM1131601','GSM1131602','GSM1131603','GSM1131604','GSM1131605','GSM1131606','GSM1131607','GSM1131608','GSM1131609','GSM1131610','GSM1131611','GSM1131612','GSM1131613','GSM1131614','GSM1131615','GSM1131616','GSM1131617','GSM1131618','GSM1131619','GSM1131620','GSM1131621','GSM1131622','GSM1131623','GSM1131624','GSM1131625','GSM1131626','GSM1131627','GSM1131628','GSM1131629','GSM1131630','GSM1131631','GSM1131632','GSM1131633','GSM1131634','GSM1131635','GSM1131636','GSM1131637','GSM1131638']}}

    del df1['DESCRIPTION'];del df2['DESCRIPTION'];del df3['DESCRIPTION'];del df4['DESCRIPTION']
    df1 = df1.set_index(keys='NAME');df2 = df2.set_index(keys='NAME');df3 = df3.set_index(keys='NAME');df4 = df4.set_index(keys='NAME')
    df1_list = list(df1.columns);df2_list = list(df2.columns);df3_list = list(df3.columns);df4_list = list(df4.columns)             
    df1_copied = df1.copy();df2_copied = df2.copy();df3_copied = df3.copy();df4_copied = df4.copy()

    df1_copied[df1_list] = RobustScaler().fit_transform(df1[df1_list])
    df2_copied[df2_list] = RobustScaler().fit_transform(df2[df2_list])
    df3_copied[df3_list] = RobustScaler().fit_transform(df3[df3_list])
    df4_copied[df4_list] = RobustScaler().fit_transform(df4[df4_list])
                                 
    data_frames = [df1_copied, df2_copied, df3_copied, df4_copied]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on=['NAME'], how='inner'), data_frames)

    batches = ['GSE7553']*len(df1.columns) + ['GSE8401']*len(df2.columns) + ['GSE15605']*len(df3.columns) + ['GSE46517']*len(df4.columns)
    df_corrected = pycombat(df_merged, batches)
    #df_corrected.to_csv("GSE_merged_batchcorrected_pm.gct", sep = "\t")

    primary = label['GSE7553']['Primary'] + label['GSE8401']['Primary'] + label['GSE15605']['Primary'] + label['GSE46517']['Primary']
    metastases = label['GSE7553']['Metastases'] + label['GSE8401']['Metastases'] + label['GSE15605']['Metastases'] + label['GSE46517']['Metastases']
    label_dict = {'Primary' : primary, 'Metastases' : metastases}
    df_corrected = df_corrected[primary + metastases]


    ## Calculate SKCM SAM scores
    sam_dict = dict()
    sam_hr_dict = dict()

    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            sam_hr_dict[sam_df.iloc[i, 0]] = dict()
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 
        
    for i in range(0, len(sam_df.index)):
        sam_hr_dict[sam_df.iloc[i, 0]][tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))] = sam_df.iloc[i, 3]

    sam_score_merged_dict = {'Primary':dict(), 'Metastases' : dict()}

    result_dict = sam_group(sam_dict['SKCM'], df_corrected, 0.3)

    for sample_type in label_dict.keys():
        for patient in label_dict[sample_type]:     
            for sam_pair in result_dict.keys():
                hr = sam_hr_dict['SKCM'][sam_pair]
                if patient in result_dict[sam_pair].iloc[0, 1]:
                    if patient not in sam_score_merged_dict[sample_type].keys():
                        sam_score_merged_dict[sample_type][patient] = 1/hr
                        continue
                    else:
                        sam_score_merged_dict[sample_type][patient] += 1/hr
                        continue
                    
    for sample_type in label_dict.keys():
        for patient in label_dict[sample_type]:
            if patient not in sam_score_merged_dict[sample_type].keys():
                sam_score_merged_dict[sample_type][patient] = 0
                continue
            

    ## Compare sam score between different two tumor types
    print(ttest_ind(list(sam_score_merged_dict['Primary'].values()), list(sam_score_merged_dict['Metastases'].values())))

    primary = list(sam_score_merged_dict['Primary'].values())
    metastases = list(sam_score_merged_dict['Metastases'].values())

    ## Draw boxplot to compare primary and metastases
    print(np.mean(primary), np.mean(metastases))
    print(np.median(primary), np.median(metastases))

    fig, ax = plt.subplots(figsize = (6,5))
    data_labels = ["Primary", "Metastases"]
    plt.boxplot([primary, metastases])#, showfliers=False)
    plt.xticks([1,2], data_labels)
    plt.xlabel('Sample Type')
    plt.ylabel('SAM score')
    plt.savefig("../Result/Compare SAM score.svg", dpi = 600)
    plt.show()