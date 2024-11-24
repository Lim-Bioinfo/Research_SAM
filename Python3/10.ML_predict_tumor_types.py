import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gzip

from combat.pycombat import pycombat
from functools import reduce
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier


def calculate_performance_metrics(data, response_predicted, label):
    # Convert labels to binary format (1 for responder, 0 for non-responder)
    actual_labels = [1 if sample in label[data]['Primary'] else 0 for sample in label[data]['Primary'] + label[data]['Metastases']]
    predicted_labels = [1 if sample in response_predicted['Primary'] else 0 for sample in label[data]['Primary'] + label[data]['Metastases']]

    # Calculate metrics
    f1 = f1_score(actual_labels, predicted_labels)
    recall = recall_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels)
    auc = roc_auc_score(actual_labels, predicted_labels)

    return {'F1 Score': f1, 'Recall': recall, 'Precision': precision, 'AUC': auc}

if __name__=="__main__":
    print("Open file")
    sam_df = pd.read_csv("../Result/All_SAM_pairs.csv")
    biomarker_df = pd.read_excel("../Input/Biomarkers_metastasis_references.xlsx")
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
    #df_corrected.to_csv("GSE_merged_batchcorrected_sv.gct", sep = "\t")

    primary = label['GSE7553']['Primary'] + label['GSE8401']['Primary'] + label['GSE15605']['Primary'] + label['GSE46517']['Primary']
    metastases = label['GSE7553']['Metastases'] + label['GSE8401']['Metastases'] + label['GSE15605']['Metastases'] + label['GSE46517']['Metastases']
    label_dict = {'Primary' : primary, 'Metastases' : metastases}
    df_corrected = df_corrected[primary + metastases]


    ## Genes in SKCM SAM pairs
    sam_dict = dict()
    for i in range(0, len(sam_df.index)):
        if sam_df.iloc[i, 0] not in sam_dict.keys():
            sam_dict[sam_df.iloc[i, 0]] = [tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2])))]
            continue
        else:
            sam_dict[sam_df.iloc[i, 0]].append(tuple(sorted((sam_df.iloc[i, 1], sam_df.iloc[i, 2]))))
            continue 

    input_pair = sam_dict['SKCM']
    selected_genes = set()
    for pair in input_pair:
        selected_genes.add(pair[0])
        selected_genes.add(pair[1])

    ## Metastasis or metastatic tumor related biomarker in SKCM from previous literatures
    candidate_dict = {'SAM genes' : list(set(df_corrected.index) & selected_genes)}

    for i in range(0, len(biomarker_df.index)):
        if biomarker_df.iloc[i, 1] in set(df_corrected.index):
            if biomarker_df.iloc[i, 0] not in candidate_dict.keys():
                candidate_dict[biomarker_df.iloc[i, 0]] = [biomarker_df.iloc[i, 1]]
                continue
            else:
                candidate_dict[biomarker_df.iloc[i, 0]].append(biomarker_df.iloc[i, 1])
                continue      

    ## Predict tumor types with various metastaasis-related biomarkers
    loo = LeaveOneOut()
    fpr_tpr = []
    aucs = []

    for selected_genes_list in sorted(candidate_dict.keys()):
        all_true, all_pred, all_prob = [], [], []
    
        df_corrected_transposed = df_corrected.T
        df_selected = df_corrected_transposed[sorted(candidate_dict[selected_genes_list])]

        X = df_selected  # Features
        sample_label_map = {sample: 1 if sample in label_dict['Primary'] else 0 for sample in df_selected.index}
        y = [sample_label_map[sample] for sample in X.index]

        # Define Classifiers with specific random states where applicable
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(max_iter=10000, random_state=452456)),
                ('gn', GaussianNB()),
                ('gb', GradientBoostingClassifier(random_state=452456))], voting='soft')

        # Execute the LOOCV
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

            voting_clf.fit(X_train, y_train)
            y_pred = voting_clf.predict(X_test)
            probabilities = voting_clf.predict_proba(X_test)[:, 1]
        
            all_true.extend(y_test)
            all_pred.extend(y_pred)
            all_prob.extend(probabilities)

        roc_auc = roc_auc_score(all_true, all_prob)
        accuracy = accuracy_score(all_true, all_pred)
        precision = precision_score(all_true, all_pred)
        recall = recall_score(all_true, all_pred)
        f1 = f1_score(all_true, all_pred)
        fpr, tpr, _ = roc_curve(all_true, all_prob)
        fpr_tpr.append((fpr, tpr))
        aucs.append(roc_auc)
    
        print(f"Gene list: {selected_genes_list}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {roc_auc}")
        print('--------------')
    

    ## Plot AUC curves with distinct colors
    colors = ['#FF4D26', '#FF8847', '#F9BC66', '#548235', '#70AD47', '#A9D18E', '#2F5597', '#0070C0', '#9DC3E6', '#664BA0', '#B756D8', '#9A96FF']
    plt.figure(figsize=(12, 12))
    for i, color in zip(range(len(fpr_tpr)), colors):
        fpr, tpr = fpr_tpr[i]
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {aucs[i]:0.3f}) for {sorted(candidate_dict.keys())[i]}', color=color)

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic for Different Models')
    plt.legend(loc="lower right")
    plt.savefig("../Result/ML performance.svg", dpi = 600)
    plt.show()