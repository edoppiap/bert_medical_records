'''
This script, defines a function create_text_from_data which takes a dataframe or a file path,
processes the data, and writes the processed data into a text file.
'''
import pandas as pd
import os
from tqdm import tqdm
from stqdm import stqdm #Streamlit-compatible progress bar
import streamlit as st
from datetime import datetime, timedelta
import argparse
import random

def create__infer_from_data(dataframe_or_file_path, output_folder, output_name = 'infer_dataset.txt', streamlit=False):
    #output_path = os.path.join(dataframe_or_folder, text_generated_name)
    if isinstance(dataframe_or_file_path, pd.DataFrame): # it means that there is directly the df file
        grouped_df = dataframe_or_file_path.groupby('keyone')
        
    elif os.path.exists(dataframe_or_file_path):
        df = pd.read_csv(dataframe_or_file_path, index_col=0)
        grouped_df = df.groupby('keyone')
    
    progress_text = 'Producing text file from csv dataset'
    if streamlit:
        loop = stqdm(grouped_df, desc=progress_text)
    else:
        loop = tqdm(grouped_df, desc=progress_text)

    results = []
    for keyone, patient in loop:
        result = f'{str(keyone)}, [CLS] '
        #result = str(keyone) + '\n'
        for _, row in patient.iterrows():
            result = result + f"{row['DIA_PRIN']}"
            if not pd.isna(row['DIA_UNO']):
                result = result + f' {row["DIA_UNO"]}'
            if not pd.isna(row['DIA_DUE']):
                result = result + f' {row["DIA_DUE"]}'
            if not pd.isna(row['DIA_TRE']):
                result = result + f' {row["DIA_TRE"]}'
            if not pd.isna(row['DIA_QUATTRO']):
                result = result + f' {row["DIA_QUATTRO"]}'
            if not pd.isna(row['DIA_CINQUE']):
                result = result + f' {row["DIA_CINQUE"]}'
            result = result + ' [SEP] '
        results.append(result+'\n')
        
        # if streamlit:
        #     my_bar.progress(i/(len(grouped_df)-1), text=progress_text)

    results = '\n'.join(results)
    
    text_dataset_path = os.path.join(output_folder, output_name)
    with open(text_dataset_path, 'w') as file:
        file.write(results)
        
    return text_dataset_path

def create_finetune_text_from_data(output_folder, file_path='/content/drive/MyDrive/EHR/base_red3.csv', output_name = 'finetune_dataset.txt'):
    df = pd.read_csv(file_path, index_col=0)
    
    grouped_df = df.groupby('keyone')

    all_diagnoses = []
    all_hospitalisations = []

    for _,patient in tqdm(grouped_df, desc='Reading input file'):
        diagnoses = []
        hospitalisations = []
        for _,row in patient.iterrows():
            diagnosis = [row['DIA_PRIN']]
            if not pd.isna(row['DIA_UNO']):
                diagnosis.append(row["DIA_UNO"])
            if not pd.isna(row['DIA_DUE']):
                diagnosis.append(row['DIA_DUE'])
            if not pd.isna(row['DIA_TRE']):
                diagnosis.append(row['DIA_TRE'])
            if not pd.isna(row['DIA_QUATTRO']):
                diagnosis.append(row['DIA_QUATTRO'])
            if not pd.isna(row['DIA_CINQUE']):
                diagnosis.append(row['DIA_CINQUE'])

            diagnoses.append(diagnosis)
            
            hospitalisation = '-'.join([str(row['GG_RIC']),str(row['MM_RIC']),str(row['AA_RIC'])])
            hospitalisations.append(datetime.strptime(hospitalisation, '%d-%m-%y'))
    
        all_diagnoses.append(diagnoses)
        all_hospitalisations.append(hospitalisations)
    
    selected_di = []
    labels = []

    for di,hos in zip(all_diagnoses,all_hospitalisations):
        i = len(di)-1
        while i > 0:
        # if len(di) > 2:
            selected_di.append(di[:i])
            if i == 1:
                labels.append(0)
                break
            month_difference = abs((hos[i].year - hos[i-1].year)*12 + hos[i].month - hos[i-1].month)
            if month_difference < 3:
                labels.append(1)
            else:
                labels.append(0)
            i-=1

    finetune_dataset_path = os.path.join(output_folder,output_name)
    with open(finetune_dataset_path, 'w') as file:
        for di,label in tqdm(zip(selected_di,labels), desc='Creating dataset for finetuning'):
            sentences = [' '.join(item) for item in di]
            file.write('[CLS] ' + ' [SEP] '.join(sentences) + f' <end> {label}\n\n')
            
def create_nsp_dataset(file_path, output_folder, output_name='nsp_dataset.txt'):
    bag = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('[CLS]'):
                sentences = [s for s in line.lstrip('[CLS]').split('[SEP]') if s.strip() != '']
                bag.extend(sentences)
    bag_size = len(bag)
    sentences_a = []
    sentences_b = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('[CLS]'):
                sentences = [s for s in line.lstrip('[CLS]').split('[SEP]') if s.strip() != '']
                num_sentences = len(sentences)
                start = 0
                while start < (num_sentences - 2):
                    sentences_a.append(sentences[start])
                    if random.random() > .5:
                        sentences_b.append(sentences[start+1])
                        labels.append(0)
                    else:
                        sentences_b.append(bag[random.randint(0, bag_size-1)])
                        labels.append(1)
                    start += 1
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file_path = os.path.join(output_folder, output_name)
    
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence_a, sentence_b, label in tqdm(zip(sentences_a, sentences_b, labels), desc='Creating nsp dataset'):
            file.write(f'[CLS] {sentence_a} [SEP] {sentence_b} <end> {label}\n\n')

def create_text_from_data(dataframe_or_file_path, output_folder, output_name = 'text_dataset.txt', streamlit=False):
    #output_path = os.path.join(dataframe_or_folder, text_generated_name)
    if isinstance(dataframe_or_file_path, pd.DataFrame): # it means that there is directly the df file
        grouped_df = dataframe_or_file_path.groupby('keyone')
        
    elif os.path.exists(dataframe_or_file_path):
        df = pd.read_csv(dataframe_or_file_path, index_col=0)
        grouped_df = df.groupby('keyone')
    
    progress_text = 'Producing text file from csv dataset'
    if streamlit:
        loop = stqdm(grouped_df, desc=progress_text)
    else:
        loop = tqdm(grouped_df, desc=progress_text)

    results = []
    for i, (_, patient) in enumerate(loop):
        result = '[CLS] '
        #result = str(keyone) + '\n'
        for _, row in patient.iterrows():
            result = result + f"{row['DIA_PRIN']}"
            if not pd.isna(row['DIA_UNO']):
                result = result + f' {row["DIA_UNO"]}'
            if not pd.isna(row['DIA_DUE']):
                result = result + f' {row["DIA_DUE"]}'
            if not pd.isna(row['DIA_TRE']):
                result = result + f' {row["DIA_TRE"]}'
            if not pd.isna(row['DIA_QUATTRO']):
                result = result + f' {row["DIA_QUATTRO"]}'
            if not pd.isna(row['DIA_CINQUE']):
                result = result + f' {row["DIA_CINQUE"]}'
            result = result + ' [SEP] '
        results.append(result+'\n')
        
        # if streamlit:
        #     my_bar.progress(i/(len(grouped_df)-1), text=progress_text)

    results = '\n'.join(results)
    
    text_dataset_path = os.path.join(output_folder, output_name)
    with open(text_dataset_path, 'w') as file:
        file.write(results)
        
    return text_dataset_path
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--file_path', type=str,
                        help='Folder where are located the input csv file')
    parser.add_argument('--output_name', type=str, default='dataset_text.txt',
                        help='Name for the output text file')
    parser.add_argument('--output_folder', type=str,
                        help='Folder where to save the output text file')
    parser.add_argument('--create_pretrain_text_file', action='store_true')
    parser.add_argument('--create_finetuning_text_data', action='store_true')
    parser.add_argument('--create_infer_text_data', action='store_true')
    parser.add_argument('--create_nsp_text_file', action='store_true')
    
    args = parser.parse_args()

    if args.create_pretrain_text_file:
        create_text_from_data(args.file_path, 
                              output_folder=args.output_folder, 
                              output_name=args.output_name)
    if args.create_finetuning_text_data:
        create_finetune_text_from_data(output_folder=args.output_folder,
                                       file_path=args.file_path,
                                       output_name=args.output_name)
    if args.create_infer_text_data:
        create__infer_from_data(args.file_path, 
                              output_folder=args.output_folder, 
                              output_name=args.output_name)
        
    if args.create_nsp_text_file:
        create_nsp_dataset(args.file_path,
                           output_folder=args.output_folder,
                           output_name=args.output_name)