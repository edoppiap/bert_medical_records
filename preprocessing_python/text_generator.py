'''
This script, defines a function create_text_from_data which takes a dataframe or a file path,
processes the data, and writes the processed data into a text file.
'''
import pandas as pd
import csv
import os
from tqdm import tqdm
import streamlit as st
from datetime import datetime, timedelta
import argparse
import random
from sklearn.model_selection import train_test_split

def read_csv_dataset(file_path):
    with open(file_path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
        columns = next(reader)
        data = []
        for row in reader:
            # workaround because sometimes the cripted string is not read correctly
            if len(row) > 6:
                row[0] = ''.join([row[0],row[1]])
                row[1:] = row[2:]
            if row[1].startswith(','):
                row[1] = row[1].lstrip(',').rstrip('"')
                row[0] = row[0]+','
            data.append(row)

    df = pd.DataFrame(data, columns=columns)
    
    types_event = df['Type_event'].unique().tolist()
    strings_event = ['I-','D-','P-','M-', 'M-']
    types_dict = {}
    for type,string in zip(types_event, strings_event):
        types_dict[type] = string
        
    return df, types_dict

def is_less_than_3_month(date_1, date_2):
    date_1 = datetime.strptime(date_1, "%d/%m/%Y")
    date_2 = datetime.strptime(date_2, "%d/%m/%Y")
    
    # return if the two dates are less than 30 days apart
    return abs(date_1 - date_2) < timedelta(days=90) 
    

def create_infer_from_data(dataframe_or_file_path, output_folder, output_name = 'infer_dataset.txt', streamlit=False):
    """Function that generate the dataset for the Prediction out of the input csv file. It is needed only for development
    reason.

    Args:
        dataframe_or_file_path (_type_): DataFrame object or a path to the input csv file
        output_folder (_type_): Folder where to save the output text file
        output_name (str, optional): Output file name. Defaults to 'infer_dataset.txt'.
        streamlit (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: The path in which has been save the text dataset
    """
    #output_path = os.path.join(dataframe_or_folder, text_generated_name)
    if isinstance(dataframe_or_file_path, pd.DataFrame): # it means that there is directly the df file
        grouped_df = dataframe_or_file_path.groupby('patientID')
        
    elif os.path.exists(dataframe_or_file_path):
        df = pd.read_csv(dataframe_or_file_path, index_col=0)
        grouped_df = df.groupby('patientID')
    
    progress_text = 'Producing text file from csv dataset'
    if streamlit:
        loop = stqdm(grouped_df, desc=progress_text)
    else:
        loop = tqdm(grouped_df, desc=progress_text)

    results = []
    for patientID, patient in loop:
        result = f'{str(patientID)}, [CLS] '
        #result = str(patientID) + '\n'
        for _, row in patient.iterrows():
            result = result + f"{row['main_ICD9']}"
            if not pd.isna(row['ICD9_1']):
                result = result + f' {row["ICD9_1"]}'
            if not pd.isna(row['ICD9_2']):
                result = result + f' {row["ICD9_2"]}'
            if not pd.isna(row['ICD9_3']):
                result = result + f' {row["ICD9_3"]}'
            if not pd.isna(row['ICD9_4']):
                result = result + f' {row["ICD9_4"]}'
            if not pd.isna(row['ICD9_5']):
                result = result + f' {row["ICD9_5"]}'
            result = result + ' [SEP] '
        results.append(result+'\n')
        
        # if streamlit:
        #     my_bar.progress(i/(len(grouped_df)-1), text=progress_text)

    results = '\n'.join(results)
    
    text_dataset_path = os.path.join(output_folder, output_name)
    with open(text_dataset_path, 'w') as file:
        file.write(results)
        
    return text_dataset_path

def create_finetune_text_from_data(output_folder, file_path='data\PHeP_simulated_data.csv', output_name = 'finetune_dataset.txt'):
    """Function that generate the finetuning dataset, it delete the last hospitalization event and label the remaining events with 1 if the 
    deleted event is earlier than 90 days, 0 otherwise

    Args:
        output_folder (_type_): Folder in which save the text_dataset
        file_path (str, optional): _description_. Defaults to 'data\PHeP_simulated_data.csv'.
        output_name (str, optional): _description_. Defaults to 'finetune_dataset.txt'.
    """
    df = pd.read_csv(file_path, index_col=0)
    
    grouped_df = df.groupby('patientID')

    all_diagnoses = []
    all_hospitalisations = []

    for _,patient in tqdm(grouped_df, desc='Reading input file'):
        diagnoses = []
        hospitalisations = []
        for _,row in patient.iterrows():
            diagnosis = [row['main_ICD9']]
            if not pd.isna(row['ICD9_1']):
                diagnosis.append(row["ICD9_1"])
            if not pd.isna(row['ICD9_2']):
                diagnosis.append(row['ICD9_2'])
            if not pd.isna(row['ICD9_3']):
                diagnosis.append(row['ICD9_3'])
            if not pd.isna(row['ICD9_4']):
                diagnosis.append(row['ICD9_4'])
            if not pd.isna(row['ICD9_5']):
                diagnosis.append(row['ICD9_5'])

            diagnoses.append(diagnosis)
            
            hospitalisations.append(datetime.strptime(str(row['date_admission']), "%Y-%m-%d"))
    
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
            
def create_class_nsp_dataset(file_path, output_folder, output_name = 'class_nsp_dataset.txt', split=False):        
    df, types_dict = read_csv_dataset(file_path)
        
    bag = []
    current_patient = None
    for (patient,_),rows in tqdm(df.groupby(['Assistito_CodiceFiscale_Criptato','sentence']), desc='Creating bags of sentences'):
        if current_patient is None or patient != current_patient:
            current_patient = patient
        sentence = ' '.join(types_dict[rows['Type_event'].iloc[0]]+rows['Code_event'])
        bag.append(sentence)
    bag_size = len(bag)
    
    current_patient = None
    pairs = []
    for (patient,_),rows in tqdm(df.groupby(['Assistito_CodiceFiscale_Criptato','sentence']), desc='Creating pairs of sentences'):
        if current_patient is None or patient != current_patient:
            if current_patient is not None:
                sentences = sentences[::-1] # invert the order of the sentences to make it cronological
                num_sentences = len(sentences)
                start = 0
                while start < num_sentences - 2:
                    pair = '[CLS] ' + sentences[start] + ' [SEP] ' # sentence a 
                    if random.random() > .5:
                        pair += sentences[start+1] + ' <end> ' # sentence b
                        pair += '1' # they are consecutive
                    else:
                        pair += bag[random.randint(0, bag_size-1)] + ' <end> ' # sentence b
                        pair += '0' # they are NOT consecutive
                    start+=1
                    pairs.append(pair)
            current_patient = patient
            sentences = []
        sentences.append(' '.join((types_dict[rows['Type_event'].iloc[0]]+rows['Code_event'])[::-1]))
    
    if split:
        train,test = train_test_split(pairs, test_size=.2, random_state=42, shuffle=True)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        output_files = [os.path.join(output_folder, 'train.txt'),os.path.join(output_folder,'test.txt')]
        for output_file,split in zip(output_files,[train,test]):
            with open(output_file, 'w') as file:
                file.write('\n'.join(split))
    else:
        with open(os.path.join(output_folder, output_name), 'w') as file:
            file.write('\n'.join(pairs))
            
def create_finetuning_dataset(file_path, output_folder, output_name = 'finetuning_dataset.txt', split=False):
    df, types_dict = read_csv_dataset(file_path)
    
    docs = []
    current_patient = None
    dates = None
    doc = None
    for (patient,_),rows in tqdm(df.groupby(['Assistito_CodiceFiscale_Criptato','sentence']),desc='Creating list of docs'):
        # print(f"{patient = }, {date = }")
        sentence_list = (types_dict[rows['Type_event'].iloc[0]]+rows['Code_event'])
        date = rows['Data'].iloc[0]
        if current_patient is None:
            sentences = [' '.join(sentence_list)+ ' [SEP]']
            dates = [date]
            current_patient = patient
        elif patient != current_patient:
            # conclude the previous cicle
            try:
                # this will reverse the list of sentences and eliminate the last item
                doc = '[CLS] ' + ' '.join(sentences[::-1][:-1]) + ' <end> '
                # the date are not been reversed, so they can be interpreted in the same order of the csv file
                if len(dates) > 2 and is_less_than_3_month(dates[0],dates[1]):
                    doc += '1'
                else:
                    doc += '0'
                    
                # print(f'{current_patient = }\n{dates = }\n{doc =}\n\n')
                docs.append(doc)
            except:
                print(f"Exception:\n{current_patient = }\n{doc = }\n{dates = }")
            
            # prepare a new cicle
            current_patient = patient
            sentences = [' '.join(sentence_list)+ ' [SEP]']
            dates = [date]
        else: # this is a new line of the same patient
            sentences.append(' '.join(sentence_list)+ ' [SEP]')
            dates.append(date)
    # process the last patient
    doc = '[CLS] ' + ' '.join(sentences[::-1][:-1]) + '<end> '
    if len(dates) > 2 and is_less_than_3_month(dates[0],dates[1]):
        doc += '1'
    else:
        doc += '0'
    docs.append(doc)
            
    if split:
        train,test = train_test_split(docs, test_size=.2, random_state=42, shuffle=True)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_files = [os.path.join(output_folder, 'train.txt'),os.path.join(output_folder,'test.txt')]
        for output_file,split in zip(output_files,[train,test]):
            with open(output_file, 'w') as file:
                file.write('\n'.join(split))
    else:
        with open(os.path.join(output_folder, output_name), 'w') as file:
            file.write('\n'.join(docs))       
        
            
def create_mlm_only_dataset(file_path, output_folder, output_name = 'class_text_dataset.txt', split=False):        
    df, types_dict = read_csv_dataset(file_path)
    
    docs = []
    current_patient = None
    doc=None
    for (patient,_),rows in tqdm(df.groupby(['Assistito_CodiceFiscale_Criptato','sentence']), desc='Creating output lists'):
        if current_patient is None or patient != current_patient:
            if current_patient is not None:
                doc = '[CLS] '+' '.join(sentences[::-1])
                docs.append(doc) #add the previous line
            sentences = []
            current_patient = patient
        sentences.append(' '.join(types_dict[rows['Type_event'].iloc[0]]+rows['Code_event']) + " [SEP]")
    doc = '[CLS] '+' '.join(sentences[::-1])
    docs.append(doc) # add the last line

    if split:
        train,test = train_test_split(docs, test_size=.2, random_state=42, shuffle=True)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_files = [os.path.join(output_folder, 'train.txt'),os.path.join(output_folder,'test.txt')]
        print('Creating train and text output files')
        for output_file,split in zip(output_files,[train,test]):
            with open(output_file, 'w') as file:
                file.write('\n'.join(split))
    else:
        with open(os.path.join(output_folder, output_name), 'w') as file:
            file.write('\n'.join(docs))

def create_text_from_data(dataframe_or_file_path, output_folder, output_name = 'text_dataset.txt', streamlit=False):
    #output_path = os.path.join(dataframe_or_folder, text_generated_name)
    if isinstance(dataframe_or_file_path, pd.DataFrame): # it means that there is directly the df file
        grouped_df = dataframe_or_file_path.groupby('patientID')
        
    elif os.path.exists(dataframe_or_file_path):
        df = pd.read_csv(dataframe_or_file_path, index_col=0)
        grouped_df = df.groupby('patientID')
    
    progress_text = 'Producing text file from csv dataset'
    if streamlit:
        loop = stqdm(grouped_df, desc=progress_text)
    else:
        loop = tqdm(grouped_df, desc=progress_text)

    results = []
    for i, (_, patient) in enumerate(loop):
        result = '[CLS] '
        #result = str(patientID) + '\n'
        for _, row in patient.iterrows():
            result = result + f"{row['main_ICD9']}"
            if not pd.isna(row['ICD9_1']):
                result = result + f' {row["ICD9_1"]}'
            if not pd.isna(row['ICD9_2']):
                result = result + f' {row["ICD9_2"]}'
            if not pd.isna(row['ICD9_3']):
                result = result + f' {row["ICD9_3"]}'
            if not pd.isna(row['ICD9_4']):
                result = result + f' {row["ICD9_4"]}'
            if not pd.isna(row['ICD9_5']):
                result = result + f' {row["ICD9_5"]}'
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
    parser.add_argument('--output_name', type=str, default='dataset.txt',
                        help='Name for the output text file')
    parser.add_argument('--output_folder', type=str,
                        help='Folder where to save the output text file')
    parser.add_argument('--create_pretrain_text_file', action='store_true')
    parser.add_argument('--create_finetuning_text_data', action='store_true')
    parser.add_argument('--create_infer_text_data', action='store_true')
    parser.add_argument('--create_nsp_text_file', action='store_true')
    parser.add_argument('--create_mlm_only_dataset', action='store_true')
    parser.add_argument('--create_nsp_class_text_data', action='store_true')
    parser.add_argument('--split', action='store_true')
    
    args = parser.parse_args()

    if args.create_pretrain_text_file:
        create_text_from_data(args.file_path, 
                              output_folder=args.output_folder, 
                              output_name=args.output_name)
        
    if args.create_finetuning_text_data:
        create_finetuning_dataset(output_folder=args.output_folder,
                                       file_path=args.file_path,
                                       output_name=args.output_name,
                                       split=args.split)
    if args.create_infer_text_data:
        create_infer_from_data(args.file_path, 
                              output_folder=args.output_folder, 
                              output_name=args.output_name)
        
    if args.create_nsp_text_file:
        create_nsp_dataset(args.file_path,
                           output_folder=args.output_folder,
                           output_name=args.output_name)
    
    if args.create_mlm_only_dataset:
        create_mlm_only_dataset(args.file_path,
                           output_folder=args.output_folder,
                           output_name=args.output_name,
                            split=args.split)

    if args.create_nsp_class_text_data:
        create_class_nsp_dataset(args.file_path,
                                 output_folder=args.output_folder,
                                 output_name=args.output_name,
                                 split=args.split)