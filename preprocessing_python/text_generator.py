'''
This script, defines a function create_text_from_data which takes a dataframe or a file path,
processes the data, and writes the processed data into a text file.
'''
import pandas as pd
import csv
import os, sys
from tqdm import tqdm
# import streamlit as st
from datetime import datetime, timedelta
import argparse
import random
from sklearn.model_selection import train_test_split

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import setup_logging
import logging

import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
import rpy2.robjects.pandas2ri as pandas2ri

FORMAT_COLUMNS = {
    'format_2' : ['patientID', 'sex', 'age', 'main_ICD9', 'ICD9_1', 'ICD9_2', 'ICD9_3', 'ICD9_4', 'ICD9_5', 'date_admission', 'date_discharge'],
    'format_3' : ['Assistito_CodiceFiscale_Criptato', 'Data', 'Code_event', 'Type_event', 'sentence', 'Description_event', 'Label_event']
}

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#   
#   Format_1: ['keyone', 'GENERE', 'eta_inizio', ...]
#
#

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 
#   Format_2: ['patientID', 'seg', 'age', ...]
#
#

def format_2_read_csv(file_path,read_hosp=False):
    df = pd.read_csv(file_path, index_col=0).groupby('patientID')
    
    patient_dict = {}
    
    for patientID,patient in tqdm(df, desc='Reading input file'):
        diagnoses = []
        if read_hosp: hospitalisations = []
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
            
            if read_hosp:
                hospitalisations.append(datetime.strptime(str(row['date_admission']), "%Y-%m-%d"))
        
        patient_dict[patientID] = {
            'diagnoses' : diagnoses
        }
        if read_hosp:
            patient_dict[patientID]['hospitalisations'] = hospitalisations
            
    return patient_dict
    

def format_2_create_nsp(file_path):
    patient_dict = format_2_read_csv(file_path)

    bag = []
    for _,value_dict in tqdm(patient_dict.items(), desc='Creating bag of sentences'):
        sentences = [' '.join(item) for item in value_dict['diagnoses']]
        bag.extend(sentences)
    bag_size = len(bag)

    pairs = []
    n_patients = 0
    for _,value_dict in tqdm(patient_dict.items(), desc='Creating pairs'):
        n_patients += 1
        sentences = [' '.join(item) for item in value_dict['diagnoses']]
        num_sentences = len(sentences)
        start = 0
        while start < num_sentences - 2:
            pair = '[CLS] ' + sentences[start] + ' [SEP] ' # sentence a
            if random.random() > .5:
                pair += sentences[start + 1] + ' <end> ' # sentence b
                pair += '1' # they are consecutive
            else:
                pair += bag[random.randint(0,bag_size-1)] + ' <end> ' # sentence b
                pair += '0' # they are not consecutive
            start += 1
            pairs.append(pair)
            
    logging.info(f'Number of patients = {n_patients}')
    logging.info(f'Created {len(pairs)} pairs (average {len(pairs)/n_patients:.2f} pairs/patient)')
    return pairs

def format_2_create_infer(file_path):
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
    
    patient_dict = format_2_read_csv(file_path)
    
    docs = []
    for patientID,diagnoses in patient_dict.items():
        diagnoses = diagnoses['diagnoses']
        result = f'{str(patientID)}, [CLS] '
        for diagnosis in diagnoses:
            result += ' '.join(diagnosis) + ' [SEP]'
        docs.append(result)
        
    return docs

def format_2_create_finetune(file_path='data/PHeP_simulated_data.csv'):
    """Function that generate the finetuning dataset, it delete the last hospitalization event and label the remaining events with 1 if the 
    deleted event is earlier than 90 days, 0 otherwise

    Args:
        output_folder (_type_): Folder in which save the text_dataset
        file_path (str, optional): _description_. Defaults to 'data/PHeP_simulated_data.csv'.
        output_name (str, optional): _description_. Defaults to 'finetune_dataset.txt'.
    """
    patient_dict = format_2_read_csv(file_path, read_hosp=True)
    
    docs = []    
    for _,value_dict in tqdm(patient_dict.items(), desc='Creating labeled sentences'):
        di = value_dict['diagnoses']
        hos = value_dict['hospitalisations']
        
        i = len(di)-1
        while i > 0:
        # if len(di) > 2:
            sentences = [' '.join(item) for item in di[:i]]
            if i == 1:
                label = 0
            else:
                month_difference = abs((hos[i].year - hos[i-1].year)*12 + hos[i].month - hos[i-1].month)
                if month_difference < 3:
                    label = 1
                else:
                    label = 0
            i-=1
            docs.append('[CLS] ' + ' [SEP] '.join(sentences) + f' [SEP] <end> {str(label)}')
            
    return docs
            
def create_mlm_only_format_2(file_path):
    patient_dict = format_2_read_csv(file_path)

    docs = []
    for _,value_dict in patient_dict.items():
        sentences = [' '.join(item) for item in value_dict['diagnoses']]
        docs.append('[CLS] ' + ' [SEP] '.join(sentences) + ' [SEP]')

    return docs

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 
#   Format_3: ['Assistito_codice_fiscale_criptato', 'Data', 'Code_event', ...]
# 
#

def read_csv_format_3(file_path):
    # this use R to read the csv file and load the correct string
    pandas2ri.activate()
    base = rpackages.importr('base')
    data_table = rpackages.importr('data.table')
    
    fread = ro.r('''function(file) {read.csv(file)}''')
    df = fread(file_path)
    if not isinstance(df, pd.DataFrame):
        df = ro.conversion.get_conversion().rpy2py(df)
    
    types_event = df['Type_event'].unique().tolist()
    strings_event = ['I','D','P','M', 'M'] # established way of concatenate classes
    types_dict = {}
    for type,string in zip(types_event, strings_event):
        types_dict[type] = string
        
    return df, types_dict

def is_less_than_3_month(date_1, date_2):
    d1 = datetime.strptime(date_1, "%d/%m/%Y")
    d2 = datetime.strptime(date_2, "%d/%m/%Y")
    
    # return if the two dates are less than 90 days apart
    return abs(d1 - d2) < timedelta(days=90)

def read_sentence(df, types_dict, use_time=False, dont_use_hypen=False):
    if use_time:
        if dont_use_hypen:
            sentence = [
                f'{event.replace(" ","").replace("[","").replace("]","")}{types_dict[df["Type_event"].iloc[0]]}-{date.split("/")[1]}'
                    for event,date in zip(df["Code_event"],df["Data"])
            ]
        else:
            sentence = [
                f'{types_dict[df["Type_event"].iloc[0]]}-{date.split("/")[1]}-{event.replace(" ","-")}'
                        for event,date in zip(df["Code_event"],df["Data"])
            ]
    else:
        if dont_use_hypen:
            sentence = [
                f'{event.replace(" ","").replace("[","").replace("]","")}{types_dict[df["Type_event"].iloc[0]]}' # sentence creation without hypen
                for event in df["Code_event"]
            ]
        else:
            sentence = [   
                f'{types_dict[df["Type_event"].iloc[0]]}-{event.replace(" ","-")}' # sentence creation with hypen
                for event in df["Code_event"]         
            ]
    return sentence

def read_sentence_nl(df, types_dict):
    return df['Description_event'].tolist()
            
def create_nsp_format_3(file_path, use_time=False, dont_use_hypen=False):   
    df, types_dict = read_csv_format_3(file_path)
    logging.info(f'Read csv input file with {len(df)} rows.')
        
    bag = []
    for _,patient_df in tqdm(df.groupby('Assistito_CodiceFiscale_Criptato'), desc='Creating bags of sentences'):
        sentences = []
        for _,sentence_df in patient_df.groupby('sentence'):
            sentence = read_sentence(sentence_df, types_dict, use_time, dont_use_hypen)
            sentences.append(' '.join(sentence))
        # make it cronologically ordered
        sentences = sentences[::-1]
        bag.extend(sentences)    
    bag_size = len(bag)
    
    pairs = []
    n_patients = 0
    for _,patient_df in  tqdm(df.groupby('Assistito_CodiceFiscale_Criptato'), desc='Creating pairs of sentences'):
        n_patients += 1
        sentences = []
        for _,sentence_df in patient_df.groupby('sentence'):
            sentence = read_sentence(sentence_df, types_dict, use_time, dont_use_hypen)
            sentences.append(' '.join(sentence))
        # make it cronologically ordered
        sentences = sentences[::-1]
        
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
    
    logging.info(f'Number of patients = {n_patients}')
    logging.info(f'Created {len(pairs)} pairs (average {len(pairs)/n_patients:.2f} pairs/patient)')
    return pairs
            
def create_finetune_format_3(file_path, use_time=False, dont_use_hypen=False, dont_use_augm=False, use_nl=False):
    df, types_dict = read_csv_format_3(file_path)
    logging.info(f'Read csv input file with {len(df)} rows.')
    
    n_patients = 0
    sum_labels = 0
    n_hosp_event = 0
    
    docs = []
    for _,patient_df in tqdm(df.groupby('Assistito_CodiceFiscale_Criptato'), desc='Creating docs'):
        sentences = []
        dates = []
        hosp_mask = []
        med_mask = []
        n_patients += 1
        
        # creating sentences and date 
        for _,sentence_df in patient_df.groupby('sentence'):
            if use_nl:
                sentence = read_sentence_nl(sentence_df, types_dict)
            else:
                sentence = read_sentence(sentence_df, types_dict, use_time, dont_use_hypen)
            date = sentence_df['Data'].iloc[-1] # the date is not always the same, we use tha last one cronologically
            sentences.append(' '.join(sentence)+ ' [SEP]')
            dates.append(date)
            
            event_labels = sentence_df['Label_event'].unique()            
            hosp_mask.append('Dimissioni - RO' in event_labels)
            med_mask.append('Farmaci D' not in event_labels and 'Farmaci S' not in event_labels)
            # logging.info(f"Label_event = {list(sentence_df['Label_event'])}")
            # logging.info(f"Dates = {list(sentence_df['Data'])}")
            
        # cronologically order the sentences and dates
        sentences = sentences[::-1]
        dates = dates[::-1]
        hosp_mask = hosp_mask[::-1]
        med_mask = med_mask[::-1]
        
        n_hosp_event += sum(hosp_mask)
        
        if dont_use_augm: # do not use augmentation 
            previous_date=None
            # look for the previous hosp event
            i=len(dates)-2 # we dont start from the last since we already know its an hosp event
            while len(dates) > 2 and i>0:
                if med_mask[i]:
                    previous_date=dates[i]
                    break
                i-=1
            label = int(previous_date is not None and is_less_than_3_month(previous_date, dates[-1]))
            # if label and n_patients < 200:
            #     logging.info(f"Positive CF = {sentence_df['Assistito_CodiceFiscale_Criptato'].iloc[0]}")
            # elif n_patients < 200:
            #     logging.info(f"Negative CF = {sentence_df['Assistito_CodiceFiscale_Criptato'].iloc[0]}")
            sum_labels += label
            doc = '[CLS] '+ ' '.join(sentences[:-1])+ f' <end> {label}' # the last event has to be used only as label
            docs.append(doc)
        else: # use augmentation
            # creating docs for training deleting from last
            i = len(sentences) -1 # we start from the last, since has to be used as label
            while i > 0:
                if hosp_mask[i]: # we have found an hospedalisation event
                    # the label has to be calculated seeing this date and the previous (if exists)
                    previous_date = None
                    j=i-1 # we start looking for the previous hosp event from this one
                    while j>0:
                        if med_mask[j]: # we have found the previous hosp event
                            previous_date = dates[j] 
                            break # found it, we can exit
                        j-=1
                    label = int(previous_date is not None and is_less_than_3_month(previous_date,dates[i]))
                    sum_labels += label # for stats reason
                    
                    # create a sequence up to the med event before this one, use this one as label
                    doc = '[CLS] '+ ' '.join(sentences[:i])+ f' <end> {label}'
                    docs.append(doc)
                i -= 1
            
            # creating docs for training deleting from first
            previous_date = None
            i = len(sentences) -2 # we know that the last is an hospedalisation, we look for the previous one
            while len(dates) > 2 and i > 0:
                if med_mask[i]:
                    previous_date = dates[i]
                    break
                i-=1
            label = int(previous_date is not None and is_less_than_3_month(previous_date, dates[-1]))
            sum_labels += label * (len(sentences) - 2)
            docs.extend(['[CLS] '+ ' '.join(sentences[i:])+ f' <end> {label}' for i in range(len(sentences) -2)])
            
    logging.info(f'Number of patients = {n_patients}')
    logging.info(f'{n_hosp_event} hospitalisation events (average {n_hosp_event/n_patients:.2f} per patient)')
    logging.info(f'Created {len(docs)} sequences (average {len(docs)/n_patients:.2f} sequences/patient)')
    logging.info(f'Number of positive labels: {sum_labels}/{len(docs)} ({sum_labels/len(docs)*100:.2f}%)')
    return docs  
        
            
def create_mlm_only_format_3(file_path, use_time=False, dont_use_hypen=False):        
    df, types_dict = read_csv_format_3(file_path)
    logging.info(f'Read csv input file with {len(df)} rows.')
    
    docs = []
    n_patients = 0
    n_sentences = 0
    for _,patient_df in tqdm(df.groupby('Assistito_CodiceFiscale_Criptato'), desc='Creating output lists'):
        sentences = []
        n_patients += 1
        
        for _,sentence_df in patient_df.groupby('sentence'):
            sentence = read_sentence(sentence_df, types_dict, use_time, dont_use_hypen)
            sentences.append(' '.join(sentence)+ " [SEP]")
            n_sentences += 1
        docs.append('[CLS] '+' '.join(sentences[::-1]))
        
    logging.info(f'Number of patients = {n_patients}')
    logging.info(f'Each patient has in average {n_sentences/n_patients:.2f} sentences')
    logging.info(f'Created {len(docs)} docs (average {len(docs)/n_patients:.2f} pairs/patient)')
    return docs

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 
#   Format_4: .txt dataset made of sentences - documents
# 
#
            
def create_nsp_format_4(file_path):
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
    
    pairs = []
    for sentence_a, sentence_b, label in tqdm(zip(sentences_a, sentences_b, labels), desc='Creating nsp dataset'):
        pairs.append(f'[CLS] {sentence_a} [SEP] {sentence_b} <end> {label}')
        
    return pairs

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
#   MAIN FUNCTION
#
#
            
def detect_format(file_path):
    exp = os.path.splitext(file_path)[-1].lower()
    if exp == '.txt': return 'format_4'
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        detected_columns = next(reader)

    for format,columns in FORMAT_COLUMNS.items():
        if detected_columns == columns:
            return format
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--file_path', type=str,
                        help='Folder where are located the input csv files')
    parser.add_argument('--output_name', type=str, default='dataset.txt',
                        help='Name for the output text file')
    parser.add_argument('--output_folder', type=str,
                        help='Folder where to save the output text file')
    parser.add_argument('--create_pretrain', action='store_true')
    parser.add_argument('--create_finetuning', action='store_true')
    parser.add_argument('--create_infer', action='store_true')
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--test_size', default=.2, type=float)
    # parser.add_argument('--create_nsp_text_file', action='store_true')
    # parser.add_argument('--create_mlm_only_dataset', action='store_true')
    # parser.add_argument('--create_nsp_class_text_data', action='store_true')
    parser.add_argument('--mlm_only', action='store_true')
    parser.add_argument('--del_beginning', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--use_time', action='store_true', 
                        help='This command embed temporal information in the text dataset. The month of the event will be included with the ICD-9 code.')
    parser.add_argument('--dont_use_hypen', action='store_true',
                        help='Use this argument if you want to generate text without the hypen that separates the ICD code from the dictionary it comes from.' \
                            +'The result will be a string with both strings concatenated.')
    parser.add_argument('--dont_use_augm', action='store_true', help='Use this parameter to not use augmentation while creating finetuning datasets')
    parser.add_argument('--use_nl', action='store_true', help='Use this parameter to use natural language in the dataset')
        
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
    logging.info(f'The output file will be saved in {args.output_folder}')
    setup_logging(args.output_folder, console="debug")
    
    format = detect_format(args.file_path)
    
    docs = None
    if format == 'format_2':    
        if args.create_pretrain:
            if args.mlm_only:
                docs = create_mlm_only_format_2(args.file_path)
            else:
                docs = format_2_create_nsp(args.file_path)
        if args.create_infer:
            docs = format_2_create_infer(args.file_path)
        if args.create_finetuning:
            docs = format_2_create_finetune(args.file_path)
            
    elif format == 'format_3':        
        if args.create_finetuning:
            docs = create_finetune_format_3(args.file_path, args.use_time, args.dont_use_hypen, args.dont_use_augm, args.use_nl)
        if args.create_pretrain:
            if args.mlm_only:
                docs = create_mlm_only_format_3(args.file_path, args.use_time, args.dont_use_hypen)
            else:
                docs = create_nsp_format_3(args.file_path, args.use_time, args.dont_use_hypen)
        if args.create_infer:
            pass
            
    elif format == 'format_4':    
        if args.create_pretrain:
            docs = create_nsp_format_4(args.file_path)
            
    if docs is None:
        logging.info('Not recognized format')
        
    elif args.split and not args.create_infer:
        train,test = train_test_split(docs, test_size=args.test_size, random_state=args.random_state, shuffle=True)
        output_files = [os.path.join(args.output_folder, 'train.txt'),os.path.join(args.output_folder,'test.txt')]
        print('Creating train and text output files')
        for output_file,split in zip(output_files,[train,test]):
            with open(output_file, 'w') as file:
                file.write('\n'.join(split))
    else:
        if args.create_infer:
            logging.info(f'Creation of inference dataset. Dataset will not be split into train and test')
        with open(os.path.join(args.output_folder, args.output_name), 'w') as file:
            file.write('\n'.join(docs))