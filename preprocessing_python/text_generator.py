import pandas as pd
import os
from tqdm import tqdm
from stqdm import stqdm
import streamlit as st

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
    
    data_folder = 'bert_medical_records/data'
    file_name = 'base_red3.csv'
    output_name = 'output.txt'
    print(os.environ.get(data_folder))

    create_text_from_data(data_folder, file_name, output_name)