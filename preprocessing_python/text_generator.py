import pandas as pd
import os
from tqdm import tqdm

def create_text_from_data(data_folder, file_name, text_generated_name):
    output_path = os.path.join(data_folder, text_generated_name)
    if not os.path.exists(output_path):
        df = pd.read_csv(os.path.join(data_folder, file_name), index_col=0)
        grouped_df = df.groupby('keyone')

        results = []
        for _, patient in tqdm(grouped_df, desc='Producing text file from csv dataset'):
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

        results = '\n'.join(results)

        with open(os.path.join(data_folder, text_generated_name), 'w') as file:
            file.write(results)
    else:
        print(f'Output text file found in {output_path}')
        
if __name__ == '__main__':
    
    data_folder = 'bert_medical_records/data'
    file_name = 'base_red3.csv'
    output_name = 'output.txt'
    print(os.environ.get(data_folder))

    create_text_from_data(data_folder, file_name, output_name)