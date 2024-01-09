import streamlit as st
import pandas as pd 
import os
from datetime import datetime

from modeling import get_bert_model
from encoder import encode
from tokenizer import define_tokenizer
from preprocessing_python.text_generator import create_text_from_data
from load_dataset import dataset_loader
from collator import define_collator
from pre_train import pre_train

def click_button():
    st.session_state.clicked = True

def app_run(output_folder):

    st.set_page_config(page_title="Pre-training BERT App", page_icon="🔗", layout="wide")

    st.title("Pre-training BERT App")
    st.subheader('Electronic Health Records Based Design')

    # ------------------------------ upload and process data -----------------------------------
    with st.expander("Explanation of the Electronic Health Records "):
        st.write("""
             ## A row represents a hospitalisation record, which consists of the following attributes
            - keyone: The ID of the patient
            - GENERE: The gender of the patient
            - eta_inizio: The age of the patient
            - ATC7: Medication prescription that be represented as ATC codes
            - TIPO_RIC_00: typy of hosipitalisation
            - DIA_PRIN: The main dignosis
            - DIA_CINQUE: The 5th dignosis
            - DIA_QUATTRO: The 4th dignosis
            - DIA_TRE: The 3rd dignosis
            - DIA_DUE: The 2nd dignosis
            - DIA_UNO: The 1st dignosis
            - GG_RIC: Hospitalization start date-day
            - MM_RIC: Hospitalization start date-month
            - AA_RIC: Hospitalization start date-year
            - GG_DIM: Hospitalization end date-day
            - MM_DIM: Hospitalization end date-month
            - AA_DIM: Hospitalization end date-year

        """)

        
    uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=["csv","txt"])
    if uploaded_file is not None:
        
        # Show raw data preview
        st.write("Raw data preview:")
        df = pd.read_csv(uploaded_file)
        
        rows = st.radio("Select the rows to use to pre-train the model",
                    ["All", "Select start and end index"])
        
        if rows == 'All':
            st.write("""Preview data:""")
            st.write(df)
            
        if rows == "Select start and end index":
            start_index = st.number_input('Select the start index', min_value=0, max_value=len(df)-1, step=1)
            end_index = st.number_input('Select the end index', min_value=start_index, 
                                        max_value=len(df)-1, step=1, value=len(df)-1)
            
            selected_rows = df.iloc[start_index:(end_index+1)]
            st.write("""Preview data:""")
            st.write(selected_rows)
            
        #st.button('Select rows', on_click=click_button)
        
        # Save uploaded files to the folder
        # data_folder = '/Users/yuting/Desktop/vs/GUI/data'  # The path to the directory where the file is saved needs to be replaced
        # file_path = os.path.join(data_folder, uploaded_file.name)
        # with open(file_path, "wb") as f:
        #     f.write(uploaded_file.getbuffer())

        output_name = st.text_input(label='Name for the generated text file', placeholder='text_dataset')
        
        if output_name != '':
            output_name += '.txt'
            # Calling the functions in the text_generator.py
            text_dataset_path = create_text_from_data(df, output_folder, output_name, streamlit=True)
                
            # Read the contents of the processed file
            #output_file_path = os.path.join(data_folder, output_name)
            if os.path.exists(text_dataset_path):
                with open(text_dataset_path, "r") as file:
                    text_dataset = file.read()

                # Show processed text
                st.subheader("Generated text to use as input for pre-train Bert")
                st.text_area("result", text_dataset, height=300)

                # Provide download link
                st.download_button(
                    label="Download the generated text",
                    data=text_dataset,
                    file_name=output_name,
                    mime="text/plain"
                )
            else:
                st.error("The processed file could not be found. Please check the processing function.")
                
            train_file, test_file, eval_file = 'train.txt', 'test.txt', 'eval.txt'
                
            d, files = dataset_loader(text_dataset_path, 
                    train_file_name=train_file, 
                    test_file_name=test_file,
                    eval_file_name=eval_file,
                    output_path=output_path)
            
            special_tokens = ['[CLS]','[SEP]','[MASK]']
            
            truncate_longer_samples = st.checkbox('Truncate longer samples (has to be true for now)', value=True)
            
            tok_name = st.selectbox(
                "Choose a tokenizer class",
                ('BertTokenizerFast', 'RetriBertTokenizer')
            )
            
            vocab_size = st.number_input(label='Vocabulary size', value=30_522)
            max_seq_length = st.number_input(label='Max lenght', value=512)
            
            if tok_name and vocab_size and max_seq_length:            
                tokenizer = define_tokenizer(special_tokens=special_tokens,
                                tokenizer_name=tok_name,
                                files=files, 
                                vocab_size=vocab_size, 
                                max_length=max_seq_length,
                                output_path=output_path)
                
                train_dataset, test_dataset = encode(d, tokenizer,
                                    max_length=max_seq_length,
                                    truncate_longer_samples=truncate_longer_samples)
                
                bert_class = st.selectbox(
                    "Choose a bert class",
                    options=('BertForMaskedLM', 'BertForNextSentencePrediction'), index=0
                )
                
                model = get_bert_model(bert_class, vocab_size, max_seq_length)
                
                data_collator = define_collator(tokenizer)
                
                if 'clicked' not in st.session_state:
                    st.session_state.clicked = False
                    
                st.button('Start train', on_click=click_button)
                
                if st.session_state.clicked:
                    pre_train(model=model,
                            data_collator=data_collator,
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            output_path=output_path)
        
    # Using object notation
    add_selectbox = st.sidebar.selectbox(
        "Choose a pre-train model",
        ("Masked Language Model (MLM)","Next Sentence Prediction (NSP)","MLM + NSP")
    )

    # Using "with" notation
    with st.sidebar:
        add_radio = st.radio(
            "Choose a pre-train model",
            [
                "Masked Language Model (MLM)",
                "Next Sentence Prediction (NSP)",
                "MLM + NSP",
            ],
            help="""the model to use for the pre-train.

            - Masked Language Model (MLM) (🐌)
            - Next Sentence Prediction (NSP) (💬)
            - MLM + NSP (💨)"""
        )

if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_path = os.path.join(current_directory, 'logs',current_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    app_run(output_path)