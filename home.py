import streamlit as st
import pandas as pd 
import os
from datetime import datetime
from torch.cuda import is_available

from modeling import get_bert_model, get_model_from_path
from encoder import encode
from tokenizer import train_tokenizer, get_tokenizer_from_path
from preprocessing_python.text_generator import create_text_from_data
from load_dataset import dataset_loader
from collator import define_collator
from pre_train import pre_train
from eval_mlm import calculate_mlm_recall

def file_updated():
    st.session_state.upload = 1

def set_train_state(i):
    st.session_state.train = i
    
def set_eval_state(i):
    st.session_state.eval = i
    
def st_calculate_mlm_recall(folder_path):
    model = get_model_from_path(folder_path)
    tokenizer = get_tokenizer_from_path(folder_path)
    
    calculate_mlm_recall(model=model,
                        tokenizer=tokenizer,
                        folder=folder_path,
                        streamlit=True)

@st.cache_data
def get_output_path():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_path = os.path.join(current_directory, 'logs', current_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

@st.cache_data
def st_dataset(text_dataset_path, train_file_name, test_file_name, eval_file_name, output_path):
    return dataset_loader(text_dataset_path, 
                    train_file_name=train_file_name, 
                    test_file_name=test_file_name,
                    eval_file_name=eval_file_name,
                    output_path=output_path)
    
@st.cache_data
def st_create_text(df, output_folder, output_name):
    return create_text_from_data(df, output_folder, output_name, streamlit=True)

@st.cache_data
def st_tokenizer(special_tokens, tokenizer_name, files, vocab_size, max_length, output_path):
    return train_tokenizer(special_tokens=special_tokens,
                                    tokenizer_name=tokenizer_name,
                                    files=files, 
                                    vocab_size=vocab_size, 
                                    max_length=max_length,
                                    output_path=output_path)
    
@st.cache_data
def st_encode(_d, _tokenizer, max_length, truncate_longer_samples):
    return encode(_d, _tokenizer,max_length=max_length,truncate_longer_samples=truncate_longer_samples)

@st.cache_data
def st_get_bert_model(bert_class, vocab_size, max_seq_length):
    return get_bert_model(bert_class, vocab_size, max_seq_length)

@st.cache_data
def st_define_collator(_tokenizer):
    return define_collator(_tokenizer)

def st_pre_train(model, data_collator, train_dataset, test_dataset, output_path):
    pre_train(model=model,data_collator=data_collator,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_path=output_path) 

def app_run():

    st.set_page_config(page_title="Pre-training BERT App", page_icon="🔗", layout="wide")
    
    if 'train' not in st.session_state:
        st.session_state.train = 0
        
    if 'eval' not in st.session_state:
        st.session_state.eval = 0

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
        
    if st.session_state.train < 1 and st.session_state.eval < 1:
        st.button('Do pre-train from the beginning', on_click=set_train_state, args=[1])
        st.button('Do eval to an existing model', on_click=set_eval_state, args=[1])
    
    if st.session_state.train >=1:
        
        output_folder = get_output_path()
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], on_change=file_updated)
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            rows = st.radio("Select the rows to use to pre-train the model",
                        ["All", "Select start and end index"])
            
            if rows == 'All':
                st.write("""Preview data:""")
                with st.spinner('Loading DataFrame'):
                    st.write(df)
                selected_rows = df
                
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
                text_dataset_path = create_text_from_data(selected_rows, output_folder, output_name, streamlit=True)
                    
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
                
                d, files = st_dataset(text_dataset_path, 
                        train_file_name=train_file, 
                        test_file_name=test_file,
                        eval_file_name=eval_file,
                        output_path=output_folder)
                
                special_tokens = ['[CLS]','[SEP]','[MASK]']
                
                truncate_longer_samples = st.checkbox('Truncate longer samples (has to be true for now)', value=True)
                
                tok_name = st.selectbox(
                    "Choose a tokenizer class",
                    ('BertTokenizerFast', 'RetriBertTokenizer')
                )
                if tok_name:
                    st.session_state.tok_name = tok_name
                
                vocab_size = st.number_input(label='Vocabulary size', value=30_522)
                if vocab_size:
                    st.session_state.vocab_size = vocab_size
                max_seq_length = st.number_input(label='Max lenght', value=512)
                if max_seq_length:
                    st.session_state.max_seq = max_seq_length
                
                bert_class = st.selectbox(
                        "Choose a bert class",
                        options=('BertForMaskedLM', 'BertForNextSentencePrediction'), index=0
                    )
                
                cuda_toggle = st.toggle('Activate CUDA', value=is_available(), disabled=not is_available(), 
                                        help='Enable this option to use the GPU to improve training sensibly (only if available).')
        
        if st.session_state.get('tok_name', None) and \
            st.session_state.get('vocab_size', None) and \
                st.session_state.get('max_seq', None):
            if st.session_state.train < 2:
                st.button('Start train', on_click=set_train_state, args=[2])
        
            if st.session_state.train >= 2:
                    with st.spinner('Training and instantiating the tokenizer'):
                        tokenizer = st_tokenizer(special_tokens=special_tokens,
                                        tokenizer_name=tok_name,
                                        files=files, 
                                        vocab_size=vocab_size, 
                                        max_length=max_seq_length,
                                        output_path=output_folder)
                    
                    train_dataset, test_dataset = st_encode(d, tokenizer,
                                        max_length=max_seq_length,
                                        truncate_longer_samples=truncate_longer_samples)
                    
                    with st.spinner('Initializing bert class'):
                        model = st_get_bert_model(bert_class, vocab_size, max_seq_length)
                    
                    data_collator = st_define_collator(tokenizer)
                    
                    with st.spinner('Doing the pre-train'):
                        st_pre_train(model=model,
                                data_collator=data_collator,
                                train_dataset=train_dataset,
                                test_dataset=test_dataset,
                                output_path=output_folder)
                    
                    st.success('Pre-train ended successfully')
                    set_eval_state(1)
    
    if st.session_state.eval >= 1:
        folder_path = st.text_input('Enter the folder path where is stored the pre-trained version of BERT to evaluate')
        if folder_path and os.path.exists(folder_path): 
            st.session_state.folder_path = folder_path
            st.write(f"Selected folder path: {folder_path}")
            
            st_calculate_mlm_recall(folder_path)
                       
    # Using object notation
    # add_selectbox = st.sidebar.selectbox(
    #     "Choose a pre-train model",
    #     ("Masked Language Model (MLM)","Next Sentence Prediction (NSP)","MLM + NSP")
    # )

    # # Using "with" notation
    # with st.sidebar:
    #     add_radio = st.radio(
    #         "Choose a pre-train model",
    #         [
    #             "Masked Language Model (MLM)",
    #             "Next Sentence Prediction (NSP)",
    #             "MLM + NSP",
    #         ],
    #         help="""the model to use for the pre-train.

    #         - Masked Language Model (MLM) (🐌)
    #         - Next Sentence Prediction (NSP) (💬)
    #         - MLM + NSP (💨)"""
    #     )

if __name__ == '__main__':
    
    app_run()