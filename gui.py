import streamlit as st
import pandas as pd

st.write("""
# Bert Medical Records (EHR)     
""")

def click_button():
    st.session_state.clicked = True

# Carica il file CSV
uploaded_file = st.file_uploader("Upload your .csv file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    rows = st.radio("Select the rows to use to pre-train the model",
                    ["All", "Select start and end index"])
    
    # upload_all = st.checkbox('Upload all rows', value=True)
    # select_rows = st.checkbox('Select rows', value=False)
    
    if rows == 'All':
        st.write("""Preview data:""")
        st.write(data)
    
    # Mostra il dataframe all'utente e permetti la selezione delle righe
    if rows == "Select start and end index":
        start_index = st.number_input('Select the start index', min_value=0, max_value=len(data)-1, step=1)
        end_index = st.number_input('Select the end index', min_value=start_index, max_value=len(data)-1, step=1, value=len(data)-1)
        
        selected_rows = data.iloc[start_index:(end_index+1)]
        st.write("""Preview data:""")
        st.write(selected_rows)
    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
        
    st.button('Click me', on_click=click_button)
    
    if st.session_state.clicked:
        st.write('Producing text dataset from csv...')
        
    # if not st.session_state.get('button_clicked', False):
    #     if st.button('Esegui'):
    #         st.session_state.button_clicked = True
    # else:
    #     st.write('Il pulsante è stato cliccato!')
        

    # Mostra il dataframe all'utente e permetti la selezione delle righe
    # selected_indices = st.multiselect('Seleziona le righe che vuoi includere', data.index)
    # selected_rows = data.loc[selected_indices]