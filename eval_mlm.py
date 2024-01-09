from transformers import pipeline

import os
import random
from tqdm import tqdm
import streamlit as st
from torch.cuda import is_available
from stqdm import stqdm

def calculate_mlm_recall(model, tokenizer, folder, streamlit=False):
    eval_path = os.path.join(folder, 'test.txt')
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, device='cuda' if is_available() else 'cpu')
    
    tokens = []
    with open(eval_path, 'r') as file:
        for line in file:
            if not line == '\n':
                line = line.removeprefix('[CLS]').removesuffix(' \n')
                sentences = line.split('[SEP]')[:-1]
                for sentence in sentences:
                    tokens.append(sentence.split(' ')[1:-1])

    count = 0
    found = {i: 0 for i in range(1,6,2)}
    
    progress_text = 'Performing evaluation'
    
    if streamlit:
        loop = stqdm(tokens, desc=progress_text)
    else:
        loop = tqdm(tokens, desc=progress_text)

    for sentence in loop:
        i = random.randint(0, len(sentence)-1)
        original = ' '.join(sentence)
        sentence[i] = fill.tokenizer.mask_token
        masked_sentence = ' '.join(sentence)
        results = fill(masked_sentence)
        count+=1

        for recall_i in range(1, 6, 2):
            for result in results[:recall_i]:
                if result['sequence'] == original:
                    found[recall_i] += 1
                    break

    eval_str = f'\nR@1: {found[1]/count:.4f} - R@3: {found[3]/count:.4f} - R@5: {found[5]/count:.4f}'
    print(eval_str)
    if streamlit:
        st.write('Model performance:',eval_str)