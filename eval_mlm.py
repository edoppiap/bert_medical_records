from transformers import pipeline

import os
import random
from tqdm import tqdm

def calculate_mlm_recall(model, tokenizer, folder):
    eval_path = os.path.join(folder, 'eval.txt')
    fill = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)
    
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

    for sentence in tqdm(tokens, desc='Performing evaluation'):
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

    print(f'\nR@1: {found[1]/count} - R@3: {found[3]/count} - R@5: {found[5]/count}')