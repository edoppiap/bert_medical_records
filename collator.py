from transformers import DataCollatorForLanguageModeling #This is a utility that helps in creating batches of data for language modeling, especially for MLM tasks.
import torch
import numpy as np

def define_collator(tokenizer):
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2, return_tensors="pt"
        )

'''
??a little confused
It seems this is a data collator for training language models using the masked language modeling technique. 
Tihis will handle the random masking of tokens in the input data.
But I think you already wrote similar one in load_ dataset.py
'''