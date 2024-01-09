from transformers import DataCollatorForLanguageModeling
import torch
import numpy as np

def define_collator(tokenizer):
    
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2, return_tensors="pt"
        )