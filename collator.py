from transformers import DataCollatorForLanguageModeling
import torch 

def define_collator(tokenizer):
    return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.2,return_tensors="pt"
        )