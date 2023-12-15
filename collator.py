from transformers import DataCollatorForLanguageModeling
import torch
import numpy as np

class DataCollatorForMaskedLMNextSP(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer=None, mlm_probability=.15, return_tensors='pt'):
        super().__init__(tokenizer=tokenizer, return_tensors=return_tensors)
        self.mlm_probability = mlm_probability
        
    def collate_fn(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Separate MLM tokens from NSP tokens
        mlm_ids = input_ids[:, :max(input_ids.shape[1] // 2)]
        nsp_ids = input_ids[:, max(input_ids.shape[1] // 2) :]

        # Mask a random percentage of MLM tokens
        if self.mlm_probability > 0:
            mlm_mask = np.random.binomial(1, self.mlm_probability, size=mlm_ids.shape)
            mlm_ids[mlm_mask] = tokenizer.mask_token_id

        # Combine MLM and NSP tensors
        input_ids = np.concatenate([mlm_ids, nsp_ids], axis=1)

        # Create attention mask for combined tokens
        attention_mask = np.concatenate([attention_mask[:, :max(attention_mask.shape[1] // 2)], attention_mask[:, max(attention_mask.shape[1] // 2) :] + 1], axis=1)

        # Combine labels
        # labels = np.concatenate([next_sentence_labels, next_sentence_labels], axis=0)
        return input_ids, attention_mask

def define_collator(pre_train_tasks, 
                    tokenizer):
    data_collator = None
    
    if pre_train_tasks == 'mlm':
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.2, return_tensors="pt"
            )
    elif pre_train_tasks == 'nsp':
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, return_tensors="pt"
            )
    
    if data_collator is None:
        raise ValueError(f'Invalid pre-train task {pre_train_tasks}')
    return data_collator