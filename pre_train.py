from transformers import TrainingArguments, Trainer, AdamW, BertModel
import os
from tqdm import tqdm
import torch

from load_dataset import PreTrainingDataset

def pre_train(model, data_collator, train_dataset, test_dataset, output_path):
    print('Defining training Arguments...')
    training_args = TrainingArguments(
        output_dir=output_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=10,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        report_to='none',
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    
    print('Do the pre-train...')
    # initialize the trainer and pass everything to it
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    print('End pre-train.')
    
    trainer.save_model(os.path.join(output_path, 'model'))
    
def run_lm_pretrain(model, optim, loader: PreTrainingDataset, n_epochs: int = 2):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    model.to(device)
    model.train()
    
    for epoch in range(n_epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    next_sentence_label = next_sentence_label,
                    labels = labels)
            
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())