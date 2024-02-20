from transformers import BertForMaskedLM, BertConfig, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizerFast
from datasets import load_metric
from transformers.data.metrics import acc_and_f1

import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_parser import parse_arguments
from modeling import get_bert_model
from tokenizer import train_tokenizer
from optimizer import get_optimizer
from load_dataset import PreTrainingDataset

def train(args, train_dataset, model, model_path):
    loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                        shuffle=True)
    
    optim = get_optimizer(parameters=model.parameters(), lr=args.learning_rate)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    model.train()
    
    for epoch in range(args.num_epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if 'next_sentence_label' in batch.keys():
                next_sentence_label = batch['next_sentence_label'].to(device)
                outputs = model(input_ids=input_ids,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask,
                        next_sentence_label = next_sentence_label,
                        labels = labels)
            else:
                outputs = model(input_ids=input_ids,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask,
                        labels = labels)
            
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            
    model.save_pretrained(model_path)
    return loss

def compute_metrics(nsp_preds,nsp_truths, mlm_preds, mlm_truths):
    print(f'Nsp metrics: {acc_and_f1(nsp_preds,nsp_truths)}')
    print(f'Mlm metrics: {acc_and_f1(mlm_preds,mlm_truths)}')
    

def eval(args, test_dataset, model, output_folder):
    loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    eval_loss = 0.0
    n_eval_step = 0
    nsp_preds = None
    nsp_truths = None
    mlm_preds = None
    mlm_truths = None
    
    loop = tqdm(loader, desc='Evaluating', leave=True)
    for batch in loop:
        model.eval()
        
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        if 'next_sentence_label' in batch.keys():
            next_sentence_label = batch['next_sentence_label'].to(device)
            outputs = model(input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    next_sentence_label = next_sentence_label,
                    labels = labels)
            
            nsp_logits = outputs[1][:, 1]
            
            if nsp_preds is None:
                nsp_preds = nsp_logits.detach().cpu().numpy()
                nsp_truths = next_sentence_label.detach().cpu().numpy()
            else:
                nsp_preds = np.append(nsp_preds, nsp_logits.detach().cpu().numpy(), axis=0)
                nsp_truths = np.append(nsp_truths, next_sentence_label.detach().cpu().numpy(), axis=0)
        else:
            outputs = model(input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    labels = labels)
            
        temp_eval_loss = outputs[0]
            
        if args.pre_train_tasks != 'nsp':
            mlm_logits = outputs[2 if args.pre_train_tasks == 'mlm_nsp' else 1]
            if mlm_preds is None:
                mlm_preds = mlm_logits[:, 1].detach().cpu().numpy()  # Extract predictions
                mlm_truths = labels.detach().cpu().numpy()  # Extract ground truths
            else:
                mlm_preds = np.append(mlm_preds, mlm_logits[:, 1].detach().cpu().numpy(), axis=0)  # Append predictions
                mlm_truths = np.append(mlm_truths, labels.detach().cpu().numpy(), axis=0)
        
        eval_loss += temp_eval_loss.cpu().item()
        n_eval_step += 1
        loop.set_postfix(loss=eval_loss)
        
    eval_loss = eval_loss / n_eval_step
    if nsp_preds is not None:
        nsp_preds = np.argmax(nsp_preds, axis=1)
    if mlm_preds is not None:
        mlm_preds = np.argmax(mlm_preds, axis=1)
    
    result = compute_metrics(nsp_preds,nsp_truths,mlm_preds,mlm_truths)
    return result
        
def main():
    args = parse_arguments()
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_path = os.path.join(current_directory, 'logs',current_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print(f'Output files will be saved in folder: {output_path}')
    
    tokenizer_path = os.path.join(output_path, 'tokenizer')
    model_path = os.path.join(output_path, 'pre_trained_model')
    
    if args.pre_train_tasks is not None:
        if args.pre_train_tasks == 'mlm':
            bert_class = 'BertForMaskedLM'
        elif args.pre_train_tasks == 'nsp':
            bert_class = 'BertForNextSentencePrediction'
        else:
            bert_class = 'BertForPreTraining'
    else:
        bert_class = args.bert_class
    
    if args.use_pretrained_bert:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    else:
        special_tokens = ['[CLS]','[SEP]','[MASK]']

        tokenizer = train_tokenizer(special_tokens=special_tokens,
                                    tokenizer_name=args.tokenizer_name,
                                    files=[args.input_file], 
                                    vocab_size=args.vocab_size, 
                                    max_length=args.max_seq_length,
                                    output_path=output_path)
        
        model = get_bert_model(bert_class, args.vocab_size, args.max_seq_length,
                               pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    
    dataset = PreTrainingDataset(tokenizer,
                                 file_path=args.input_file,
                                 mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0,
                                 nsp= bert_class == 'BertForPreTraining' or bert_class == 'BertForNextSentencePrediction')
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8,.2])
    
    if args.do_train:
        loss = train(args,train_dataset,model,model_path)
        print(f'Average loss = {loss}')
    
    if args.do_eval:
        result = eval(args, test_dataset, model, output_folder=model_path)
        print(f'{result = }')
    
if __name__ == '__main__':
    main()