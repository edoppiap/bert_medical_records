from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from transformers.data.metrics import acc_and_f1

import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from custom_parser import parse_arguments
from load_dataset import FinetuningDataset, InferDataset
from optimizer import get_optimizer

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
            
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            
    model.save_pretrained(model_path)
    return loss

def compute_metrics(preds, truths):
    print(sum(preds==truths))
    return acc_and_f1(preds, truths)

def eval(args, test_dataset, model, output_folder):
    loader = DataLoader(test_dataset, batch_size=args.train_batch_size,shuffle=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    eval_loss = 0.0
    n_eval_step = 0
    preds = None
    truths = None
    
    loop = tqdm(loader, desc='Evaluating', leave=True)
    for batch in loop:
        model.eval()
        
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        
        temp_eval_loss, logits = outputs[:2]
        # pred = logits.detach().cpu().numpy()
        
        eval_loss += temp_eval_loss.mean().item()
        n_eval_step += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            truths = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            truths = np.append(truths, labels.detach().cpu().numpy(), axis=0)
            
        loop.set_postfix(loss=eval_loss)
    
    eval_loss = eval_loss / n_eval_step
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, truths)
    
    return result

def predict(args, data, model, output_folder):
    loader = DataLoader(data, batch_size=args.train_batch_size, shuffle=False)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    preds = None
    
    loop = tqdm(loader, desc='Create predicted labels', leave=True)
    for batch in loop:
        model.eval()
        
        input_ids = batch['input_ids'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)
        
        logits = outputs[0] # since label is not provided, the first value is the logits
        
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    
    return preds        

def main():
    args = parse_arguments()
    
    assert (args.do_train and args.do_eval) \
        or not args.do_eval \
        or (args.model_input and args.do_eval) \
        or (args.do_train and not args.do_eval) \
        or (args.use_pretrained_bert and args.do_eval), \
            '--do_eval present without one between --do_train, --model_input or --use_pretrained_bert. You need to train, pass or select a pretrain model to evaluate'
            
    assert (args.predict and args.model_input) \
        or (args.predict and args.use_pretrained_bert), \
            '--predict present without one between --model_input or --use_pretrained_bert. You need to pass a pretrained model to infer'
    
    if args.output_dir:
        output_path = args.output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_path = os.path.join(current_directory, 'logs',current_time)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
    print(f'Output files will be saved in folder: {output_path}')
    
    model_path = os.path.join(output_path, 'finetuned_model')
    
    if args.use_pretrained_bert:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
    else:    
        tokenzier_folder = os.path.join(args.model_input, 'tokenizer')
        if os.path.exists(tokenzier_folder):
            tokenizer = BertTokenizerFast.from_pretrained(tokenzier_folder)
        else:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        if args.do_train:
            model_folder = os.path.join(args.model_input, 'pre_trained_model')
            model = BertForSequenceClassification.from_pretrained(model_folder)
        elif args.predict:
            model = BertForSequenceClassification.from_pretrained(args.model_input)
    
    if not predict:
        dataset = FinetuningDataset(tokenizer, 
                                file_path=args.input_file, 
                                max_length=args.max_seq_length)
    else:
        dataset = InferDataset(tokenizer, 
                                file_path=args.input_file, 
                                max_length=args.max_seq_length)
        
    if not args.predict:
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8,.2])
    
        print(f'{len(train_dataset) = }\n{len(test_dataset) = }')
    
    if args.do_train:
        loss = train(args, train_dataset, model, model_path)
        print(f'Average loss = {loss}')
    if args.do_eval:
        result = eval(args, test_dataset, model, output_folder=model_path)
        
        print(f'{result = }')
    if args.predict:
        preds = predict(args, dataset, model, output_folder=model_path)
        
        df = pd.DataFrame({'keyone':dataset.patients, 'pred':preds})
        file_name = os.path.join(output_path, 'prediction.csv')
        df.to_csv(file_name, index=False)
    
    
if __name__ == '__main__':
    main()