from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
from transformers.data.metrics import acc_and_f1

import logging
import sys
import multiprocessing
import os
from datetime import datetime
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils import setup_logging
from custom_parser import parse_arguments
from load_dataset import NewFinetuningDataset, InferDataset
from optimizer import get_optimizer
from transformers import get_scheduler

def train(args, train_dataset, model, model_path, output_path):
    start_time = datetime.now()
    loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                        shuffle=True)
    
    optim = get_optimizer(parameters=model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(name=args.scheduler_name, 
                              optimizer=optim, 
                              num_warmup_steps=args.num_warmup_steps, 
                              num_training_steps=args.num_train_steps)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        loop = tqdm(loader, leave=True)
        epoch_start_time = datetime.now()
        for step,batch in enumerate(loop):
            
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
            
            if (step+1) % args.gradient_accumulation_steps == 0:
                optim.step()
                scheduler.step()
                optim.zero_grad()
                global_step += 1
                
                if global_step % args.save_checkpoints_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    checkpoint_path = os.path.join(output_path, f'{checkpoint_prefix}-{global_step}')
                    if not os.path.exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    model.save_pretrained(checkpoint_path)
                    torch.save(args, os.path.join(checkpoint_path, 'training_args.bin'))
                    logging.info(f'Saving model checkpoint to {checkpoint_path}')
            
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
            
            logging.info(f"Epoch {epoch:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {loss:.4f}")
            if args.num_train_steps > 0 and global_step > args.num_train_steps:
                loop.close()
                break
            
        logging.info(f"Epoch {epoch:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                  f"loss = {loss:.4f}")
        if args.num_train_steps > 0 and global_step > args.num_train_steps:
            break
    
    logging.info(f"Trained for {epoch + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")
    model.save_pretrained(model_path)
    torch.save(args, os.path.join(model_path, 'training_args.bin'))
    return loss


# Oltre ad F1 andrebbe calcolato anche il recall e precision
def compute_metrics(preds, truths):
    logging.info(sum(preds==truths))
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
        or (args.predict and args.use_pretrained_bert) \
        or not args.predict, \
            '--predict present without one between --model_input or --use_pretrained_bert. You need to pass a pretrained model to infer'
    
    if args.output_dir:
        output_path = args.output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    elif args.model_input:
        assert os.path.exists(args.model_input) and os.path.isdir(args.model_input), \
            'Path for model input not valid'
        output_path = args.model_input
    else:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_path = os.path.join(current_directory, 'logs',current_time)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
    setup_logging(output_path, console='debug')
    
    if args.model_input:
        model_folder = os.path.join(args.model_input, 'pre_trained_model')
        loaded_args = torch.load(os.path.join(model_folder, 'training_args.bin'))
        
        args.max_seq_length = loaded_args.max_seq_length
        
    logging.info(f'\nStart finetuning')
    logging.info(f'Arguments: {args}')
    logging.info(" ".join(sys.argv))    
    logging.info(f'Output files will be saved in folder: {output_path}')
    if output_path == args.model_input:
        logging.info('Saving the model in the same folder of the pretrained one')
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    
    model_path = os.path.join(output_path, 'finetuned_model')
    
    if args.use_pretrained_bert:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
    else:    
        tokenizer_folder = os.path.join(args.model_input, 'tokenizer')
        if os.path.exists(tokenizer_folder):
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_folder)
        else:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        if args.do_train:
            model_folder = os.path.join(args.model_input, 'pre_trained_model')
            logging.info(f'Loading pretrained model from {model_folder}')
            try:
                model = BertForSequenceClassification.from_pretrained(model_folder)
            except:
                model = BertForSequenceClassification.from_pretrained(args.model_input)
        elif args.predict:
            model = BertForSequenceClassification.from_pretrained(args.model_input)
    
    if not args.predict:
        if os.path.isfile(args.input_file):
            dataset = NewFinetuningDataset(tokenizer, 
                                file_path=args.input_file, 
                                max_length=args.max_seq_length)
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8,.2])
            if not args.do_train:
                del train_dataset
            if not args.do_eval:
                del test_dataset
        elif os.path.isdir(args.input_file):
            train_file = os.path.join(args.input_file, 'train.txt')
            test_file = os.path.join(args.input_file, 'test.txt')
            assert os.path.isfile(train_file) and os.path.isfile(test_file), \
                'Folder passed as input file but no train.txt or test.txt file found'
            if args.do_train:
                train_dataset = NewFinetuningDataset(tokenizer,
                                        file_path=train_file,
                                        max_length=args.max_seq_length)
            if args.do_eval:
                test_dataset = NewFinetuningDataset(tokenizer,
                                        file_path=test_file,
                                        max_length=args.max_seq_length)
    else:
        dataset = InferDataset(tokenizer, 
                                file_path=args.input_file,
                                max_length=args.max_seq_length)
        
    if args.do_train and args.do_eval:  
        logging.info(f'There are {len(train_dataset)} documents in the train datatset and {len(test_dataset)} in the evaluation one.')
    elif args.do_train:
        logging.info(f'Only the train will be performed. There are {len(train_dataset)} documents in the dataset')
    elif args.do_eval:
        logging.info(f'Only the evaluation will be performed. There are {len(test_dataset)} documents in the dataset')
    elif args.predict:
        logging.info(f'It will predict labels for the dataset. There are {len(dataset)} documents in the dataset.')
    
    if args.do_train:
        loss = train(args, train_dataset, model, model_path, output_path)
        logging.info(f'Average loss = {loss}')
    if args.do_eval:
        result = eval(args, test_dataset, model, output_folder=model_path)
        
        logging.info(f'{result = }')
    if args.predict:
        preds = predict(args, dataset, model, output_folder=model_path)
        
        df = pd.DataFrame({'keyone':dataset.patients, 'pred':preds})
        file_name = os.path.join(output_path, 'prediction.csv')
        df.to_csv(file_name, index=False)
    
    
if __name__ == '__main__':
    main()