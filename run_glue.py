from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
# from transformers.data.metrics import acc_and_f1
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import logging
import sys
import multiprocessing
import os
from datetime import datetime
import torch
import torch.nn as nn
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
    return model, loss


def compute_metrics(preds, truths, output_path, save_images=False):
    accuracy = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='binary')
    recall = recall_score(truths, preds, average='binary')
    precision = precision_score(truths, preds, average='binary')
    mcc = matthews_corrcoef(truths,preds)
    
    conf_matr = confusion_matrix(truths, preds)
    tn, fp, _, _ = conf_matr.ravel()
    specificity = tn / (tn + fp)
    
    if (save_images):
        cmtx = pd.DataFrame(
            conf_matr, 
            index=['P', 'N'], 
            columns=['PP', 'PN']
        )
        logging.debug(f'Confusion matrix:\nTot prediction={conf_matr.sum()}\n{cmtx}')
        logging.debug(f'An image of the confusion matrix is saved in {output_path}')
        matr_path = os.path.join(output_path, 'conf_matr_0.png')
        txt_path = os.path.join(output_path, 'classified_sentences_0.txt')
        i=0
        while True:
            if os.path.isfile(matr_path):
                i+=1
                matr_path = os.path.join(output_path,f'conf_matr_{i}.png')
                txt_path = os.path.join(output_path,f'classified_sentences_{i}.txt')
            else:
                break
                
        
        disp = ConfusionMatrixDisplay(conf_matr, display_labels=['Negative', 'Positive'])
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig(matr_path)
        plt.close()
        
        with open(txt_path, 'w') as f:
            for i,(pred,truth) in enumerate(zip(preds,truths)):
                f.write(f'Sentence {i}: {pred}/{truth} (predicted/truth)\n')
    
    # return acc,f1,recall,precision,specificity
    return {
        'acc':accuracy,
        'mcc':mcc,
        'f1':f1,
        'rec':recall,
        'pre':precision,
        'spe':specificity
    }

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
    result = compute_metrics(preds, truths, output_folder, save_images=args.debug)
    
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

def load_training_args(args):
    # TODO: if an input model has been passed, do some check to see if the arguments are correctly passed.
    # Override the parser arguments and load the saved ones.
    
    finetune_model_folder = os.path.join(args.model_input, 'finetuned_model')
    pretrain_model_folder = os.path.join(args.model_input, 'pre_trained_model')
    if os.path.exists(os.path.join(args.model_input, 'training_args.bin')):
        loaded_args = torch.load(os.path.join(args.model_input, 'training_args.bin'))
        logging.debug('Loaded training arguments from provided folder')
    elif os.path.exists(os.path.join(finetune_model_folder, 'training_args.bin')):
        loaded_args = torch.load(os.path.join(finetune_model_folder, 'training_args.bin'))
        logging.debug('Loaded training arguments from finetuned_model')
    elif os.path.exists(os.path.join(pretrain_model_folder, 'training_args.bin')):
        loaded_args = torch.load(os.path.join(pretrain_model_folder, 'training_args.bin'))
        logging.debug('Loaded training arguments from pretrained_model')
    else:
        logging.debug('No saved training arguments found')
        return args
    
    if args.max_seq_length != loaded_args.max_seq_length:
        logging.warning('Detected loaded arguments different from command arguments. BERT config arguments will override the specified ones.')
        args.max_seq_length = loaded_args.max_seq_length
        args.hidden_size = loaded_args.hidden_size
        args.num_hidden_layers = loaded_args.num_hidden_layers
        args.num_attention_heads = loaded_args.num_attention_heads
        args.intermediate_size = loaded_args.intermediate_size
        args.hidden_act = loaded_args.hidden_act
        args.hidden_dropout_prob = loaded_args.hidden_dropout_prob
        args.attention_probs_dropout_prob = loaded_args.attention_probs_dropout_prob
        args.initializer_range = loaded_args.initializer_range
        args.layer_norm_eps = loaded_args.layer_norm_eps
        args.type_vocab_size = loaded_args.type_vocab_size        
        
    return args

def load_model(model_input, tokenizer_folder=None, use_pretrained=False, do_train=False, do_eval=False, predict=False):
    
    if use_pretrained:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    else:
        if tokenizer_folder == None:
            tokenizer_folder = os.path.join(model_input, 'tokenizer')
        if os.path.exists(tokenizer_folder):
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_folder)
            logging.debug(f'Loaded custom tokenizer from {tokenizer_folder}')
        else:
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            logging.debug(f'Loaded pretrained tokenizer from HuggingFace (bert-base-uncased)')
            
        if do_train or do_eval:
            try:
                model = BertForSequenceClassification.from_pretrained(model_input)
                logging.debug(f'Loaded pretrained/finetuned model from selected folder {model_input}')
            except:
                try:
                    finetuned_folder = os.path.join(model_input, 'finetuned_model')
                    model = BertForSequenceClassification.from_pretrained(finetuned_folder)
                    logging.debug(f'Loading pretrained model from {finetuned_folder}')
                except:
                    logging.debug(f'Trying loading model from pre_trained_model folder')
                    pretrained_folder = os.path.join(model_input, 'pre_trained_model')
                    model = BertForSequenceClassification.from_pretrained(pretrained_folder)
                    logging.debug(f'Loading pretrained model from {pretrained_folder}')
        elif predict:
            model = BertForSequenceClassification.from_pretrained(model_input)
        
    return tokenizer, model

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
            
    setup_logging(output_path, console='debug' if args.debug else 'info')
    
    if args.model_input:
        args = load_training_args(args)
        # model_folder = os.path.join(args.model_input, 'pre_trained_model')
        # loaded_args = torch.load(os.path.join(args.model_input, 'training_args.bin'))
        # args.max_seq_length = loaded_args.max_seq_length
    
    if args.test_split == 0 and args.do_eval:
        args.do_eval = False
        
    if args.random_seed is not None:
        logging.debug(f'Setting the seed as {args.random_seed}')
        torch.manual_seed(args.random_seed)
        
    logging.info(f'Start finetuning')
    logging.info(f'Arguments: {args}')
    logging.info(" ".join(sys.argv))    
    logging.info(f'Output files will be saved in folder: {output_path}')
    if output_path == args.model_input:
        logging.debug('Saving the model in the same folder of the pretrained one')
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    
    # TODO:
    #   PRINTARE IL SEED
    #   SALVARE IN QUALCHE MODO I PESI DEL MODELLO PRETRAINATO
    
    if args.save_finetuned_folder is not None:
        finetune_save_folder = os.path.join(args.save_finetuned_folder, 'finetuned_model')
        if os.path.exists(finetune_save_folder) and args.do_train:
            logging.warning('Finetuned folder already present. This folder will be overwritten. If you want to perform a further finetune, be aware which model you are passing to the --model_input folder to avoid losing the wrong model.')
        model_path = finetune_save_folder
    else:
        model_path = os.path.join(output_path, 'finetuned_model')
            
    
    tokenizer, model = load_model(args.model_input,
                                  tokenizer_folder=args.pre_trained_tokenizer_path,
                                  use_pretrained=args.use_pretrained_bert,
                                  do_train=args.do_train,
                                  do_eval=args.do_eval,
                                  predict=args.predict)    
    
    if not args.predict:
        if os.path.isfile(args.input_file):
            if args.do_train and not args.do_eval:
                logging.debug(f'Loading the entire dataset to perform training')
                train_dataset = NewFinetuningDataset(tokenizer,
                                               file_path=args.input_file,
                                               max_length=args.max_seq_length)
            elif not args.do_train and args.do_eval:
                logging.debug(f'Loading the entire dataset to perform evaluation')
                test_dataset = NewFinetuningDataset(tokenizer,
                                               file_path=args.input_file,
                                               max_length=args.max_seq_length)
                labels = [test_dataset[i]['labels'].item() for i in tqdm(range(len(test_dataset)), desc='Getting labels')]
                logging.info(f'Positive labels: {labels.count(1)}/{len(labels)}, {labels.count(1)/len(labels)*100:.2f}%')
            else:
                logging.debug(f'Splitting the dataset in {(1-args.test_split)*100:.2f}% train and {args.test_split*100:.2f}% test')
                dataset = NewFinetuningDataset(tokenizer, 
                                    file_path=args.input_file, 
                                    max_length=args.max_seq_length)
                labels = [dataset[i]['labels'].item() for i in tqdm(range(len(dataset)), desc='Getting labels')]
                logging.info(f'Positive labels: {labels.count(1)}/{len(labels)}, {labels.count(1)/len(labels)*100:.2f}%')
                train_indices, test_indices = train_test_split(
                    range(len(dataset)),
                    test_size=args.test_split,
                    stratify=labels,
                    random_state=args.random_seed
                )
                train_dataset = torch.utils.data.Subset(dataset, train_indices)
                test_dataset = torch.utils.data.Subset(dataset, test_indices)
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
        logging.info(f'There are {len(train_dataset)} documents in the train dataset and {len(test_dataset)} in the evaluation one.')
    elif args.do_train:
        logging.info(f'Only the train will be performed. There are {len(train_dataset)} documents in the dataset')
    elif args.do_eval:
        logging.info(f'Only the evaluation will be performed. There are {len(test_dataset)} documents in the dataset')
    elif args.predict:
        logging.info(f'It will predict labels for the dataset. There are {len(dataset)} documents in the dataset.')
    
    if args.do_train:
        model, loss = train(args, train_dataset, model, model_path, output_path)
        logging.info(f'Average loss = {loss}')
    if args.do_eval:
        result = eval(args, test_dataset, model, output_folder=output_path)
        
        logging.info(f'{result = }')
    if args.predict:
        preds = predict(args, dataset, model, output_folder=model_path)
        
        df = pd.DataFrame({'keyone':dataset.patients, 'pred':preds})
        file_name = os.path.join(output_path, 'prediction.csv')
        df.to_csv(file_name, index=False)
    
    
if __name__ == '__main__':
    main()