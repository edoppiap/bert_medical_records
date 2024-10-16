from transformers import BertForMaskedLM, BertConfig, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizerFast
from transformers.data.metrics import acc_and_f1
from transformers import get_scheduler
# from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

import os
import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import logging
import multiprocessing

from utils import setup_logging
from custom_parser import parse_arguments
from modeling import get_bert_model, get_model_from_path
from tokenizer import train_tokenizer, get_tokenizer_from_path, get_tokenizer
from optimizer import get_optimizer
from load_dataset import PreTrainingDataset, NewPreTrainingDataset

def train(args, train_dataset, model, output_path):
    start_time = datetime.now()
    loader = DataLoader(train_dataset, batch_size=args.train_batch_size,
                        shuffle=True)
    
    optim = get_optimizer(parameters=model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_scheduler(name=args.scheduler_name, 
                              optimizer=optim, 
                              num_warmup_steps=args.num_warmup_steps, 
                              num_training_steps=args.num_train_steps)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model.to(device)
    model.train()
    global_step = 0
    
    for epoch in range(args.num_epochs):
        epoch_start_time = datetime.now()
        loop = tqdm(loader, leave=True)
        for step,batch in enumerate(loop):

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
                    logging.info(f"Saving model checkpoint to {checkpoint_path}")
            
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
    model.save_pretrained(output_path)
    torch.save(args, os.path.join(output_path, 'training_args.bin'))
    return loss

def compute_metrics(nsp_preds,nsp_truths, mlm_preds, mlm_truths):
    result = {}
    if nsp_preds is not None:
        nsp_acc = torch.sum(nsp_preds == nsp_truths).item() / len(nsp_truths)
        logging.info(f'Nsp acc: {nsp_acc}')
        result['nsp_acc'] = nsp_acc
    if mlm_preds is not None:
        mlm_acc = torch.sum(mlm_preds == mlm_truths).item() / len(mlm_truths)
        logging.info(f'Mlm acc: {mlm_acc}')
        result['mlm_acc'] = mlm_acc
    # logging.info(f'recall_at_k: {recall_score(mlm_truths.numpy(), top_k_indeces.numpy(), average="samples")}')
    return result
    

def eval(args, test_dataset, model, mask_token_id):
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
            
            nsp_logits = outputs[2 if args.pre_train_tasks == 'mlm_nsp' else 1]
            
            if nsp_preds is None:
                nsp_preds = nsp_logits.detach().cpu()
                nsp_truths = next_sentence_label.detach().cpu()
            else:
                nsp_preds = torch.cat((nsp_preds, nsp_logits.detach().cpu()), dim=0)
                nsp_truths = torch.cat((nsp_truths, next_sentence_label.detach().cpu()), dim=0)
        else:
            outputs = model(input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    labels = labels)
            
        temp_eval_loss = outputs[0]
            
        if args.pre_train_tasks != 'nsp':
            mlm_logits = outputs[1]
            
            mask = (input_ids != labels).cpu()

            if mask.any():
                # logging.info(f'\n{torch.topk(mlm_logits.detach().cpu()[mask,:], 5, dim=1) = }')
                # mlm_pred = torch.argmax(mlm_logits.detach().cpu()[mask,:], dim=1)
                # # logging.info(f'{mlm_pred}')
                # # _, top_k_indices = torch.topk(mlm_pred[mask], 5, dim=1)
                # logging.info(f'{mlm_pred = }\n{labels.detach().cpu()[mask] = }')
                if mlm_preds is None:
                    mlm_preds = mlm_logits.detach().cpu()[mask,:]  # Extract predictions
                    mlm_truths = labels.detach().cpu()[mask]  # Extract ground truths
                else:
                    mlm_preds = torch.cat((mlm_preds, mlm_logits.detach().cpu()[mask,:]), dim=0)  # Append predictions
                    mlm_truths = torch.cat((mlm_truths, labels.detach().cpu()[mask]), dim=0)
        
        eval_loss += temp_eval_loss.cpu().item()
        n_eval_step += 1
        loop.set_postfix(loss=temp_eval_loss.item())
        
    eval_loss = eval_loss / n_eval_step
    if nsp_preds is not None:
        nsp_preds = torch.argmax(nsp_preds, dim=1)
    if mlm_preds is not None:
        mlm_preds = torch.argmax(mlm_preds, dim=1)
        # _,top_k_indeces = torch.topk(mlm_preds, 5, dim=1)
        # logging.info(f'{top_k_indeces = }\n{mlm_truths}')
    
    # logging.info(f'{nsp_preds.size() = } - {nsp_truths.size() = }')
    # logging.info(f'{mlm_preds.size() = } - {mlm_truths.size() = }')
    
    result = compute_metrics(nsp_preds,nsp_truths,mlm_preds,mlm_truths)
    return result
        
def main():
    args = parse_arguments()    
    
    assert (args.do_train and args.do_eval) \
        or (args.model_input and args.do_eval) \
        or (args.do_train and not args.do_eval) \
        or (args.use_pretrained_bert and args.do_eval), \
            '--do_eval present without one between --do_train, --model_input or --use_pretrained_bert. You need to train, pass or select a pretrain model to evaluate'
    
    if args.output_dir:
        output_path = args.output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    else:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_path = os.path.join(current_directory, 'output',current_time)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
                
    setup_logging(output_path, console="debug")

    logging.info(f'Arguments: {args}')
    logging.info(" ".join(sys.argv))    
    logging.info(f'Output files will be saved in folder: {output_path}')
    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    logging.info(f'Pretrain task selected: {args.pre_train_tasks}')
    if args.pre_train_tasks == 'mlm':
        bert_class = 'BertForMaskedLM'
    elif args.pre_train_tasks == 'nsp':
        bert_class = 'BertForNextSentencePrediction'
    elif args.pre_train_tasks == 'mlm_nsp':
        bert_class = 'BertForPreTraining'
    
    if args.test_split == 0 and args.do_eval:
        args.do_eval = False 
        
    if args.model_input:
        model_path = os.path.join(args.model_input, 'pre_trained_model')
        tokenizer_path = os.path.join(args.model_input, 'tokenizer')
    else:
        model_path = None
        tokenizer_path = None
        
    tokenizer = get_tokenizer(path=tokenizer_path,
                                args=args,
                                output_path=output_path)
    if args.k_fold == 1:        
        model = get_bert_model(bert_class_name = bert_class,
                            args=args,
                            pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                            input_path=model_path)
    else:
        models = []
        for _ in range(args.k_fold):
            models.append(
                get_bert_model(bert_class_name = bert_class,
                            args=args,
                            pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
                            input_path=model_path)
            )
            
    if os.path.isfile(args.input_file):
        dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=args.input_file,
                                    max_length=args.max_seq_length,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
        if args.k_fold == 1:
            logging.info(f'Splitting the dataset in {1-args.test_split*100:.2f}% train and {args.test_split*100:.2f}% test')
            train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1-args.test_split,args.test_split])
        else:
            skf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
            train_dataset,test_dataset = [],[]
            for fold, (train_i, test_i) in enumerate(skf.split(dataset)):
                logging.info(f'Creating split dataset {fold+1}')
                train_dataset.append(Subset(dataset, train_i))
                test_dataset.append(Subset(dataset, test_i)) 
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
            train_dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=train_file,
                                    max_length=args.max_seq_length,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
        if args.do_eval:
            test_dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=test_file,
                                    max_length=args.max_seq_length,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
    
    if args.k_fold == 1:
        if args.do_train and args.do_eval: 
            if args.pre_train_tasks == 'nsp' or args.pre_train_tasks == 'mlm_nsp':
                logging.info(f'There are {len(train_dataset)} pairs of sequences in the train dataset and {len(test_dataset)} in the evaluation one.')
            else:
                logging.info(f'There are {len(train_dataset)} documents in the train datatset and {len(test_dataset)} in the evaluation one.')
        elif args.do_train:
            if args.pre_train_tasks == 'nsp' or args.pre_train_tasks == 'mlm_nsp':
                logging.info(f'Only the train will be performed. There are {len(train_dataset)} pairs of sequences in the dataset')
            else:
                logging.info(f'Only the train will be performed. There are {len(train_dataset)} documents in the dataset')
        elif args.do_eval:
            if args.pre_train_tasks == 'nsp' or args.pre_train_tasks == 'mlm_nsp':
                logging.info(f'Only the evaluation will be performed. There are {len(test_dataset)} pairs of sequences in the dataset')
            else:
                logging.info(f'Only the evaluation will be performed. There are {len(test_dataset)} documents in the dataset')
    else:
        logging.info(f'K-fold cross-validation training with k={args.k_fold}.')
        logging.info(f'There are about {len(train_dataset[0])} documents/sequences in the train dataset and about {len(test_dataset[0])} in the evaluation one.\n'\
            +'(They can vary a little between split)')
    
    if args.do_train:
        if args.k_fold ==1:
            model_output_path = os.path.join(output_path, 'pre_trained_model')
            loss = train(args,train_dataset,model,model_output_path)
        else:
            loss = 0
            start_time = datetime.now()
            for i,X in enumerate(train_dataset):
                model_output_path = os.path.join(output_path, f'model_{i}', 'pre_trained_model')
                loss += train(args, X, models[i], model_output_path)
            loss = loss / len(train_dataset)
        logging.info(f'Average loss = {loss}')
        logging.info(f"{len(train_dataset)} models trained with cross-validation, in total in {str(datetime.now() - start_time)[:-7]}")
    
    if args.do_eval:
        if args.k_fold == 1:
            eval(args, test_dataset, model, mask_token_id=tokenizer.mask_token_id)
        else:
            accs = {}
            for i,Y in enumerate(test_dataset):
                logging.info(f'Model n. {i}:')
                result = eval(args, Y, models[i], mask_token_id=tokenizer.mask_token_id)
                if 'mlm_acc' in result:
                    accs['mlm_acc'] = accs.setdefault('mlm_acc', 0) + result['mlm_acc']
                if 'nsp_acc' in result:
                    accs['nsp_acc'] = accs.setdefault('nsp_acc', 0) + result['nsp_acc']
            if 'mlm_acc' in accs:
                logging.info(f'Average mlm_acc = {accs["mlm_acc"] / len(test_dataset)}')
            if 'nsp_acc' in accs:
                logging.info(f'Average nsp_acc = {accs["nsp_acc"] / len(test_dataset)}')
    
if __name__ == '__main__':
    main()