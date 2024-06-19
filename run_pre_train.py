from transformers import BertForMaskedLM, BertConfig, BertForPreTraining, BertForNextSentencePrediction
from transformers import BertTokenizerFast
from transformers.data.metrics import acc_and_f1
# from sklearn.metrics import recall_score

import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from custom_parser import parse_arguments
from modeling import get_bert_model, get_model_from_path
from tokenizer import train_tokenizer, get_tokenizer_from_path
from optimizer import get_optimizer
from load_dataset import PreTrainingDataset, NewPreTrainingDataset

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
    if nsp_preds is not None:
        print(f'Nsp acc: {torch.sum(nsp_preds == nsp_truths).item() / len(nsp_truths)}')
    if mlm_preds is not None:
        print(f'Mlm acc: {torch.sum(mlm_preds == mlm_truths).item() / len(mlm_truths)}')
    # print(f'recall_at_k: {recall_score(mlm_truths.numpy(), top_k_indeces.numpy(), average="samples")}')
    

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
                # print(f'\n{torch.topk(mlm_logits.detach().cpu()[mask,:], 5, dim=1) = }')
                # mlm_pred = torch.argmax(mlm_logits.detach().cpu()[mask,:], dim=1)
                # # print(f'{mlm_pred}')
                # # _, top_k_indices = torch.topk(mlm_pred[mask], 5, dim=1)
                # print(f'{mlm_pred = }\n{labels.detach().cpu()[mask] = }')
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
        # print(f'{top_k_indeces = }\n{mlm_truths}')
    
    # print(f'{nsp_preds.size() = } - {nsp_truths.size() = }')
    # print(f'{mlm_preds.size() = } - {mlm_truths.size() = }')
    
    result = compute_metrics(nsp_preds,nsp_truths,mlm_preds,mlm_truths)
    return result
        
def main():
    args = parse_arguments()
    
    assert (args.do_train and args.do_eval) \
        or (args.model_input and args.do_eval) \
        or (args.do_train and not args.do_eval) \
        or (args.use_pretrained_bert and args.do_eval), \
            '--do_eval present without one between --do_train, --model_input or --use_pretrained_bert. You need to train, pass or select a pretrain model to evaluate'
    
    if args.do_train:
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
        
        tokenizer_output_path = os.path.join(output_path, 'tokenizer')
        model_output_path = os.path.join(output_path, 'pre_trained_model')
    
    if args.pre_train_tasks is not None:
        if args.pre_train_tasks == 'mlm':
            bert_class = 'BertForMaskedLM'
        elif args.pre_train_tasks == 'nsp':
            bert_class = 'BertForNextSentencePrediction'
        elif args.pre_train_tasks == 'mlm_nsp':
            bert_class = 'BertForPreTraining'
    else:
        bert_class = args.bert_class
        
    if args.model_input:
        model = get_model_from_path(bert_class, os.path.join(args.model_input, 'pre_trained_model'))
        tokenizer_path = os.path.join(args.model_input, 'tokenizer')
        if os.path.exists(tokenizer_path):
            tokenizer = get_tokenizer_from_path()
        else:
            tokenizer = tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif args.use_pretrained_bert:
        special_tokens = ['[CLS]','[SEP]','[MASK]']
        
        # custom_tokenizer = train_tokenizer(special_tokens=special_tokens,
        #                             tokenizer_name=args.tokenizer_name,
        #                             files=[args.input_file], 
        #                             vocab_size=args.vocab_size, 
        #                             max_length=args.max_seq_length,
        #                             output_path=output_path)
        # new_tokens = [token for token,_ in sorted(custom_tokenizer.vocab.items(), key=lambda x: x[1])]
        
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
        # num_add_tokens = tokenizer.add_tokens(new_tokens)
        # print(f'Added {num_add_tokens} tokens')
        
        if bert_class == 'BertForMaskedLM':
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        elif bert_class == 'BertForNextSentencePrediction':
            model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        elif bert_class == 'BertForPreTraining':
            model = BertForPreTraining.from_pretrained('bert-base-uncased')
        
        # model.resize_token_embeddings(len(tokenizer))
    elif args.do_train:
        special_tokens = ['[CLS]','[SEP]','[MASK]']

        tokenizer = train_tokenizer(special_tokens=special_tokens,
                                    tokenizer_name=args.tokenizer_name,
                                    files=[args.input_file], 
                                    vocab_size=args.vocab_size, 
                                    max_length=args.max_seq_length,
                                    output_path=output_path)
        
        model = get_bert_model(bert_class, args.vocab_size, args.max_seq_length,
                               pad_token_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
        
    if os.path.isfile(args.input_file):
        dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=args.input_file,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8,.2])
    elif os.path.isdir(args.input_file):
        train_file = os.path.join(args.input_file, 'train.txt')
        test_file = os.path.join(args.input_file, 'test.txt')
        assert os.path.isfile(train_file) and os.path.isfile(test_file), \
            'Folder passed as input file but no train.txt or test.txt file found'
        train_dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=train_file,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
        test_dataset = NewPreTrainingDataset(tokenizer,
                                    file_path=test_file,
                                    mlm=args.mlm_percentage if bert_class == 'BertForMaskedLM' or bert_class == 'BertForPreTraining' else 0)
    
    if args.do_train:
        loss = train(args,train_dataset,model,model_output_path)
        print(f'Average loss = {loss}')
    
    if args.do_eval:
        eval(args, test_dataset, model, mask_token_id=tokenizer.mask_token_id)
    
if __name__ == '__main__':
    main()