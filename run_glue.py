from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

import os
from datetime import datetime
import torch
from tqdm import tqdm

from custom_parser import parse_arguments
from load_dataset import FinetuningDataset
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
            
            inputs_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs_ids=inputs_ids,
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

def eval(args, test_dataset, model, output_folder):
    return

def main():
    args = parse_arguments()
    
    current_directory = os.path.dirname(os.path.abspath(__file__))
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
    output_path = os.path.join(current_directory, 'logs',current_time)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print(f'Output files will be saved in folder: {output_path}')
    
    model_path = os.path.join(output_path, 'finetuned_model')
    
    tokenzier_folder = os.path.join(args.model_input, 'tokenizer')
    tokenizer = BertTokenizerFast.from_pretrained(tokenzier_folder)
    
    model_folder = os.path.join(args.model_input, 'model')
    model = BertForSequenceClassification.from_pretrained(model_folder)
    
    dataset = FinetuningDataset(tokenizer, 
                                file_path=args.input_file, 
                                max_length=args.max_seq_length)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [.8,.2])
    
    loss = train(args, train_dataset, model, model_path)
    print(f'Average loss = {loss}')
    
    
if __name__ == '__main__':
    main()