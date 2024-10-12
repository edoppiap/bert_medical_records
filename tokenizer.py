import os
import json
import logging

import argparse
from datetime import datetime
from tqdm import tqdm

from tokenizers import Tokenizer
from tokenizers import BertWordPieceTokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from transformers import BertTokenizerFast, RetriBertTokenizer, RetriBertTokenizerFast
from transformers import BertTokenizer

def get_tokenizer_from_string(tokenizer_name: str, model_path, vocab_size):
  tokenizer = None
  if tokenizer_name == 'BertTokenizerFast':
    tokenizer = BertTokenizerFast.from_pretrained(model_path, vocab_size=vocab_size)
  # this class can be extended on further explorations
    
  if tokenizer == None:
    raise ValueError(f'Invalid tokenizer name {tokenizer_name}')
  
  return tokenizer

def get_tokenizer_from_path(path):
  # tokenizer_folder = os.path.join(path, 'tokenizer')
  return BertTokenizerFast.from_pretrained(path)

def get_tokenizer(args, output_path, path=None):
  special_tokens = ['[CLS]','[SEP]','[MASK]','[UNK]','[PAD]']
  
  if args.use_pretrained_bert:
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  
  elif path is not None:
    if os.path.exists(path):
      logging.info(f'Loading the custom tokenizer from {path =}')
      tokenizer = BertTokenizerFast.from_pretrained(path)
    else:
      logging.info(f'Tokenizer path provided invalid, loadind an already pretrained version from Huggingface')
      tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  else:
    logging.info(f'No tokenizer provided, training one from skratch on the dataset')
    if os.path.isdir(args.input_file):
      train_path = os.path.join(args.input_file, 'train.txt')
      test_path = os.path.join(args.input_file, 'test.txt')
      assert os.path.isfile(train_path) and os.path.isfile(test_path), 'You passed a folder that do not contain any train.txt and test.txt files'
      files = [train_path, test_path]
    else:
      files = [args.input_file]
    
    tokenizer = train_tokenizer(special_tokens=special_tokens,
                                    tokenizer_name=args.tokenizer_name,
                                    files=files, 
                                    vocab_size=args.vocab_size, 
                                    max_length=args.max_seq_length,
                                    output_path=output_path)
  return tokenizer

def train_tokenizer(tokenizer_name, special_tokens, files, vocab_size, max_length, output_path):  
  tokenizer_path = os.path.join(output_path,'tokenizer')
  
  if os.path.exists(os.path.join(tokenizer_path, 'config.json')):
    logging.info('Tokenizer found')
    
  else:
    logging.info('Train the tokenizer')

    # initialize the WordPiece tokenizer
    # CLASS THAT CAN BE CHOSED FROM THE UI
    tokenizer = BertWordPieceTokenizer()
    # train the tokenizer
    tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
    
    # enable truncation up to the maximum 512 tokens
    tokenizer.enable_truncation(max_length=max_length)
    
    # make the directory if not already there
    if not os.path.isdir(tokenizer_path):
        os.makedirs(tokenizer_path)
    # save the tokenizer 
    tokenizer.save_model(tokenizer_path)
    
    # dumping some of the tokenizer config to config file, 
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(tokenizer_path, "config.json"), "w") as f:
        tokenizer_cfg = {
          "do_lower_case": False, # False seems better
          "unk_token": "[UNK]",
          "sep_token": "[SEP]",
          "pad_token": "[PAD]",
          "cls_token": "[CLS]",
          "mask_token": "[MASK]",
          "model_max_length": max_length,
          "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)
    
    logging.info(f'Custom tokenzier trained and saved in {tokenizer_path}')
    
  tokenizer = get_tokenizer_from_string(tokenizer_name, tokenizer_path, vocab_size)
  
  return tokenizer

if __name__ == '__main__':
  
  """ Run this file to train and test how a custom tokenizer truncates the sequences in a dataset with a specific max_seq_length.  
  """
  
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  parser.add_argument('--input_file', type=str, 
                      help='File/folder in which are located the input csv files to test the tokenizer')
  parser.add_argument('--output_dir', type=str,
                        help='Folder where to save the output text file')
  parser.add_argument('--max_seq_length', type=int, default=512, choices=[128, 512],
                        help='The maximum total input sequence length after WordPiece tokenization. '+\
                            'Set this parameter with the same value of the used during training to know how many sequences will be truncated')
  parser.add_argument('--use_pretrained_bert', action='store_true',
                        help='This will initialize the bert model as already pre-trained')
  parser.add_argument('--vocab_size', type=int, default=30_522, help=' ')
  parser.add_argument('--tokenizer_name', type=str, default='BertTokenizerFast',
                        choices=['BertTokenizerFast'],
                        help='Tokenizer used during pre-train')
  
  args = parser.parse_args()
  
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
  
  tokenizer = get_tokenizer(args=args,
                            output_path=output_path)
  
  n_lines = 0
  n_truncated_lines = 0
  truncated_sequences = []
  with open(args.input_file, 'r') as f:
    for line in tqdm(f, desc='Reading input file'):
      n_lines += 1
      if '<end>' in line:
        sequence = line.split('<end>')[0]
        if len(sequence.split('SEP')) == 2:
          sentence_a, sentence_b = sequence.split('[SEP]')
          inputs = tokenizer(sentence_a,sentence_b, return_tensors='pt')
        else:
          inputs = tokenizer(sequence, return_tensors='pt')
        if len(inputs['input_ids'][0]) > args.max_seq_length:
          truncated_sequences.append(sequence)
          n_truncated_lines += 1
      else:
        inputs = tokenizer(line, return_tensors='pt')
        if len(inputs['input_ids'][0]) > args.max_seq_length:
          truncated_sequences.append(line)
          n_truncated_lines += 1
  print(f'Tokenizer will truncate {n_truncated_lines}/{n_lines} sequences ({n_truncated_lines/n_lines*100:.2f}%) with max_seq_length = {args.max_seq_length} on this dataset')
  if len(truncated_sequences) > 0:
    print(f'Example of truncated sequence:\nFull sequence: {truncated_sequences[0]}\nTokenized sequence: {tokenizer.tokenize((truncated_sequences[0]).split("<end>")[0], max_length=512, truncation=True, padding="max_length" )}')