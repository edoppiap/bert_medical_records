import os
import json
import logging

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
  special_tokens = ['[CLS]','[SEP]','[MASK]']
  
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