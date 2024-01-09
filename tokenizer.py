import os
import json

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
  elif tokenizer_name == 'RetriBertTokenizer':
    tokenizer = RetriBertTokenizer.from_pretrained(model_path, vocab_size=vocab_size)
    
  if tokenizer == None:
    raise ValueError(f'Invalid tokenizer name {tokenizer_name}')
  
  return tokenizer

def get_tokenizer_from_path(path):
  tokenizer_folder = os.path.join(path, 'tokenizer')
  return BertTokenizerFast.from_pretrained(tokenizer_folder)

def define_tokenizer(tokenizer_name, special_tokens, files, vocab_size, max_length, output_path):  
  tokenizer_path = os.path.join(output_path,'tokenizer')
  
  if os.path.exists(os.path.join(tokenizer_path, 'config.json')):
    print('Tokenizer found')
    
  else:
    print('Train the tokenizer')

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
    
    print('Saving the tokenizer')
    
  print('Loading the workable tokenizer')
  tokenizer = get_tokenizer_from_string(tokenizer_name, tokenizer_path, vocab_size)
  
  return tokenizer