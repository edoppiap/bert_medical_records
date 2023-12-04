import os
import json

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

def define_tokenizer(special_tokens, files, vocab_size, max_length):  
  model_path = 'model/pretrained_bert'
  
  if os.path.exists(os.path.join(model_path, 'config.json')):
    tokenizer = BertTokenizerFast.from_pretrained(model_path, vocab_size=vocab_size)
    
  else:

    # initialize the WordPiece tokenizer
    # CLASS THAT CAN BE CHOSED FROM THE UI
    tokenizer = BertWordPieceTokenizer()
    # train the tokenizer
    tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
    # enable truncation up to the maximum 512 tokens
    tokenizer.enable_truncation(max_length=max_length)

    
    # make the directory if not already there
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    # save the tokenizer  
    tokenizer.save_model(model_path)
    # dumping some of the tokenizer config to config file, 
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(model_path, "config.json"), "w") as f:
        tokenizer_cfg = {
          "do_lower_case": True,
          "unk_token": "[UNK]",
          "sep_token": "[SEP]",
          "pad_token": "[PAD]",
          "cls_token": "[CLS]",
          "mask_token": "[MASK]",
          "model_max_length": max_length,
          "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)
        
    tokenizer = BertTokenizerFast.from_pretrained(model_path, vocab_size=vocab_size)
  
  return tokenizer