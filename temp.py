from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForPreTraining
from torch.optim import AdamW

import os
import json
import random
import torch
from tqdm import tqdm

#------------------------------------------------------------------------------------------------------#
# DEFINE DATASET CLASS
#
#
class PreTrainingDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, file_path='dataset.txt', block_size=512):
    assert os.path.isfile(file_path)
    directory, filename = os.path.split(file_path)
    self.tokenizer = tokenizer
    self.bag = []
    with open(file_path, 'r', encoding='utf-8') as file:
      for line in file:
        if line.startswith('[CLS]'):
          sentences = [s for s in line.lstrip('[CLS]').split('[SEP]') if s.strip() != '']
          self.bag.extend(sentences)
    self.bag_size = len(self.bag)
    self.inputs = self.create_inputs(file_path)

  def create_inputs(self, file_path):
    sentence_a = []
    sentence_b = []
    label = []

    with open(file_path, 'r', encoding='utf-8') as file:
      for line in file:
        if line.startswith('[CLS]'):
          sentences = [s for s in line.lstrip('[CLS]').split('[SEP]') if s.strip() != '']
          num_sentences = len(sentences)
          if num_sentences > 1:
            start = random.randint(0, (num_sentences-2))
            sentence_a.append(sentences[start])
            if random.random() > .5:
              sentence_b.append(sentences[start+1])
              label.append(1)
            else:
              sentence_b.append(self.bag[random.randint(0, self.bag_size-1)])
              label.append(0)
    
    inputs = self.tokenizer(sentence_a, sentence_b, return_tensors='pt',
                   max_length=max_length, truncation=True, padding='max_length')
    inputs['next_sentence_label'] = torch.LongTensor([label]).T

    inputs['labels'] = inputs.input_ids.detach().clone()
    vocab_ids = list(self.tokenizer.vocab.values())
    rand = torch.rand(inputs.input_ids.shape)

    # mask 15% of token randomly
    # don't mask the CLS token (101)
    # don't mask the padding token (102)
    # don't mask the SEP token (0)
    mask_arr = (rand < .15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    for i in range(inputs.input_ids.shape[0]):
      
      selection = torch.flatten(mask_arr[i].nonzero()).tolist()
      
      num_tokens = len(selection)
      num_mask = int(.8 * num_tokens)
      num_random_word = int(.1 * num_tokens)

      random.shuffle(selection)

      # Replace 80% of tokens with the mask token
      for j in range(num_mask):
          inputs.input_ids[i, selection[j]] = 103  # mask token id

      # Replace 10% of tokens with a random word
      for j in range(num_mask, num_mask + num_random_word):
          inputs.input_ids[i, selection[j]] = vocab_ids[random.randint(0, len(vocab_ids) - 1)]

      # The remaining 10% of tokens are unchanged

      # 80% of the time replace with the mask token
      # if rand_value < .8:
      #   inputs.input_ids[i, selection] = 103 # this is the mask token id
      # # 10% of the time replace with a random word
      # elif rand_value > .8 and rand_value < .9:
      #   inputs.input_ids[i, selection] = vocab_ids[random.randint(0, len(vocab_ids)-1)]
      # the other 10% of the time do not replace the chosen word
    return inputs

  def __len__(self):
    return len(self.inputs.input_ids) 

  def __getitem__(self,idx):
    return {key: torch.tensor(val[idx]) for key,val in self.inputs.items()}

#------------------------------------------------------------------------------------------------------#
# TRAIN THE TOKENIZER
#
#
special_tokens = ['[CLS]','[SEP]','[MASK]']

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
max_length = 512
# whether to truncate
truncate_longer_samples = True

# initialize the WordPiece tokenizer
# CLASS THAT CAN BE CHOSED FROM THE UI
tokenizer = BertWordPieceTokenizer()
# train the tokenizer
tokenizer.train(files=['dataset.txt'], vocab_size=vocab_size, special_tokens=special_tokens)
# enable truncation up to the maximum 512 tokens
tokenizer.enable_truncation(max_length=max_length)

model_path = 'temp-bert'
# make the directory if not already there
if not os.path.isdir(model_path):
    os.mkdir(model_path)
    
# save the tokenizer
tokenizer.save_model(model_path)
# dumping some of the tokenizer config to config file,
# including special tokens, whether to lower case and the maximum sequence length
with open(os.path.join(model_path, "config.json"), "w") as f:
    tokenizer_cfg = {
      "do_lower_case": False,
      "unk_token": "[UNK]",
      "sep_token": "[SEP]",
      "pad_token": "[PAD]",
      "cls_token": "[CLS]",
      "mask_token": "[MASK]",
      "model_max_length": max_length,
      "max_len": max_length,
    }
    json.dump(tokenizer_cfg, f)

#------------------------------------------------------------------------------------------------------#
# LOAD THE PRETRAINED TOKENIZER
#
#
tokenizer = BertTokenizerFast.from_pretrained(model_path, vocab_size=vocab_size)

#------------------------------------------------------------------------------------------------------#
# INSTANTIATING THE DATALOADER
#
#
dataset = PreTrainingDataset(tokenizer, file_path='/content/dataset.txt')
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

#------------------------------------------------------------------------------------------------------#
# INSTANTIATING THE MODEL
#
#
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

model_config = BertConfig(vocab_size=vocab_size, 
                          max_position_embeddings=max_length,
                          hidden_size=768,
                          num_hidden_layers=12,
                          num_attention_heads=12,
                          intermediate_size=3072, hidden_act='gelu',
                          hidden_dropout_prob=0.1,
                          attention_probs_dropout_prob=0.1,
                          type_vocab_size=1,
                          initializer_range=0.02,
                          layer_norm_eps=1e-12,
                          pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
                          gradient_checkpointing=False,)

model = BertForPreTraining(config=model_config)
model.to(device)
model.train()

optim = AdamW(model.parameters(), lr=5e-5)

#------------------------------------------------------------------------------------------------------#
# TRAIN LOOP
#
#
for epoch in range(2):
  loop = tqdm(loader, leave=True)
  for batch in loop:
    optim.zero_grad()

    input_ids = batch['input_ids'].to(device)
    token_type_ids = batch['token_type_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    next_sentence_label = batch['next_sentence_label'].to(device)
    labels = batch['labels'].to(device)

    print(f'\n{input_ids.shape = }')
    print(f'{token_type_ids.shape = }')
    print(f'{attention_mask.shape = }')
    print(f'{next_sentence_label.shape = }')
    print(f'{labels.shape =}')

    outputs = model(input_ids=input_ids,
                    token_type_ids = token_type_ids,
                    attention_mask = attention_mask,
                    next_sentence_label = next_sentence_label,
                    labels = labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

    loop.set_description(f'Epoch {epoch}')
    loop.set_postfix(loss=loss.item())