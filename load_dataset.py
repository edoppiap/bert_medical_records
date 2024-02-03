from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer

import os, random
import torch

class PreTrainingDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer,  mlm, nsp, file_path='dataset.txt', max_length=512):
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
    self.inputs = self.create_inputs(file_path, max_length, mlm, nsp)

  def create_inputs(self, file_path, max_length, mlm, nsp):
    
    if nsp:
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
                label.append(0)
              else:
                sentence_b.append(self.bag[random.randint(0, self.bag_size-1)])
                label.append(1)

      inputs = self.tokenizer(sentence_a, sentence_b, return_tensors='pt',
                    max_length=max_length, truncation=True, padding='max_length')
      inputs['next_sentence_label'] = torch.LongTensor([label]).T
    else:
      documents = []
      
      with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
          if line.startswith('[CLS]'):
            documents.append(line)
      
      inputs = self.tokenizer(documents, return_tensors='pt',
                              max_length=max_length, truncation=True, padding='max_length',
                              add_special_tokens=False) # special tokens already present in the dataset

    inputs['labels'] = inputs.input_ids.detach().clone()
    vocab_ids = list(self.tokenizer.vocab.values())
    rand = torch.rand(inputs.input_ids.shape)
    # print(f'{rand = }')

    # mask 15% of token randomly
    # don't mask the CLS token (0)
    # don't mask the padding token (102)
    # don't mask the SEP token (1)
    mask_arr = (rand < mlm) * (inputs.input_ids != self.tokenizer.convert_tokens_to_ids('[CLS]')) * (inputs.input_ids != self.tokenizer.convert_tokens_to_ids('[SEP]')) * (inputs.input_ids != self.tokenizer.convert_tokens_to_ids('[PAD]'))

    for i in range(inputs.input_ids.shape[0]):

      selection = torch.flatten(mask_arr[i].nonzero()).tolist()
      # inputs.input_ids[i, selection] = self.tokenizer.convert_tokens_to_ids('[MASK]')

      num_tokens = len(selection)
      num_mask = int(.8 * num_tokens)
      num_random_word = int(.1 * num_tokens)

      random.shuffle(selection)

      # Replace 80% of tokens with the mask token
      for j in range(num_mask):
          inputs.input_ids[i, selection[j]] = self.tokenizer.convert_tokens_to_ids('[MASK]')  # mask token id

      # Replace 10% of tokens with a random word
      for j in range(num_mask, num_mask + num_random_word):
          inputs.input_ids[i, selection[j]] = vocab_ids[random.randint(0, len(vocab_ids) - 1)]

      # The remaining 10% of tokens are unchanged

    return inputs

  def __len__(self):
    return len(self.inputs.input_ids)

  def __getitem__(self,idx):
    return {key: torch.tensor(val[idx]) for key,val in self.inputs.items()}

def get_loader(tokenizer: BertTokenizer, file_path='dataset.txt', max_length=512, batch_size=16, mlm=.15, nsp=True):
    dataset = PreTrainingDataset(tokenizer, mlm, nsp, file_path, max_length)
    loader = loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
    useful for using the texts to train the tokenizer
    (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
        for t in dataset["text"]:
            print(t, file=f)

def dataset_loader(input_file, train_file_name, test_file_name, eval_file_name, output_path):
    #file_path = os.path.join(input_file, text_generated_name)
    dataset = get_loader('text', data_files=input_file, split='train')
    
    d_temp = dataset.train_test_split(test_size=.15, shuffle=True)
    d_temp_2 = d_temp['test'].train_test_split(test_size=10, shuffle=True)

    d = DatasetDict({
        'train': d_temp['train'],
        'test': d_temp_2['train'],
        'eval': d_temp_2['test']})
    
    train_path = os.path.join(output_path,train_file_name)
    test_path = os.path.join(output_path,test_file_name)
    eval_path = os.path.join(output_path, eval_file_name)
                    
    # save the training set to train.txt
    dataset_to_text(d["train"], train_path)
    # save the testing set to test.txt
    dataset_to_text(d["test"], test_path)
    # save the eval set to eval.txt
    dataset_to_text(d["eval"], eval_path)
    
    return d, [train_path]