'''
class PreTrainingDataset:  Loads and processes text data from a file.
It extracts sentences from lines starting with '[CLS]' and splits them at '[SEP]'.
Generates input data for the model. It supports Masked Language Modeling and Next Sentence Prediction pre-training tasks.

function get_loader:Creates and returns a DataLoader for the pre-training dataset.
'''
# from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer

import os, random
import torch

class FinetuningDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer, file_path='finetune_dataset.txt', max_length=512):
    assert os.path.isfile(file_path)
    self.tokenizer = tokenizer
    with open(file_path, 'r', encoding='utf-8') as file:
      docs = []
      labels = []
      for line in file:
        if line.startswith('[CLS]'):
          doc = line.split('<end>')[0]
          label = line.split('<end>')[1].replace('\n', '')
          docs.append(doc)
          labels.append(int(label))
    # self.docs = docs
    # self.labels = labels
    self.inputs = self.create_inputs(docs, labels, max_length)
    
  def create_inputs(self, docs, labels, max_length):
    inputs = self.tokenizer(docs, return_tensors='pt',
                  max_length=max_length, truncation=True, padding='max_length',
                  add_special_tokens=False) # special tokens already present in the dataset
    inputs['labels'] = torch.LongTensor(labels).T
    
    return inputs
    
  def __len__(self):
    return len(self.inputs.input_ids)

  def __getitem__(self,idx):
    return {key: torch.tensor(val[idx]) for key,val in self.inputs.items()}
          

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
    # When nsp is True, the function reads a file and splits the text into pairs of sentences (Sentence A and Sentence B). 
    # It randomly decides whether Sentence B is the actual next sentence in the document (label 0) or a random sentence 
    # (label 1). These sentences are then tokenized.
    if nsp:
      sentence_a = []
      sentence_b = []
      label = []

      with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
          if line.startswith('[CLS]'):
            sentences = [s for s in line.lstrip('[CLS]').split('[SEP]') if s.strip() != '']
            num_sentences = len(sentences)
            start = 0
            while start < (num_sentences - 2):
              # if num_sentences > 1:
              # start = random.randint(0, (num_sentences-2))
              sentence_a.append(sentences[start])
              if random.random() > .5:
                sentence_b.append(sentences[start+1])
                label.append(0)
              else:
                sentence_b.append(self.bag[random.randint(0, self.bag_size-1)])
                label.append(1)
              start += 1

      inputs = self.tokenizer(sentence_a, sentence_b, return_tensors='pt',
                    max_length=max_length, truncation=True, padding='max_length')
      inputs['next_sentence_label'] = torch.LongTensor([label]).T
    else:
      #If nsp is False, the entire document is tokenized as a single sequence without splitting into sentence pairs.
      documents = []
      
      with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
          if line.startswith('[CLS]'):
            documents.append(line)
      #The function tokenizes the inputs using the specified BERT tokenizer. It pads or truncates sequences to max_length.
      inputs = self.tokenizer(documents, return_tensors='pt',
                              max_length=max_length, truncation=True, padding='max_length',
                              add_special_tokens=False) # special tokens already present in the dataset

#For MLM, it duplicates the input IDs to create labels
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

def get_dataset(tokenizer: BertTokenizer, file_path='dataset.txt', max_length=512, mlm=.15, nsp=True):
    dataset = PreTrainingDataset(tokenizer, mlm, nsp, file_path, max_length)
    return torch.utils.data.random_split(dataset, [.8,.2])

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

# def dataset_loader(input_file, train_file_name, test_file_name, eval_file_name, output_path):
#     #file_path = os.path.join(input_file, text_generated_name)
#     dataset = get_dataset('text', data_files=input_file, split='train')
    
#     d_temp = dataset.train_test_split(test_size=.15, shuffle=True)
#     d_temp_2 = d_temp['test'].train_test_split(test_size=10, shuffle=True)

#     d = DatasetDict({
#         'train': d_temp['train'],
#         'test': d_temp_2['train'],
#         'eval': d_temp_2['test']})
    
#     train_path = os.path.join(output_path,train_file_name)
#     test_path = os.path.join(output_path,test_file_name)
#     eval_path = os.path.join(output_path, eval_file_name)
                    
#     # save the training set to train.txt
#     dataset_to_text(d["train"], train_path)
#     # save the testing set to test.txt
#     dataset_to_text(d["test"], test_path)
#     # save the eval set to eval.txt
#     dataset_to_text(d["eval"], eval_path)
    
#     return d, [train_path]