import os

import custom_parser

from preprocessing_python.gen_drugs_input import create_text_from_data
from load_dataset import dataset_to_text_2
from encoder import encode
from tokenizer import define_tokenizer
from modeling import define_model
from collator import define_collator
from pre_train import pre_train

args = custom_parser.parse_arguments()

train_file, test_file = 'train.txt', 'test.txt'

if args.input_file is None:
    data_folder = 'bert_medical_records/data'
    file_name = 'base_red3.csv'
    output_name = 'output.txt'
    create_text_from_data(data_folder, file_name, output_name)

    d, files = dataset_to_text_2(data_folder, output_name,
                        train_file_name=train_file, 
                        test_file_name=test_file)
else:
    d, files = dataset_to_text_2(args.input_file, 
                        train_file_name=train_file, 
                        test_file_name=test_file)


special_tokens = ['[CLS]','[SEP]','[MASK]']

#files = [train_file]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
#vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
#max_length = 512
# whether to truncate
truncate_longer_samples = True    # TODO: ADD PARSER ARGUMENT (has to be true)

tokenizer = define_tokenizer(special_tokens, files, args.vocab_size, args.max_seq_length)
train_dataset, test_dataset = encode(d, tokenizer,
                                     max_length=args.max_seq_length,
                                     truncate_longer_samples=truncate_longer_samples)

model = define_model(args.vocab_size, args.max_seq_length)

data_collator = define_collator(tokenizer)

pre_train(model=model,
          data_collator=data_folder,
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          output_path='model/pretrain_bert')