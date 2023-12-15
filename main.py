import os

import custom_parser
from preprocessing_python.text_generator import create_text_from_data
from load_dataset import dataset_loader
from encoder import encode
from tokenizer import define_tokenizer
from modeling import get_bert_model
from collator import define_collator
from pre_train import pre_train

from datetime import datetime

args = custom_parser.parse_arguments()

current_directory = os.path.dirname(os.path.abspath(__file__))
current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
output_path = os.path.join(current_directory, 'logs',current_time)
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
print(f'Output files will be saved in folder: {output_path}')

train_file, test_file, eval_file = 'train.txt', 'test.txt', 'eval.txt'

if args.input_file is None:
    data_folder = 'bert_medical_records/data'
    file_name = 'base_red3.csv'
    text_generated_name = 'output.txt'
    create_text_from_data(data_folder, file_name, text_generated_name)

    d, files = dataset_loader(data_folder, text_generated_name,
                        train_file_name=train_file, 
                        test_file_name=test_file,
                        eval_file_name=eval_file,
                        output_path=output_path)
else:
    d, files = dataset_loader(args.input_file, 
                        train_file_name=train_file, 
                        test_file_name=test_file,
                        eval_file_name=eval_file,
                        output_path=output_path)


special_tokens = ['[CLS]','[SEP]','[MASK]']

#files = [train_file]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
#vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size)
#max_length = 512
# whether to truncate
truncate_longer_samples = True    # TODO: ADD PARSER ARGUMENT (has to be true for now)

tokenizer = define_tokenizer(special_tokens=special_tokens,
                             tokenizer_name=args.tokenizer_name,
                             files=files, 
                             vocab_size=args.vocab_size, 
                             max_length=args.max_seq_length,
                             output_path=output_path)

train_dataset, test_dataset = encode(d, tokenizer,
                                     max_length=args.max_seq_length,
                                     truncate_longer_samples=truncate_longer_samples)

model = get_bert_model(args.bert_class, args.vocab_size, args.max_seq_length)

data_collator = define_collator(args.pre_train_tasks, tokenizer)

pre_train(model=model,
          data_collator=data_collator,
          train_dataset=train_dataset,
          test_dataset=test_dataset,
          output_path=output_path)