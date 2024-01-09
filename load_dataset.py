from datasets import load_dataset, DatasetDict

import os

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
    dataset = load_dataset('text', data_files=input_file, split='train')
    
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