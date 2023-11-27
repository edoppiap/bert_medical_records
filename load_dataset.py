from datasets import load_dataset

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

def dataset_to_text(file_path, train_file_name, test_file_name):
    #file_path = os.path.join(data_folder, 'output.txt')
    dataset = dataset_to_text('text', data_files='data/output.txt', split='train')
    d = dataset.train_test_split(test_size=.1)
    
    train_file, test_file = 'train.txt', 'test.txt'
                    
    # save the training set to train.txt
    dataset_to_text(d["train"], train_file)
    # save the testing set to test.txt
    dataset_to_text(d["test"], test_file)
    
    return d