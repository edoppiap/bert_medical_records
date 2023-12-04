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

def dataset_to_text_2(data_folder, ouput_name, train_file_name, test_file_name):
    file_path = os.path.join(data_folder, ouput_name)
    dataset = load_dataset('text', data_files=file_path, split='train')
    d = dataset.train_test_split(test_size=.1)
    
    train_path = os.path.join(data_folder,train_file_name)
    test_path = os.path.join(data_folder,test_file_name)
                    
    # save the training set to train.txt
    dataset_to_text(d["train"], train_path)
    # save the testing set to test.txt
    dataset_to_text(d["test"], test_path)
    
    return d, [train_path]