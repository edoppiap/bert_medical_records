# Pre-training language model on Electronic Health Records

This repository contains the code to pre-training and finetuning the [BERT](https://huggingface.co/docs/transformers/model_doc/bert) model on EHR data. 
<!-- It is associated with the paper: **PHeP: TrustAlert Open-Source Platform for Enhancing Predictive Healthcare with Deep Learning** -->

## Generating textual datasets

The EHR data we used to build the BERT model is composed of a sequence of [ICD-9](https://archive.cdc.gov/www_cdc_gov/nchs/icd/icd9cm.htm#:~:text=ICD-9-CM%20is%20the,10%20for%20mortality%20coding%20started.) codes from hospitalisation and medication events. Each event consists of several ICD-9 codes that may represent diseases diagnosed or medicines prescribed in that event. All codes related to the same event are concatenated to form a sequence that the model can read as a sentence. \
For ethical reasons, the dataset used to train the models can't be shared. 

<!-- Due to etical reason the dataset contained in this repository is a sintetic one. You can find the input csv file in the `data` folder -->

### Create pre-train and finetuning dataset

Run to create the pre-train text file: 
```shell
!python preprocessing_python/text_generator.py \
        --file_path {CSV_FILE_PATH} \
        --output_folder {OUTPUT_PATH} \
        --create_pretrain
```

Run to create the finetuning text file:
```shell
!python preprocessing_python/text_generator.py \
        --file_path {CSV_FILE_PATH} \
        --output_folder {OUTPUT_PATH} \
        --create_finetuning
```

You can choose to do an 80/20 test split and store the results in two separate files with the `--split` argument. The dataset for the mlm task can be different as there is no need to create sentence pairs, so the argument `--mlm_only` should be added when creating the pretrain text file.

## Pre-train
Run to perform pre-training:
```shell
!python run_pre_train.py \
      --do_eval \
      --do_train \
      --pre_train_tasks mlm_nsp \
      --input_file {PRETRAINING_DATASET_PATH}
```

You can specify the pre-train task with the command `--pre_train_tasks`. For all possible combinations of arguments, run `--h`. If you have splitted the dataset in the previous step, simply pass the folder in which the two files are stored to the `--input_file` argument. 

K_fold cross-validation can be performed to train k instances of the model. To do this, you can specify the number of folds to use with `--k_fold {k}`. This requires significant resources. 

At the end of pre-train you should see output like this:
```
Nsp acc: 0.9403330249768732
Mlm acc: 0.48940998487140697
```

## Finetunig
Run to perform finetuning:
```shell
!python run_glue.py \
      --do_train \
      --do_eval \
      --model_input {PRETRAINED_MODEL_PATH}
      --input_file {FINETUNING_DATASET_PATH}
```
In this case is expected to have an already pre-trained BERT model to load the weights from. In case you want to run the finetuning on bert-base-uncased you can add this command `--use_pretrained_bert`.
 

<!-- At the end of finetuning you should see output like this:
```
result = {'acc': 0.8662891986062717, 'f1': 0.9002923026956805, 'acc_and_f1': 0.8832907506509762}
``` -->

## Hyperparameters
You can specify different parameters, like `--train_batch_size`, and if you want a list of hyperparameters you can run:

```shell
!python run_glue.py -h
```

<!-- You can find some examples in the `examples.ipynb` notebook. -->
