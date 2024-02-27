# Pre-training language model on Electronic Health Records

This repository contains the code to run series of different pre-train and finetuning task on BERT model for EHR data. We explore the type of embedding that BERT would be able to produce out of this different type of data and we proposed a finetune task that aims to predict urgent hospitalisation. 

## Dataset
Due to etical reason the dataset contained in this repository is a sintetic one. You can find the input csv file in the `data` folder

### Create pre-train and finetuning dataset
Run this to create the pre-train text file: 
```shell
!python preprocessing_python/text_generator.py \
        --file_path data/base_red3.csv \
        --output_folder /data \
        --create_pretrain_text_file
```

Run this to create the finetuning text file:
```shell
!python preprocessing_python/text_generator.py \
        --file_path data/base_red3.csv \
        --output_folder /data \
        --output_name finetuning_dataset.txt \
        --create_finetuning_text_data
```

## Pre-train
Run this to run the pre-train:
```shell
!python run_pre_train.py \
      --do_eval \
      --do_train \
      --pre_train_tasks mlm_nsp \
      --input_file /data/dataset_text.txt 
```

You can specify the pre-train task with the command `--pre_train_tasks` or with `--bert_class` passing the name of the BERT class model you want to test (the available classes are [BertForPreTraining](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining), [BertForMaskedLM](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM), and [BertForNextSentencePrediction](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForNextSentencePrediction)).

At the end of pre-train you should see output like this:
```
Nsp acc: 0.9403330249768732
Mlm acc: 0.48940998487140697
```

## Finetunig
Run this to run the finetuning:
```shell
!python run_glue.py \
      --do_train \
      --do_eval \
      --input_file /data/finetuning_dataset.txt 
```
In this case is expected to have an already pre-trained BERT checkpoint to load the weights from. In case you want to run the finetuning on bert-base-uncased you can add this command `--use_pretrained_bert`. 

At the end of finetuning you should see output like this:
```
result = {'acc': 0.8662891986062717, 'f1': 0.9002923026956805, 'acc_and_f1': 0.8832907506509762}
```

## Hyperparameters
You can specify different parameters, like `--train_batch_size`, and if you want a list of hyperparameters you can run:

```shell
!python run_glue.py -h
```

You can find some examples in the `examples.ipynb` notebook.
