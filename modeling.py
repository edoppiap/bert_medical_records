'''
Setting up BERT models with specific configurations and classes, tailored for different tasks like (MLM), (NSP) or (MLM and NSP).
'''
from transformers import BertConfig, BertForMaskedLM, BertForPreTraining, BertForNextSentencePrediction
import os
import logging

bert_classes = {
    'BertForMaskedLM' : BertForMaskedLM,
    'BertForNextSentencePrediction': BertForNextSentencePrediction,
    'BertForPreTraining' : BertForPreTraining
}

def get_model(bert_class_name, config, pretrained=False, input_path=None):
    
    if bert_class_name not in bert_classes:
        raise ValueError(f'Invalid bert class name {bert_class_name}')        
    
    if pretrained:
        model = bert_classes[bert_class_name].from_pretrained('bert-base-uncased', config=config)
        logging.info('Loaded an already pretrained version of BERT from server')
    elif input_path is not None:
        assert os.path.exists(input_path), 'Invalid model input path provided'
        
        model = bert_classes[bert_class_name].from_pretrained(input_path)
        logging.info('Loaded an already pretrained version of BERT from folder path')
    else:
        model = bert_classes[bert_class_name](config)
        logging.info(f'Loaded the raw architecture of {type(model)} (has to be pretrained!)')
    
    return model

def get_model_from_path(bert_class:str, path):
    
    if bert_class == 'BertForMaskedLM':
        model = BertForMaskedLM.from_pretrained(path)
    elif bert_class == 'BertForPreTraining':
        model = BertForPreTraining.from_pretrained(path)
    elif bert_class == 'BertForNextSentencePrediction':
        model = BertForNextSentencePrediction.from_pretrained(path)
    else:
        raise ValueError(f'invalid bert class name {bert_class}')
    
    return model

def get_bert_model(bert_class_name, args, pad_token_id, input_path=None):
    
    model_config = None
    
    if args.bert_config_file is not None:
        assert os.path.isfile(args.bert_config_file), 'You should pass a json file to load the desired Bert Config'
        model_config = BertConfig.from_json_file(
            args.bert_config_file
        )
    elif input_path is None:    
        model_config = BertConfig(vocab_size=args.vocab_size,
                                max_position_embeddings=args.max_seq_length,
                                hidden_size=args.hidden_size,\
                                num_hidden_layers=args.num_hidden_layers,
                                num_attention_heads=args.num_attention_heads,
                                intermediate_size=args.intermediate_size,
                                hidden_act=args.hidden_act,
                                hidden_dropout_prob=args.hidden_dropout_prob,
                                attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                                initializer_range=args.initializer_range,
                                layer_norm_eps=args.layer_norm_eps,
                                pad_token_id=pad_token_id,
                                gradient_checkpointing=False)
        
    model = get_model(bert_class_name = bert_class_name, 
                                  config = model_config,
                                  pretrained = args.use_pretrained_bert,
                                  input_path = input_path)

    # CLASS THAT CAN BE CHOSED FROM THE UI
    #model = BertForMaskedLM(config=model_config) # this one implements only the MLM task
    # in reality I should use the class transformers.BertForPreTraining(config)
    # that implements both MLM and NSP tasks

    # they use a custom implementatino of Bert, define inside modeling (I think I will use that)
    
    return model