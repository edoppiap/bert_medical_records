'''
Setting up BERT models with specific configurations and classes, tailored for different tasks like (MLM), (NSP) or (MLM and NSP).
'''
from transformers import BertConfig, BertForMaskedLM, BertForPreTraining, BertForNextSentencePrediction
import os

def get_model_from_string(bert_class_name, config):
    model = None
    if bert_class_name == 'BertForMaskedLM':
        model = BertForMaskedLM(config)
    elif bert_class_name == 'BertForPreTraining':
        model = BertForPreTraining(config)
    elif bert_class_name == 'BertForNextSentencePrediction':
        model = BertForNextSentencePrediction(config)
        
    if model is None:
        raise ValueError(f'Invalid bert class name {bert_class_name}')
    
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

def get_bert_model(bert_class_name, vocab_size, max_length, pad_token_id):
    # CLASS THAT CAN BE CHOSED FROM THE UI
    model_config = BertConfig(vocab_size=vocab_size,
                                max_position_embeddings=max_length,
                                hidden_size=768,\
                                num_hidden_layers=12,
                                num_attention_heads=12,
                                intermediate_size=3072,
                                hidden_act='gelu',
                                hidden_dropout_prob=0.1,
                                attention_probs_dropout_prob=0.1,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                pad_token_id=pad_token_id,
                                gradient_checkpointing=False,)
    
    model = get_model_from_string(bert_class_name, model_config)

    # CLASS THAT CAN BE CHOSED FROM THE UI
    #model = BertForMaskedLM(config=model_config) # this one implements only the MLM task
    # in reality I should use the class transformers.BertForPreTraining(config)
    # that implements both MLM and NSP tasks

    # they use a custom implementatino of Bert, define inside modeling (I think I will use that)
    
    return model