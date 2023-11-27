from transformers import BertConfig, BertForMaskedLM

def define_model(vocab_size, max_length):
    # CLASS THAT CAN BE CHOSED FROM THE UI
    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)

    # CLASS THAT CAN BE CHOSED FROM THE UI
    model = BertForMaskedLM(config=model_config) # this one implements only the MLM task
    # in reality I should use the class transformers.BertForPreTraining(config)
    # that implements both MLM and NSP tasks

    # they use a custom implementatino of Bert, define inside modeling (I think I will use that)
    
    return model