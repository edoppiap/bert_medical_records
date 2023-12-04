from transformers import TrainingArguments, Trainer

def pre_train(model, data_collator, train_dataset, test_dataset, output_path):
    training_args = TrainingArguments(
        output_dir=output_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=10,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=8,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=64,  # evaluation batch size
        logging_steps=1000,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=1000,
        load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        report_to='none',
        # save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    
    # initialize the trainer and pass everything to it
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )