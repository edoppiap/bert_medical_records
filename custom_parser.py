import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #-----------------------------------------------------------------#
    # NEW ARGUMENTS
    #
    #
    parser.add_argument('--vocab_size', type=int, default=30_522, help=' ')
    
    #-----------------------------------------------------------------#
    # PRETRAINING ARGUMENTS
    #
    #
    parser.add_argument('--bert_config_file', type=str, default=None,
                        help='The config json file corresponding to the pre-trained BERT model. '+\
                            'This specifies the model architecture.')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input TF example files (can be a glob or comma separated).')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='The output directory where the model checkpoints will be written.')
    parser.add_argument('--init_checkpoint', type=str, default=None, 
                        help='Initial checkpoint (usually from a pre-trained BERT model).')
    parser.add_argument('--max_seq_length', type=int, default=128, choices=[128, 512],
                        help='The maximum total input sequence length after WordPiece tokenization. '+\
                            'Sequences longer than this will be truncated, and sequences shorter '+\
                             'than this will be padded. Must match data generation.')
    parser.add_argument('--max_predictions_per_seq', type=int, default=20,
                        help='Maximum number of masked LM predictions per sequence. '+\
                            'Must match data generation.')
    parser.add_argument('--do_train', action='store_true', 
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true', 
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Total batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=9,
                        help='Total batch size for eval.')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='The initial learning ratee for Adam')
    parser.add_argument('--num_train_steps', type=int, default=100_000,
                        help='Number of training steps.')
    parser.add_argument('--num_warmup_steps', type=int, default=10_000,
                        help='Number of warmup steps.')
    parser.add_argument('--eval_interval_steps', type=int, default=10_000,
                        help='Number of steps between evaluations.')
    parser.add_argument('--save_checkpoints_steps', type=int, default=10_000,
                        help='How often to save the model checkpoint.')
    parser.add_argument('--iterations_per_loop', type=int, default=1_000,
                        help='How many steps to make in each estimator call.')
    parser.add_argument('--max_eval_steps', type=int, default=100,
                        help='Maximum number of eval steps.')
    parser.add_argument('--use_tpu', action='store_true',
                        help='Whether to use TPU or GPU/CPU.')
    
    parser.add_argument('--tpu_zone', type=str, default=None,
                        help="GCE zone where the Cloud TPU is located in. If not "+\
                            "specified, we will attempt to automatically detect the GCE project from "+\
                            "metadata.")
    parser.add_argument('--gcp_project', type=str, default=None,
                        help="Project name for the Cloud TPU-enabled project. If not "+\
                            "specified, we will attempt to automatically detect the GCE project from "+\
                            "metadata.")
    parser.add_argument('--master', type=str, default=None,
                        help='TensorFlow master URL.')
    parser.add_argument('--num_tpu_cores', type=int, default=8,
                        help='Only used if `use_tpu` is True. Total number of TPU cores to use.')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = 'output'
        
    return args