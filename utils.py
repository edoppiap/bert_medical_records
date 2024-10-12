import logging 
import traceback
import sys
import os

import torch

def setup_logging(output_folder: str, console: str = 'debug',
                  info_filename:str = 'info.log', debug_filename:str = 'debug.log'):
    
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)
    
    if info_filename is not None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)
        
    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)
        
    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)
        
    def my_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
        logging.info("Experiment finished (with some errors)")
    sys.excepthook = my_handler
    
def save_checkpoint(state:dict, is_best:bool, output_folder:str, ckpt_filename: str = 'last_checkpoint.pth'):
    checkpoint_path = os.path.join(output_folder,ckpt_filename)
    torch.save(state, checkpoint_path)
    
    