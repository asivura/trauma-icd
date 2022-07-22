import os
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer)
import logging
logger = logging.getLogger(__name__)

def load_tokenizer(model_name, model_dir):

    path = os.path.join(model_dir, model_name)
    if not os.path.exists(os.path.join(path, 'tokenizer.json')):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)

    return tokenizer


def load_model(model_name, model_dir):
    
    path = os.path.join(model_dir, model_name)
    if not os.path.exists(os.path.join(path, 'config.json')):
        logger.info(f"No pre-trained model was found at model_dir/model_name, so initializing new model to {path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2)
        model.save_pretrained(path)
    else:
        logger.info(f"Found pre-trained model config file at model_dir/model_name, so loading model checkpoint from {path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=2)

    return model