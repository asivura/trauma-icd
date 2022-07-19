import os
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer)

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
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2)
        model.save_pretrained(path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=2)

    return model