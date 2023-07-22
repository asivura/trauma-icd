import os
import json
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer)
import logging
logger = logging.getLogger(__name__)

def load_tokenizer(model_name, model_dir, experiment_name):

    path = os.path.join(model_dir, model_name)
    if not os.path.exists(os.path.join(path, 'tokenizer.json')):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path)

    return tokenizer


def load_model(model_name, model_dir, experiment_name):
    
    path = os.path.join(model_dir, model_name, experiment_name)
    trainer_state_path = os.path.join(path, 'trainer_state.json')
    if not os.path.exists(trainer_state_path):
        logger.info(f"Initializing new model because no existing trainer_state.json was found at: {path}...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        model.save_pretrained(path)
    else:
        logger.info(f"Found trainer_state.json...")
        with open (trainer_state_path) as f:
            trainer_state = json.load(f)
        file_paths = ", ".join([str(p) for p in os.listdir(path)])
        logger.info(f"Found the following files: {file_paths}")
        check_path = input(f"Please select the checkpoint directory under {path} (e.g., type one of {[x for x in os.listdir(path) if 'checkpoint' in x]})")
        best_checkpoint_dir = os.path.join(path, check_path, "")
        best_metric = trainer_state["best_metric"]
        logger.info(f"Loading checkpoint at: {best_checkpoint_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(
            best_checkpoint_dir,
            num_labels=2
        )

    return model