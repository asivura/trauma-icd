from dataset import load_ds, generate_pairs_ds, encode_ds
from model import load_tokenizer, load_model
from evaluate import compute_metrics

from dataclasses import dataclass, field, asdict
from typing import Optional, List

import numpy as np
import sys
import os
import pickle
import logging

logger = logging.getLogger(__name__)

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments
)

import transformers
from torch.utils.data import DataLoader, WeightedRandomSampler  
from scipy.special import softmax
from scipy.special import expit as sigmoid

import datasets

import shutil
import json

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    
    model_name: str = field(
        metadata={"help": "Model path from /models or model identifier from huggingface.co/models"}
    )

    experiment_name: str = field(
        metadata={"help": "The experiment name. Use as suffix for output paths"},
    )
    
    valid_labels: str = field(
        metadata={"help": "One of: ['5-char', '4-char', '4-char-top50']"},
    )

    model_dir: Optional[str] = field(
        default='/content/gdrive/Shareddrives/cs224n-final-project/models',
        metadata={"help": "Full path to /models directory"},
    )

    # cache_dir: str = field(
    #     default='/content/tmp',
    #     metadata={"help": (
    #         "The directory to save checkpoints during training."
    #         "We copy this direcotry to model_dir after the training"
    #         "to avoid unnessary syncrnization with Google Drive for file during training."),
    #     },
    # )  

    data_dir: Optional[str] = field(
        default='/content/gdrive/Shareddrives/cs224n-final-project/injury-icd-dataset',
        metadata={"help": "Full path to /injury-icd-dataset directory"},
    )



@dataclass
class TraumalIcdTrainingArguments:

    per_device_train_batch_size: str = field(
        default=16,
        metadata={"help": "The batch size per GPU/TPU core/CPU for training."},
    )

    per_device_eval_batch_size: int = field(
        default=32,
        metadata={"help": "The batch size per GPU/TPU core/CPU for evaluation."}
    )

    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for AdamW."})
    num_train_epochs: float = field(default=20.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=1000, metadata={"help": "Linear warmup over warmup_steps."})
    metric_for_best_model: str = field(default="eval_f1_score_weighted", metadata={"help": "The metric for selecting the best model for final evaluation."})
    train_on_full: bool = field(default=False, metadata={"help": "Whether train on both the train and validation splits, and eval on holdout test split."})
    is_evaluate: bool = field(default=False, metadata={"help": "Whether to just evaluate a previously saved checkpoint (and not train anything)."})

    log_level: Optional[str] = field(
        default=logging.INFO,
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', "
                "'info', 'warning', 'error' and 'critical' Defaults to 'passive'."),
            "choices": transformers.utils.logging.get_log_levels_dict(),
        },
    )



class TraumalIcdTrainer(Trainer):
    def __init__(self, model, tokenizer, output_dir, icd_codes, train_ds=None, eval_dataset=None, traumal_icd_training_args=None, logging_dir=None, compute_metrics=None):
        if traumal_icd_training_args is None:
            traumal_icd_training_args = TraumalIcdTrainingArguments()

        training_args = TrainingArguments(
            output_dir=output_dir, 
            per_device_train_batch_size=int(traumal_icd_training_args.per_device_train_batch_size),
            per_device_eval_batch_size=int(traumal_icd_training_args.per_device_eval_batch_size),
            evaluation_strategy="epoch",
            metric_for_best_model=traumal_icd_training_args.metric_for_best_model,
            greater_is_better=True,
            logging_strategy='no',
            save_strategy='epoch',
            log_level=logging.getLevelName(traumal_icd_training_args.log_level).lower(),
            num_train_epochs=traumal_icd_training_args.num_train_epochs,
            save_total_limit=1,
            overwrite_output_dir=True,
            optim='adamw_torch',
            learning_rate=traumal_icd_training_args.learning_rate,
            lr_scheduler_type='linear',
            warmup_steps=traumal_icd_training_args.warmup_steps,
            logging_dir=logging_dir,
            disable_tqdm=True,
            load_best_model_at_end=True,
            include_inputs_for_metrics=True,
            report_to="none"
            )


        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )

        self.icd_codes = icd_codes
        self.output_dir = output_dir

    def evaluate(self, eval_dataset = None, ignore_keys = None, metric_key_prefix = "eval"):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        self.save_state()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        logger.info("Saved trainer state and performing evaluation...")
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        label_pos = dict((x,y) for y, x in enumerate(self.icd_codes.keys()))
        patient_pos = dict((x,y) for y, x in enumerate(list(set(eval_dataset['patient_id']))))
        predictions = softmax(output.predictions, axis=1)[:, 1]
        # predictions = softmax(output.predictions, axis=1)
        # predictions = sigmoid(output.predictions)
        targets = np.zeros((len(patient_pos), len(label_pos)))
        probs = np.zeros((len(patient_pos), len(label_pos)))
        logger.info(f"Shape of predictions {predictions.shape}")
        logger.info(f"Shape of targets {targets.shape}")
        logger.info(f"Shape of probs {probs.shape}")
        # for i, row in enumerate(eval_dataset):
        #     assert row['label'] == output.label_ids[i]
        #     targets[patient_pos[row['patient_id']], label_pos[row['icd_code']]] = row['label']
        #     probs[patient_pos[row['patient_id']], label_pos[row['icd_code']]] = predictions[i]

        for i, row in enumerate(eval_dataset):
        #   for j, icd_code in enumerate(row['icd_code']): 
            icd_code = row['icd_code']
            targets[patient_pos[row['patient_id']], label_pos[icd_code]] = row['label']
            # probs[patient_pos[row['patient_id']], label_pos[icd_code]] = predictions[i, j]
            probs[patient_pos[row['patient_id']], label_pos[icd_code]] = predictions[i]
            # probs[patient_pos[row['patient_id']], label_pos[icd_code]] = predictions[i, 1]
        output.metrics.update(compute_metrics(targets, probs, prefix=metric_key_prefix))
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        eval_pickle_path = os.path.join(self.output_dir, "eval_results.pickle")
        eval_results = {"metrics": output.metrics, "probs": probs, "targets": targets, "patient_pos": patient_pos, "label_pos": label_pos}
        with open(eval_pickle_path, 'wb') as f:
            pickle.dump(eval_results, f)
        logger.info(f"The evaluation results have been saved to to: {eval_pickle_path}")

        return output.metrics
    


    def get_train_dataloader(self) -> DataLoader:
        train_ds = self.train_dataset
        data_collator = self.data_collator
        train_ds = self._remove_unused_columns(train_ds, description="training")

        n_examples = len(train_ds)
        n_pos_examples = sum(train_ds['label'])
        n_neg_examples = n_examples - n_pos_examples
        neg_weight = n_pos_examples/n_neg_examples
        pos_weight = 1
        logger.info(f"There are {n_pos_examples} positive and {n_neg_examples} negative examples in training set.")
        weights = list(map(lambda x: pos_weight if x==1 else neg_weight, train_ds['label']))
        sampler = WeightedRandomSampler(weights, int(n_pos_examples*2), replacement=False)
        logger.info(f"Using weighted sampling: positive weight = {pos_weight:.2f} , negative weight = {neg_weight:.2f}")
        logger.info(f"We will draw {int(n_pos_examples*2)} context-label pairs during one epoch.")
        
        return DataLoader(
            train_ds,
            sampler=sampler,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def save_args(output_dir, model_args, traumal_icd_training_args, training_args):
    all_args = {
        'model_args': asdict(model_args),
        'traumal_icd_training_args': asdict(traumal_icd_training_args),
        'traning_args': training_args.to_dict()
    }

    with open(os.path.join(output_dir, 'all_args.json'), "w") as f:
        json.dump(all_args, f, indent=4, sort_keys=True)


def main():
    parser = HfArgumentParser((ModelArguments, TraumalIcdTrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.NOTSET
    )
    
    ds_path = os.path.join(model_args.model_dir, model_args.model_name, model_args.experiment_name, 'encoded_ds')
    ds, label_names = load_ds(model_args.data_dir, model_args.valid_labels, training_args.train_on_full)
    logger.info(f"Loaded dataset with size {ds.shape}")
    
    logger.info(f'Loading base model and tokenizer: {model_args.model_name}...')
    tokenizer = load_tokenizer(model_args.model_name, model_args.model_dir, model_args.experiment_name)
    model = load_model(model_args.model_name, model_args.model_dir, model_args.experiment_name)

    if not os.path.exists(ds_path):
        logger.info(f'No tokenized dataset exist yet, so generating context-label pairs...')
        pairs_ds = generate_pairs_ds(ds, label_names, model_args.data_dir)
        logger.info(f'Finished generating context-label pairs, now tokenizing them into dataset...')
        ds_encoded = encode_ds(pairs_ds, tokenizer, model.config.max_position_embeddings)
        ds_encoded.save_to_disk(ds_path)
        logger.info(f"Saved tokenized dataset to {ds_path}")
    else:
        logger.info(f"Loading existing tokenized dataset...")

    ds_encoded = datasets.load_from_disk(ds_path)
    logger.info(f"Loaded tokenized dataset from {ds_path}")
    
    logging_dir = os.path.join(model_args.model_dir, 'runs', model_args.model_name, 
        model_args.experiment_name)

    output_dir = os.path.join(model_args.model_dir, model_args.model_name, 
        model_args.experiment_name)
   
    output_model_dir = os.path.join(model_args.model_dir,
                                model_args.model_name, 
                                model_args.experiment_name)
    

    eval_ds_reduced = ds_encoded['reduced_eval']
    eval_ds_full = ds_encoded['eval']

    trainer = TraumalIcdTrainer(model, tokenizer, output_dir, label_names,
                            train_ds=ds_encoded['train'],
                            eval_dataset=eval_ds_reduced,
                            # eval_dataset=eval_ds_full,
                            logging_dir=logging_dir,
                            traumal_icd_training_args=training_args)

    if not training_args.is_evaluate:
        logger.info(f'Training...')
        trainer.train()
    
        trainer.save_state()
        save_args(output_dir, model_args, training_args, trainer.args)  
        
    if not training_args.is_evaluate:
        logger.info(f'Saving...')
        if os.path.exists(output_model_dir):
            shutil.rmtree(output_model_dir)
        shutil.copytree(output_dir, output_model_dir) 
    
    logger.info('Evaluating on the full test/validation dataset...')
    validation_metrics = trainer.evaluate(eval_ds_full) #, eval_args={"model_args": model_args, "training_args": training_args})
    logger.info(validation_metrics)
    with open(os.path.join(output_dir, 'best_scores.json'), 'w') as f:
        json.dump(validation_metrics, f)
        

if __name__ == "__main__":
    main()