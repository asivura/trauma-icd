import pandas as pd
import os
import datasets
import random

import logging
logger = logging.getLogger(__name__)

def load_ds(data_dir, is_pretrain=False):
    cases = pd.read_csv(os.path.join(data_dir, 'case.csv'))
    cases["context"] = cases.apply (lambda row: f'{row["tertiary_impression"]} \n\n {row["tertiary_imaging_report"]}', axis=1)
    
    if not is_pretrain:
        valid_labels_fn = 'label-non-superficial.txt'
        case_labels_path = os.path.join(data_dir, 'case-labels.csv')
        logger.info(f"Not in pretrain mode, so loading 4-character ICD-10 labels and dataset...")
    else:
        valid_labels_fn = 'label-non-superficial-5-char.txt'
        case_labels_path = os.path.join(data_dir, 'case-labels-5-char.csv')
        logger.info("Currently in pretrain mode, so loading 5-character ICD-10 labels and dataset...")
    
    valid_labels_path = os.path.join(data_dir, valid_labels_fn)
    case_labels = pd.read_csv(case_labels_path)

    with open(valid_labels_path) as f:
        valid_lables=[x.strip('\n') for x in f.readlines()]

    logger.info(f"{len(case_labels)} cases are read from {case_labels_path}")
    logger.info(f"{len(valid_lables)} valid labels are read from {valid_labels_path}")

    patient_ids = dict()
    for ds_name in ['train', 'validation', 'test']:
        with open(os.path.join(data_dir, f'{ds_name}.txt')) as f:
            ids=[x.strip('\n') for x in f.readlines()]
        patient_ids[ds_name] = ids
        
    case_labels = case_labels[case_labels.label.isin(valid_lables)]
    cases = cases[cases.patient_id.isin(case_labels.patient_id)]
    cases = cases.merge(case_labels.groupby('patient_id', as_index=False).agg({'label': lambda x: sorted(x)}))
    cases.drop(['tertiary_exam', 'tertiary_imaging_report', 'tertiary_impression', 'total_text_len'], axis=1, inplace=True)

    label_names = case_labels.drop_duplicates(['label']).drop('patient_id', axis=1).copy()
    label_names = dict(zip(label_names.label, label_names.label_name))
    ds = datasets.DatasetDict()
    for ds_name in ['train', 'validation', 'test']:
        ds[ds_name] = datasets.Dataset.from_pandas(cases[cases.patient_id.isin(patient_ids[ds_name])])

    return ds, label_names


def generate_pairs_ds(ds_dict, label_names, data_dir, is_pretrain=False):
    res = datasets.DatasetDict()
    for split, ds in ds_dict.items():
        pair_ds = {'patient_id': [], 'context': [], 'icd_code':[], 'icd_name': [], 'label': []}
        num_examples = 0
        for row in ds:
            row_labels = set(row['label'])
            for code, name in label_names.items():
                pair_ds['patient_id'].append(row['patient_id'])
                pair_ds['context'].append(row['context'])
                pair_ds['icd_code'].append(code)
                pair_ds['icd_name'].append(name)
                pair_ds['label'].append(int(code in row_labels))
                num_examples += 1
        res[split] = datasets.Dataset.from_dict(pair_ds)
        logger.info(f"The {split} dataset has {num_examples} context-label pairs.")
    
    res['reduced_validation'] = generate_reduced_pairs_ds(ds_dict['validation'], label_names, data_dir)
    
    return res

def generate_reduced_pairs_ds(ds, label_names, data_dir):

    label_sim_path = os.path.join(data_dir, 'icd-name-davinci-001-simularity-scores.csv')
    logger.info(f"Generating reduced validation pairs based on GPT-3 embeddings at {label_sim_path}")
    label_sim = pd.read_csv(label_sim_path)
    label_sim = label_sim[label_sim.label_2.isin(label_names)]
    label_sim = label_sim.sort_values(['label_1', 'sim'], ascending=[True, False])
    top5_sim_labels = dict()
    for code, df in label_sim.groupby('label_1'):
        # top5_sim_labels[code] = df.sort_values('sim', ascending=False)[:5].label_2.to_list()
        top5_sim_labels[code] = df.sort_values('sim', ascending=False).label_2.unique().tolist()[:5]
    
    pair_ds = {'patient_id': [], 'context': [], 'icd_code':[], 'icd_name': [], 'label': []}
    random.seed(42)
    for row in ds:
        row_labels = set(row['label'])
        neg_labels = []
        for label in row_labels:
            neg_labels += top5_sim_labels[label]
        neg_labels = set(neg_labels)
        if len(neg_labels) < 30:
            neg_labels = neg_labels.union(random.sample(set(label_names.keys())-row_labels-neg_labels, 30-len(neg_labels)))
        for code in row_labels.union(neg_labels):
            pair_ds['patient_id'].append(row['patient_id'])
            pair_ds['context'].append(row['context'])
            pair_ds['icd_code'].append(code)
            pair_ds['icd_name'].append(label_names[code])
            pair_ds['label'].append(int(code in row_labels))
    logger.info(f"Finished generating reduced validation pairs. It contains {len(pair_ds['label'])} context-label pairs.")
    return datasets.Dataset.from_dict(pair_ds)



def encode_ds(ds, tokenizer, max_seq_length):
    def encode(examples):
        res = tokenizer(examples['icd_name'], 
                        examples['context'], 
                        truncation='only_second', max_length=max_seq_length)

        for col in examples.keys():
            res[col] = examples[col]
        
        return res
    logger.info(f"Keys of ds: {ds.keys()}, len: {len(ds)}")
    ds_encoded = ds.map(encode, batched=True)
    ds_encoded.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    return ds_encoded

