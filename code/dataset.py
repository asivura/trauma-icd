import pandas as pd
import os
import datasets
import random

import logging

logger = logging.getLogger(__name__)


def load_ds(data_dir, valid_labels, train_on_full=False, read_cases=True, cases=None):
    
    cases = pd.read_csv(os.path.join(data_dir, "case.csv"))
    cases["context"] = cases.apply(
        lambda row: f'{row["tertiary_impression"]} \n\n {row["tertiary_imaging_report"]}',
        axis=1,
    )
    logger.info("Loading dataaset...")
    if valid_labels == "4-char":
        valid_labels_fn = "label-non-superficial.txt"  # txt file name where each line is a diagnosis code.
        case_labels_path = os.path.join(data_dir, "case-labels.csv")
        logger.info(f"Valid labels are: 4-character ICD-10 codes.")
    elif valid_labels == "4-char-with-superficial":
        valid_labels_fn = "label-with-superficial.txt"
        case_labels_path = os.path.join(data_dir, "case-labels-with-superficial.csv")
        logger.info(f"Valid labels are: 4-character ICD-10 codes including superficial codes.")
    elif valid_labels == "4-char-top10":
        valid_labels_fn = "label-non-superficial-top10.txt"
        case_labels_path = os.path.join(data_dir, "case-labels.csv")
        logger.info("Valid labels are: Top 10 4-character ICD-10 codes.")
    elif valid_labels == "4-char-top50":
        valid_labels_fn = "label-non-superficial-top50.txt"
        case_labels_path = os.path.join(data_dir, "case-labels.csv")
        logger.info("Valid labels are: Top 50 4-character ICD-10 codes.")
    elif valid_labels == "5-char":
        valid_labels_fn = "label-non-superficial-5-char.txt"
        case_labels_path = os.path.join(data_dir, "case-labels-5-char.csv")
        logger.info("Valid labels are: 5-character ICD-10 codes.")
    elif valid_labels == "4-and-5-char":
        valid_labels_fn = "label-non-superficial-4-and-5-char.txt"
        case_labels_path = os.path.join(data_dir, "case-labels-4-and-5-char.csv")
        logger.info("Valid labels are: both 4-character and 5-character ICD-10 codes.")
    else:
        raise ValueError(
            "Please set valid_labels as one of"
            " ['4-char', '4-char-top50', '5-char', '4-and-5-char']"
            )
    valid_labels_path = os.path.join(data_dir, valid_labels_fn)
    case_labels = pd.read_csv(case_labels_path)

    with open(valid_labels_path) as f:
        valid_lables = [x.strip("\n") for x in f.readlines()]

    logger.info(f"{len(case_labels)} cases were read from: {case_labels_path}")
    logger.info(f"{len(valid_lables)} valid labels were read from: {valid_labels_path} : {str(valid_lables)}")

    patient_ids = dict()
    if train_on_full:
        for ds_name in ["train_and_validation", "test"]:
            with open(os.path.join(data_dir, f"{ds_name}.txt")) as f:
                ids = [x.strip("\n") for x in f.readlines()]
            patient_ids[ds_name] = ids
        patient_ids["eval"] = patient_ids.pop("test")
        patient_ids["train"] = patient_ids.pop("train_and_validation")  #
        logger.info(
            f"Using the full training data and will evaluate on holdout test set patients."
        )
    else:
        for ds_name in ["train", "validation"]:
            with open(os.path.join(data_dir, f"{ds_name}.txt")) as f:
                ids = [x.strip("\n") for x in f.readlines()]
            patient_ids[ds_name] = ids
        patient_ids["eval"] = patient_ids.pop("validation")
        logger.info(
            f"Using the subset training data and will evaluate on validation set patients."
        )
    for split, pids in patient_ids.items():
        logger.info(f"The {split} split has {len(pids)} patient IDs.")

    case_labels = case_labels[case_labels.label.isin(valid_lables)]
    logger.info(f"The valid case_labels have shape {case_labels.shape}")
    cases = cases[cases.patient_id.isin(case_labels.patient_id)]
    cases = cases.merge(
        case_labels.groupby("patient_id", as_index=False).agg(
            {"label": lambda x: sorted(x)}
        )
    )
    cases.drop(
        [
            "tertiary_exam",
            "tertiary_imaging_report",
            "tertiary_impression",
            "total_text_len",
        ],
        axis=1,
        inplace=True,
    )

    label_names = (
        case_labels.drop_duplicates(["label"]).drop("patient_id", axis=1).copy()
    )
    label_names = dict(zip(label_names.label, label_names.label_name))
    ds = datasets.DatasetDict()
    # for ds_name in ['train', 'validation', 'train_and_validation', 'test']:
    for ds_name in ["train", "eval"]:
        ds[ds_name] = datasets.Dataset.from_pandas(
            cases[cases.patient_id.isin(patient_ids[ds_name])]
        )

    return ds, label_names


def generate_pairs_ds(ds_dict, label_names, data_dir, generate_reduced_pairs=True):
    res = datasets.DatasetDict()
    for split, ds in ds_dict.items():
        pair_ds = {
            "patient_id": [],
            "context": [],
            "icd_code": [],
            "icd_name": [],
            "label": [],
        }
        num_examples = 0
        for row in ds:
            row_labels = set(row["label"])
            for code, name in label_names.items():
                pair_ds["patient_id"].append(row["patient_id"])
                pair_ds["context"].append(row["context"])
                pair_ds["icd_code"].append(code)
                pair_ds["icd_name"].append(name)
                pair_ds["label"].append(int(code in row_labels))
                num_examples += 1
        res[split] = datasets.Dataset.from_dict(pair_ds)
        logger.info(f"The {split} dataset has {num_examples} context-label pairs.")

    if generate_reduced_pairs:
        res["reduced_eval"] = generate_reduced_pairs_ds(
            ds_dict["eval"], label_names, data_dir
        )

    return res


def generate_reduced_pairs_ds(ds, label_names, data_dir):

    label_sim_path = os.path.join(
        data_dir, "icd-name-davinci-001-simularity-scores.csv"
    )
    logger.info(
        f"Generating reduced validation pairs based on GPT-3 embeddings at: {label_sim_path}"
    )
    label_sim = pd.read_csv(label_sim_path)
    label_sim = label_sim[label_sim.label_2.isin(label_names)]
    label_sim = label_sim.sort_values(["label_1", "sim"], ascending=[True, False])
    top5_sim_labels = dict()
    for code, df in label_sim.groupby("label_1"):
        # top5_sim_labels[code] = df.sort_values('sim', ascending=False)[:5].label_2.to_list()
        top5_sim_labels[code] = (
            df.sort_values("sim", ascending=False).label_2.unique().tolist()[:5]
        )

    pair_ds = {
        "patient_id": [],
        "context": [],
        "icd_code": [],
        "icd_name": [],
        "label": [],
    }
    random.seed(42)
    for row in ds:
        row_labels = set(row["label"])
        neg_labels = []
        for label in row_labels:
            neg_labels = top5_sim_labels[label]
        neg_labels = set(neg_labels)
        if len(neg_labels) < 30:
            other_labels = set(label_names.keys()) - row_labels - neg_labels
            neg_labels = neg_labels.union(
                random.sample(
                    other_labels,
                    min(30 - len(neg_labels), len(other_labels)),
                )
            )
        for code in row_labels.union(neg_labels):
            pair_ds["patient_id"].append(row["patient_id"])
            pair_ds["context"].append(row["context"])
            pair_ds["icd_code"].append(code)
            pair_ds["icd_name"].append(label_names[code])
            pair_ds["label"].append(int(code in row_labels))
    logger.info(
        f"Finished generating reduced validation pairs. It contains {len(pair_ds['label'])} context-label pairs."
    )
    return datasets.Dataset.from_dict(pair_ds)


def encode_ds(ds, tokenizer, max_seq_length):
    def encode(examples):
        encoded = tokenizer(
            examples["icd_name"],
            examples["context"],
            truncation="only_second",
            max_length=max_seq_length,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        encoded['label'] = examples['label']
        for col in examples.keys():
            encoded[col] = examples[col]
        return encoded

    def print_columns(ds_encoded):
        print("Columns in the dataset:")
        for column in ds_encoded.column_names:
            print(column)


    print("Encoding dataset right now...")
    ds_encoded = ds.map(encode, batched=True)
    ds_encoded.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label", "patient_id", "icd_code", "icd_name", "context"]
    )
    print_columns(ds_encoded)
    print_columns(ds_encoded["reduced_eval"])
    

    return ds_encoded
